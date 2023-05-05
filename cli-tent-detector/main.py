import os
import argparse
import torch
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.modules.loss import BCEWithLogitsLoss
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import BinaryF1Score
from application.config import IFormat
from application.dataset import SegmentationDataset
from application.evaluator import evaluator
from application.operator import Operator
from application.unet import UNet
from application.utils import load_data
from application.visualizations import save_data

ROOT_DIR = Path(__file__).parent
MODEL_PATH = [mp for mp in (ROOT_DIR / 'models').glob(f'*.{IFormat.model}')][0]

# Default Training Settings #
EPOCHS = 200
BATCH_SIZE = 8
INIT_LR = 0.0001
TEST_SPLIT = 0.2
RANDOM_STATE = 42
#############################

IMAGE_SIZE = (512, 512) # The model expects tiles to be this size
DEVICE = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
PIN_MEMORY = True if DEVICE != 'cpu' else False

def main(x_dir: Path, output_dir: Path, checkpoint_path: Path, training: bool=False, **kwargs):
    p_output_path = output_dir / 'predictions'
    y_dir = kwargs.get('y_dir')

    if training and not y_dir:
        raise Exception('Cannot train without binary masks!')

    print(f'Using device: {DEVICE}')

    if DEVICE == 'cpu' and training:
        print('Warning: Training on CPU. This might be exceptionally slow.')

    operator = None
    if checkpoint_path:
        model = None
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        try:
            checkpoint.eval()
        except AttributeError:
            print('Loading checkpoint')
            model = kwargs.get('model', UNet())
            model.load_state_dict(checkpoint)
        else:
            print('Loading model')
            model = checkpoint

        operator = Operator(p_output_path,
                            model,
                            BCEWithLogitsLoss(),
                            Adam,
                            kwargs.get('lr', INIT_LR),
                            [BinaryF1Score(), BinaryJaccardIndex()],
                            device=DEVICE)
    elif training:
        print('No model path specified, creating a new model.')
        operator = Operator(p_output_path,
                            kwargs.get('model', UNet()),
                            BCEWithLogitsLoss(),
                            Adam,
                            kwargs.get('lr', INIT_LR),
                            [BinaryF1Score(), BinaryJaccardIndex()],
                            device=DEVICE)
    else:
        raise Exception('Cannot use an untrained model for inference!')

    transformations = transforms.Compose([transforms.ToPILImage(), 
                                          transforms.Resize(kwargs.get('image_size', IMAGE_SIZE), antialias=True), 
                                          transforms.ToTensor()])

    results = None
    if y_dir:
        y_data = load_data(x_dir, y_dir, kwargs.get('labels_path'))
        t_loader = v_loader = None
        if training:
            training_data, validation_data = train_test_split(
                y_data, test_size=kwargs.get('split', TEST_SPLIT), random_state=kwargs.get('random_state', RANDOM_STATE))
            training_dataset = SegmentationDataset(training_data, transformations)
            validation_dataset = SegmentationDataset(
                validation_data, transformations)

            t_loader = DataLoader(training_dataset, shuffle=True, batch_size=kwargs.get('batch', BATCH_SIZE),
                                pin_memory=PIN_MEMORY, num_workers=os.cpu_count(), persistent_workers=True)
            v_loader = DataLoader(validation_dataset, shuffle=True, batch_size=kwargs.get('batch', BATCH_SIZE),
                                pin_memory=PIN_MEMORY, num_workers=os.cpu_count(), persistent_workers=True)

            t_count = len(training_data)
            v_count = len(validation_data)

            t_ratio = 1-v_count/t_count
            v_ratio = v_count/t_count

            print(f'Training to Validation Ratio\n')
            print(f'Training ({t_count}): \t{t_ratio*100:>10.2f}%')
            print(f'Validation ({v_count}): \t{v_ratio*100:>10.2f}%')
            print(f'Total ({t_count + v_count}): \t\t{t_ratio*100 + v_ratio*100:>10.2f}%\n')

            assert t_ratio + v_ratio == 1  # Sanity Check

        y_paths = save_data(output_dir / 'truths', y_data, transformations, tqdm, heatmap_title='Actual Tents per Region')
        results = evaluator(p_output_path, y_data, [operator], transformations, x_overview_path=y_paths['x_overview_path'], 
                            t_loader=t_loader, v_loader=v_loader, epochs=kwargs.get('epochs', EPOCHS))
    else:
        results = evaluator(p_output_path, load_data(x_dir, None, None), [operator], transformations)


    print(f'Finished. Results:\n{results}') # TODO: tell user which directory to look in for the results (also return the directory, so we can pipe this to another program)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect tents in satellite imagery.')
    parser.add_argument('input', type=Path, help='Specifies the input directory.')
    parser.add_argument('output', type=Path, help='Specifies the output directory.')
    parser.add_argument('--checkpoint', '-c', type=Path, default=MODEL_PATH, help='Specifies the model checkpoint path. If not specified a new model will be created.')
    parser.add_argument('--image_size', type=tuple[int,int], default=IMAGE_SIZE, help='Sets the image size.')
    parser.add_argument('--train', '-t', action='store_true', help='Puts the model in training mode.')
    parser.add_argument('--epochs', '-e', type=int, default=EPOCHS, help='Sets the number of epochs. Only relevant when -t is also set.')
    parser.add_argument('--batch', '-b', type=int, default=BATCH_SIZE, help='Sets the batch size. Only relevant when -t is also set.')
    parser.add_argument('--split', '-s', type=float, choices=range(0,1), default=TEST_SPLIT, help='Sets the test split. Only relevant when -t is also set.')
    parser.add_argument('--rate', '-r', type=float, default=INIT_LR, help='Sets the learning rate. Only relevant when -t is also set.')
    parser.add_argument('--random_state', type=int, default=RANDOM_STATE, help='Sets the random state. Only relevant when -t is also set.')
    

    # parser.add_argument('--verbose', '-v', action='store_true', help='Specifies whether or not to be verbose.')


    args = parser.parse_args()


    # Default Data
    # data_dir = ROOT_DIR / 'data' / 'sarpol-zahab-tents' / 'data'
    # main(data_dir / 'images', ROOT_DIR / 'output', MODEL_PATHS[0], labels_path=data_dir / 'sarpol_counts.csv', y_dir=data_dir / 'labels')

    main(args.input, args.output, args.checkpoint, args.train, epochs=args.epochs, batch=args.batch, 
         split=args.split, lr=args.rate, random_state=args.random_state, image_size=args.image_size)
