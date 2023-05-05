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
from application.config import IFormat, ModelSource
from application.dataset import SegmentationDataset
from application.evaluator import evaluator
from application.model import Operator
from application.unet import UNet
from application.utils import load_data
from application.visualizations import save_data

ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / 'models'
MODEL_PATHS = [mp for mp in (MODELS_DIR).glob(f'*.{IFormat.model}')]

DEVICE = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
PIN_MEMORY = True if DEVICE != 'cpu' else False

# TODO: move these settings to their own file?
IMAGE_SIZE = (512, 512)
N_EPOCHS = 200
BATCH_SIZE = 8
INIT_LR = 0.0001
TEST_SPLIT = 0.2
RANDOM_STATE = 42

MODEL_SOURCE = ModelSource.default

def main(x_dir: Path, output_dir: Path, checkpoint_path: Path, training: bool=False, **kwargs):
    p_output_path = output_dir / 'predictions'
    y_dir = kwargs.get('y_dir')

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
                            [BinaryF1Score(), BinaryJaccardIndex()],
                            device=DEVICE)
    elif training:
        print('No model path specified, creating a new model.')
        operator = Operator(p_output_path,
                            kwargs.get('model', UNet()),
                            BCEWithLogitsLoss(),
                            Adam,
                            [BinaryF1Score(), BinaryJaccardIndex()],
                            device=DEVICE)
    else:
        raise Exception('Cannot use an untrained model for inference!')

    transformations = transforms.Compose([transforms.ToPILImage(), 
                                          transforms.Resize(IMAGE_SIZE, antialias=True), 
                                          transforms.ToTensor()])

    results = None
    if y_dir:
        y_data = load_data(x_dir, y_dir, kwargs.get('labels_path'))
        t_loader = v_loader = None
        if training:
            training_data, validation_data = train_test_split(
                y_data, test_size=TEST_SPLIT, random_state=RANDOM_STATE)
            training_dataset = SegmentationDataset(training_data, transformations)
            validation_dataset = SegmentationDataset(
                validation_data, transformations)

            t_loader = DataLoader(training_dataset, shuffle=True, batch_size=BATCH_SIZE,
                                pin_memory=PIN_MEMORY, num_workers=os.cpu_count(), persistent_workers=True)
            v_loader = DataLoader(validation_dataset, shuffle=True, batch_size=BATCH_SIZE,
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
        results = evaluator(p_output_path, y_data, [operator], transformations, x_overview_path=y_paths['x_overview_path'], t_loader=t_loader, v_loader=v_loader, epochs=N_EPOCHS)
    else:
        results = evaluator(p_output_path, load_data(x_dir, None, None), [operator], transformations)


    print(f'Finished. Results:\n{results}') # TODO: tell user which directory to look in for the results (also return the directory, so we can pipe this to another program)
    return results


if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description='Detect tents in satellite imagery.')
    # parser.add_argument('input_path', type=str, help='Specifies the input directory.')
    # parser.add_argument('output_path', type=str, help='Specifies the output directory.')

    # parser.add_argument('--new', '-n', type=str, help='Creates and trains a new model.')
    # parser.add_argument('--verbose', '-v', type=bool, help='Specifies whether or not to be verbose.')

    # args = parser.parse_args()

    # print(args)
    
    # Default Data
    data_dir = ROOT_DIR / 'data' / 'sarpol-zahab-tents' / 'data'
    main(data_dir / 'images', ROOT_DIR / 'output', MODEL_PATHS[0], labels_path=data_dir / 'sarpol_counts.csv', y_dir=data_dir / 'labels')
