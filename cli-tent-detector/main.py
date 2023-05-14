import os
import argparse
import pandas as pd
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
from application.dataset import SegmentationDataset
from application.evaluator import evaluator
from application.operator import Operator
from application.unet import UNet
from application.config import IOFormat
from application.utils import load_data, Tiler
from application.visualizations import save_data

ROOT_DIR = Path(__file__).parent
MODEL_PATH = [mp for mp in (ROOT_DIR / 'models').glob(f'*.{IOFormat.model}')][0]
MODEL_TYPE = UNet()
EPOCHS = 200
BATCH_SIZE = 8
INIT_LR = 0.0001
TEST_SPLIT = 0.2
RANDOM_STATE = 42
KERNEL_SIZE = (512, 512) # The model expects tiles to be this size
DEVICE = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
PIN_MEMORY = True if DEVICE != 'cpu' else False


def main(x_dir: Path, output_dir: Path, checkpoint_path: Path | torch.nn.Module, training: bool=False, **kwargs) -> dict[str, dict[str, Path | None]]:
    model_type = kwargs.get('model', MODEL_TYPE)
    epochs = kwargs.get('epochs', EPOCHS)
    batch_size = kwargs.get('batch', BATCH_SIZE)
    lr = kwargs.get('lr', INIT_LR)
    split_size = kwargs.get('split', TEST_SPLIT)
    random_state = kwargs.get('random_state', RANDOM_STATE)
    kernel_size = kwargs.get('kernel_size', KERNEL_SIZE)
    device = kwargs.get('device', DEVICE)
    pin_memory = kwargs.get('pin_memory', PIN_MEMORY)
    verbose = kwargs.get('verbose', False)
    pbar = tqdm if verbose else None
    labels_path = kwargs.get('labels_path')
    y_dir = kwargs.get('y_dir')
    p_output_path = output_dir / 'predictions'
    
    if training and not y_dir:
        raise Exception('Cannot train without binary masks!')

    if verbose: print(f'Using device: {device}')

    if device == 'cpu' and training:
        if verbose: print('Warning: Training on CPU. This might be exceptionally slow.')

    operator = None
    
    if checkpoint_path:
        model = None
        checkpoint = None
        try:
            checkpoint_path.eval()
        except AttributeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            try:
                checkpoint.eval()
            except AttributeError:
                if verbose: print('Loading checkpoint')
                model = model_type
                model.load_state_dict(checkpoint)
            else:
                if verbose: print('Loading model')
                model = checkpoint
        else:
            if verbose: print('Using supplied model')
            model = checkpoint_path

        operator = Operator(p_output_path,
                            model,
                            BCEWithLogitsLoss(),
                            Adam,
                            lr,
                            [BinaryF1Score(), BinaryJaccardIndex()],
                            device=device)
    elif training:
        if kwargs.get('verbose', False): print('No model path specified, creating a new model.')
        operator = Operator(p_output_path,
                            model_type,
                            BCEWithLogitsLoss(),
                            Adam,
                            lr,
                            [BinaryF1Score(), BinaryJaccardIndex()],
                            device=device)
    else:
        raise Exception('Cannot use an untrained model for inference!')
    
    tiler = Tiler(kernel_size)
    transformations = transforms.Compose([transforms.ToPILImage(), 
                                          transforms.Resize(kernel_size, antialias=True), 
                                          transforms.ToTensor()])

    results = None
    if y_dir:
        y_data = load_data(x_dir, y_dir, labels_path)
        t_loader = v_loader = None
        if training:
            training_data, validation_data = train_test_split(
                y_data, test_size=split_size, random_state=random_state)
            training_dataset = SegmentationDataset(training_data, transformations)
            validation_dataset = SegmentationDataset(
                validation_data, transformations)

            t_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size,
                                pin_memory=pin_memory, num_workers=os.cpu_count(), persistent_workers=True)
            v_loader = DataLoader(validation_dataset, shuffle=True, batch_size=batch_size,
                                pin_memory=pin_memory, num_workers=os.cpu_count(), persistent_workers=True)

            t_count = len(training_data)
            v_count = len(validation_data)
            t_ratio = 1-v_count/t_count
            v_ratio = v_count/t_count
            if verbose:
                print(f'Training to Validation Ratio\n')
                print(f'Training ({t_count}): \t{t_ratio*100:>10.2f}%')
                print(f'Validation ({v_count}): \t{v_ratio*100:>10.2f}%')
                print(f'Total ({t_count + v_count}): \t\t{t_ratio*100 + v_ratio*100:>10.2f}%\n')
            assert t_ratio + v_ratio == 1  # Sanity Check

        y_paths = save_data(output_dir / 'truths', y_data, transformations, pbar, heatmap_title='Actual Tents per Region')
        results = evaluator(p_output_path, y_data, [operator], transformations, tiler, pbar, x_overview_path=y_paths['x_overview_path'], 
                            t_loader=t_loader, v_loader=v_loader, epochs=epochs)
    else:
        results = evaluator(p_output_path, pd.DataFrame({'x_paths': x_dir, 'y_paths': None}, index=[0]), [operator], transformations, tiler, pbar)

    if verbose: print(f'Results saved to {output_dir}')
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect tents in satellite imagery.')
    parser.add_argument('input', type=Path, help='Specifies the input directory.')
    parser.add_argument('output', type=Path, help='Specifies the output directory.')
    parser.add_argument('--checkpoint', '-c', type=Path, default=MODEL_PATH, help='Specifies the model checkpoint path. If not specified a new model will be created.')
    parser.add_argument('--kernel_size', type=tuple[int,int], default=KERNEL_SIZE, help='Sets the image size.')
    parser.add_argument('--train', '-t', action='store_true', help='Puts the model in training mode.')
    parser.add_argument('--epochs', '-e', type=int, default=EPOCHS, help='Sets the number of epochs. Only relevant when -t is also set.')
    parser.add_argument('--batch', '-b', type=int, default=BATCH_SIZE, help='Sets the batch size. Only relevant when -t is also set.')
    parser.add_argument('--split', '-s', type=float, choices=range(0,1), default=TEST_SPLIT, help='Sets the test split. Only relevant when -t is also set.')
    parser.add_argument('--rate', '-r', type=float, default=INIT_LR, help='Sets the learning rate. Only relevant when -t is also set.')
    parser.add_argument('--random_state', type=int, default=RANDOM_STATE, help='Sets the random state. Only relevant when -t is also set.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Specifies whether or not to be verbose.')

    args = parser.parse_args()

    main(args.input, args.output, args.checkpoint, args.train, epochs=args.epochs, batch=args.batch, 
         split=args.split, lr=args.rate, random_state=args.random_state, kernel_size=args.kernel_size, verbose=args.verbose)
    