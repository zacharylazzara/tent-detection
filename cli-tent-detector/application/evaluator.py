import os
import math
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from application.dataset import SegmentationDataset
from application.operator import Operator
from application.visualizations import save_data


def evaluator(prediction_output_dir, data: pd.DataFrame, operators: list[Operator], transformations: transforms.Compose = None, pbar: tqdm = None, **kwargs) -> dict[str, dict[str, str | None]]:
    dataset = SegmentationDataset(data, transformations)
    row_size = int(math.sqrt(len(dataset)))
    if row_size < 1:
        raise Exception(
            "Dataset contains no data. Make sure you set the input format to the correct format!")
    loader = DataLoader(dataset,
                        shuffle=kwargs.get('shuffle', False),
                        batch_size=row_size,
                        pin_memory=kwargs.get(
                            'pin_memory', kwargs.get('pin_memory', False)),
                        num_workers=os.cpu_count(),
                        persistent_workers=kwargs.get('persistent_workers', True))

    paths = {}
    for operator in operators:
        if kwargs.get('t_loader') and kwargs.get('v_loader'):
            operator.train(kwargs.get('epochs', 1), kwargs.get('t_loader'), kwargs.get('v_loader'), pbar,
                        save_model=True, save_history=True)
        p_data = operator.predict(loader, pbar, save_predictions=True)[1].combine_first(data).sort_index()
        paths[str(operator)] = save_data(prediction_output_dir, p_data, transformations, pbar, y='p', heatmap_title='Detected Tents per Region', **kwargs)

    return paths
