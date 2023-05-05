import os
import math
import pandas as pd
import numpy as np
from enum import StrEnum
from pathlib import Path
from torch.utils.data import DataLoader
from application.config import IFormat, OFormat
from application.dataset import SegmentationDataset


def make_dirs(directories: dict[str, str]) -> StrEnum:
    for directory in directories.values():
        if not os.path.exists(directory):
            os.makedirs(directory)
    return StrEnum('Directories', directories)


def load_data(x_images_dir: str, y_masks_dir: str, csv_path: str, **kwargs) -> pd.DataFrame:
    if csv_path:
        return c_data(x_images_dir, y_masks_dir, csv_path, **kwargs)
    else:
        return d_data(x_images_dir, y_masks_dir, **kwargs)


def c_data(x_images_dir: str, y_masks_dir: str, csv_path: str, **kwargs) -> pd.DataFrame:
    return associate_data(x_images_dir, y_masks_dir, csv_data(csv_path, **kwargs), **kwargs)


def d_data(x_images_dir: str, y_masks_dir: str = None, **kwargs) -> pd.DataFrame:
    return associate_data(x_images_dir, y_masks_dir, directory_data(x_images_dir, kwargs.get('format', IFormat)), **kwargs)


def associate_data(x_images_dir: str, y_masks_dir: str, rows: list, **kwargs) -> pd.DataFrame:
    """
    Returns a dataframe with the feature and target paths, along with the image
    name and number of tents (i.e., labels).
    """
    if kwargs.get('assert_square', True):
        assert len(
            rows) % 2 == 0, f'Number of rows ({len(rows)}) are not even!'
    return pd.DataFrame({
        'names': [row[0].split('.')[0] for row in rows],
        'image_paths': [str(next(Path(x_images_dir).glob(row[0]))) if x_images_dir else None for row in rows],
        'mask_paths': [str(next(Path(y_masks_dir).glob(row[0]))) if y_masks_dir else None for row in rows],
        'labels': [int(row[1]) for row in rows]
    }).set_index('names').astype({'labels': 'int'})


def csv_data(csv_path, **kwargs) -> list[tuple[str, int]]:
    rows = pd.read_csv(csv_path, header=kwargs.get('header'))
    return list(rows.itertuples(index=False, name=None))


def directory_data(directory: str, format: IFormat) -> list[tuple[str, int]]:
    names = []
    for filepath in Path(directory).glob(f'*.{format.image}'):
        names.append((filepath.name, -1))
    return names
