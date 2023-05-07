import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from application.config import PathEnum


def make_dirs(directories: dict[str, Path]) -> PathEnum:
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return PathEnum('Directories', directories)


def load_data(x_images_dir: str, y_masks_dir: str, csv_path: str, **kwargs) -> pd.DataFrame:
    if csv_path:
        return c_data(x_images_dir, y_masks_dir, csv_path, **kwargs)
    else:
        return d_data(x_images_dir, y_masks_dir, **kwargs)


def c_data(x_images_dir: str, y_masks_dir: str, csv_path: str, **kwargs) -> pd.DataFrame:
    return associate_data(x_images_dir, y_masks_dir, csv_data(csv_path, **kwargs), **kwargs)


def d_data(x_images_dir: str, y_masks_dir: str = None, **kwargs) -> pd.DataFrame:
    return associate_data(x_images_dir, y_masks_dir, directory_data(x_images_dir), **kwargs)


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


def directory_data(directory: str) -> list[tuple[str, int]]:
    names = []
    for filepath in Path(directory).glob(f'*'):
        names.append((filepath.name, -1))
    return names


class Tiler():
    def __init__(self, kernel_size: tuple[int, int]) -> None:
        self.kernel_size = kernel_size

    def __tile(self, image: Image, output_dir: Path, kernel_size: tuple[int, int], output_format: str, **kwargs) -> list[Path]:
            # Adapted from https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
            output_dir.mkdir(parents=kwargs.get('make_tiles_parents', False), exist_ok=kwargs.get('overwrite_tiles_dir', False))

            image_height, image_width, image_channels = image.shape
            tile_height, tile_width = kernel_size

            # TODO: do tiles this way too (the inverse of this operation to merge tiles)
            tiles = image.reshape(image_height//tile_height, tile_height,
                                    image_width//tile_width, tile_width, image_channels).swapaxes(1, 2)

            image_paths = []

            index = 0
            for row in tiles:
                for column in row:
                    filename = output_dir / f'tile_{index}{output_format}'
                    Image.fromarray(column).save(filename)
                    image_paths.append(filename)
                    index += 1

            return image_paths
    
    def _merge(self, image: Image, output_dir: Path, kernel_size: tuple[int, int], output_format: str, **kwargs):
        pass

    def tile(self, xy_paths: pd.DataFrame, **kwargs) -> pd.DataFrame | None:
        assert len(xy_paths) == 1
        x_path = xy_paths['x_paths'][0]
        y_path = xy_paths['y_paths'][0]
        assert x_path.is_file()

        paths = {'x_paths': [], 'y_paths': []}
        with Image.open(xy_paths['x_paths'][0]).convert("RGB") as x:
            x = np.asarray(x)
        if x.shape[:2] > self.kernel_size:
            paths['x_paths'] = self.__tile(x, x_path.parent / 'x_tiles', self.kernel_size, x_path.suffix, **kwargs)
            if y_path:
                with Image.open(xy_paths['y_paths'][0]).convert("L") as y:
                    y = np.asarray(y)
                paths['y_paths'] = self.__tile(y, y_path.parent / 'y_tiles', self.kernel_size, y_path.suffix)
            else:
                paths['y_paths'] = [None for _ in paths['x_paths']]
            return pd.DataFrame.from_dict(paths).sort_values('x_paths', ignore_index=True)
        else:
            return None
        
    def merge(self, xy_tile_paths: pd.DataFrame):
        pass

        # TODO: untile this way (the inverse of this operation to merge tiles)
        # tiles = image.reshape(image_height//tile_height, tile_height, image_width//tile_width, tile_width, image_channels).swapaxes(1, 2)

        # Note that we might want kernel_size to be a class property, as we must untile with the same settings
        # TODO: we might just want to move tile and untile to a completely separate Tiler class; maybe we just pass the tiling class or function in when 
        # initializing the dataset?

