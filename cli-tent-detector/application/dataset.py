import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Used for training the model."""
    # Adapted from: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

    def __init__(self, dataframe, transformations=None):
        self.dataframe = dataframe
        self.transformations = transformations

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, index):
        """
        If there's no mask path, generate a blank mask. This is for the case where
        we only want to perform predictions and as such haven't included any masks.
        Works as expected when mask paths are included.
        """
        
        image_paths = self.dataframe.iloc[index]['image_paths']
        mask_paths = self.dataframe.iloc[index]['mask_paths']
        image = cv2.cvtColor(cv2.imread(image_paths), cv2.COLOR_BGR2RGB)
        mask = None

        if mask_paths:
            mask = cv2.threshold(cv2.imread(
                mask_paths, cv2.IMREAD_GRAYSCALE), 150, 255, cv2.THRESH_BINARY)[1]
        else:
            mask = cv2.threshold(np.zeros(
                (image.shape[0], image.shape[1], 1), dtype=np.uint8), 150, 255, cv2.THRESH_BINARY)[1]
        if self.transformations:
            image = self.transformations(image)
            mask = self.transformations(mask)

        return image, mask, self.dataframe.iloc[index]['labels'], self.dataframe.index[index]


class PredictionDataset(Dataset):
    """Used for predictions (when not using default dataset?)."""
    # TODO: get this working properly and make sense; this should be used whenever we're predicting and work for all images not just the sarpol ones

    def __init__(self, image_path: Path, directory: Path, transformations=None, kernel_size: tuple[int, int] = (512, 512), **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        self.transformations = transformations
        self.image_paths = self.__tile(
            image_path, directory, kernel_size, kwargs.get('output_format', 'png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        with Image.open(image_path).convert("RGB") as image:
            if self.transformations:
                image = self.transformations(np.asarray(image))
        return (image, image_path.name)

    def __tile(self, image_path: Path, directory: Path, kernel_size: tuple[int, int], output_format: str) -> list[Path]:
        # Adapted from https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7
        with Image.open(image_path).convert("RGB") as image:
            image = np.asarray(image)

        if image.shape[:2] > kernel_size:
            image_height, image_width, image_channels = image.shape
            tile_height, tile_width = kernel_size

            # TODO: do tiles this way too (the inverse of this operation to merge tiles)
            tiles = image.reshape(image_height//tile_height, tile_height,
                                  image_width//tile_width, tile_width, image_channels).swapaxes(1, 2)

            image_paths = []
            for r, row in enumerate(tiles):
                for c, column in enumerate(row):
                    filename = Path(
                        f'{str(directory)}/tile_r{r}c{c}.{output_format}')
                    Image.fromarray(column).save(filename)
                    image_paths.append(filename)

            return image_paths
        else:
            return [image_path]
