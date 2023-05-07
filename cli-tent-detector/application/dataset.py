import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from application.utils import Tiler


class SegmentationDataset(Dataset):
    """Used for training the model."""
    # Adapted from: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

    def __init__(self, xy_paths: pd.DataFrame, transformations: transforms.Compose | None = None, tiler: Tiler | None = None, **kwargs) -> None:
        self.transformations = transformations
        if len(xy_paths.index) == 1: # True when we have a single image path or when we have a directory path
            paths = {'x_paths': [], 'y_paths': []}

            x_path = xy_paths['x_paths'][0]
            y_path = xy_paths['y_paths'][0]

            if x_path.is_dir():
                paths['x_paths'] = [path for path in Path(x_path).glob(f'*')]
                if y_path:
                    assert y_path.is_dir()
                    paths['y_paths'] = [next(Path(y_path).glob(path.name)) for path in paths['x_paths']]
                else:
                    paths['y_paths'] = [None for _ in paths['x_paths']]
                self.paths = pd.DataFrame.from_dict(paths).sort_values('x_paths', ignore_index=True)
                return None
            elif tiler: # If we have an image send it to the tiler
                self.paths = tiler.tile(xy_paths, overwrite_tiles_dir=True)
                if self.paths is not None: return None # We only get paths back if tiling was necessary
                
        # If we have multiple paths, or one non-directory path of an image less than or equal to the kernel size, just copy the dataframe
        self.paths = xy_paths[['x_paths', 'y_paths']].copy(deep=kwargs.get('deep_copy', True))

    def __len__(self) -> int:
        return len(self.paths.index)

    def __getitem__(self, index):
        """
        If there's no mask path, generate a blank mask. This is for the case where
        we only want to perform predictions and as such haven't included any masks.
        Works as expected when mask paths are included.
        """
        
        x_path = self.paths.iloc[index]['x_paths']
        y_path = self.paths.iloc[index]['y_paths']
        assert x_path.exists()

        image = cv2.cvtColor(cv2.imread(str(x_path)), cv2.COLOR_BGR2RGB)
        mask = None

        if y_path:
            mask = cv2.threshold(cv2.imread(str(y_path), cv2.IMREAD_GRAYSCALE), 150, 255, cv2.THRESH_BINARY)[1]
        else:
            mask = cv2.threshold(np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8), 150, 255, cv2.THRESH_BINARY)[1]
        if self.transformations:
            image = self.transformations(image)
            mask = self.transformations(mask)

        return image, mask, x_path.name
