import os
import math
import cv2
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from application.config import IOFormat
from application.dataset import SegmentationDataset
from application.utils import make_dirs


def save_data(output_dir, path_data, transformations, pbar=None, **kwargs) -> dict[str, str | None]:
    dataset = SegmentationDataset(path_data, transformations)
    loader = DataLoader(dataset, shuffle=False, batch_size=int(math.sqrt(len(
        dataset))), pin_memory=False, num_workers=os.cpu_count(), persistent_workers=True)
    vis = Visualizations(output_dir, kwargs.get('format', IOFormat))

    y = kwargs.get('y', 'y')

    if pbar: print('Saving overviews')
    x_overview_path, y_overview_path = vis.tile(loader,
                                                kwargs.get('x_overview_name', 'x_overview' if not kwargs.get(
                                                    'x_overview_path') else None),
                                                f'{y}_overview',
                                                pbar)
    x_overview_path = kwargs.get('x_overview_path', x_overview_path)
    
    overlay_path = heatmap_path = None
    if x_overview_path:
        ext = x_overview_path.suffix
        if pbar: print('Saving overlays')
        np.vectorize(vis.save_overlay_from_path)(
            path_data['x_paths'], path_data['y_paths'], np.vectorize((lambda n: f'{n}{ext}'))(path_data.index), 0.7, tile=True)

        if pbar: print('Saving overview overlay')
        overlay_path = vis.save_overlay_from_path(
            x_overview_path, y_overview_path, f'{y}_overlay{ext}', 0.7)

        if pbar: print('Saving heatmap')
        heatmap_path = vis.save_heatmap(overlay_path, path_data, f'{y}_heatmap{ext}', kwargs.get(
            'heatmap_title', 'Number of Tents per Region'))

    return {'x_overview_path': x_overview_path,
            'y_overview_path': y_overview_path,
            'overlay_path': overlay_path,
            'heatmap_path': heatmap_path}


class Visualizations():
    """Allows us to work with the images while ensuring we stay in the right directory."""

    def __init__(self, root_directory: Path, format: IOFormat = IOFormat) -> None:
        self.format = format
        self.dirs = make_dirs({'output': root_directory,
                               'tiles': root_directory / 'tiles',
                               't_overlay': root_directory / 'tiles' / 'overlay'})

    def merge(self, tiles: torch.Tensor, tile_row) -> torch.Tensor:
        """Merges a row of tiles. The total number of tiles must be divisible by 2."""
        if tiles == None:
            tiles = torch.cat(tuple(tile_row), 2)
        else:
            tiles = torch.cat((tiles, torch.cat(tuple(tile_row), 2)), 1)
        return tiles

    def tile(self, loader: DataLoader, output_name_x: str, output_name_y: str, pbar: tqdm = None) -> tuple[Path, Path]:
        """Tiles using a loader. Total number of tiles must be divisible by 2."""
        output_path_x = output_path_y = ext = output_x = output_y = None
        
        if pbar:
            pbar = pbar(loader)
            pbar.set_description(f'Tiling')
        for i, (x, y, filename) in enumerate(pbar) if pbar else enumerate(loader):
            if i == 0: ext = f'.{filename[i].split(".")[1]}' # Gets the extension from the filename
            if output_name_x:
                output_x = self.merge(output_x, x)
            if output_name_y:
                output_y = self.merge(output_y, y)

        if output_x is not None:
            output_path_x = self.dirs.output / f'{output_name_x}{ext}'
            save_image(output_x, output_path_x)
        if output_y is not None:
            output_path_y = self.dirs.output / f'{output_name_y}{ext}'
            save_image(output_y, output_path_y)

        return output_path_x, output_path_y

    def overlay_from_path(self, background_path: Path, foreground_path: Path, bg_opacity: float = 1, fg_opacity: float = 1):
        with Image.open(background_path).convert('RGB') as background_image:
            background_image = background_image
        with Image.open(foreground_path).convert('RGB') as foreground_image:
            foreground_image = foreground_image
        return self.overlay_image(background_image, foreground_image, bg_opacity, fg_opacity)

    def save_overlay_from_path(self, background_path: Path, foreground_path: Path, output_filename: str, bg_opacity: float = 1, fg_opacity: float = 1, tile: bool = False) -> Path:
        return self.save_overlay(self.overlay_from_path(background_path, foreground_path, bg_opacity, fg_opacity), output_filename, tile)

    def overlay_image(self, background_image, foreground_image, bg_opacity: float = 1, fg_opacity: float = 1):
        # Make sure foreground image matches background image size
        foreground_image = np.array(foreground_image.resize(background_image.size))
        background_image = np.array(background_image)

        for channel in range(1, 2):
            foreground_image[foreground_image[:, :, channel] > 0, channel] = 0

        overlay = cv2.addWeighted(
            background_image, bg_opacity, foreground_image, fg_opacity, 0)
        return Image.fromarray(overlay)

    def save_overlay(self, overlayed_image: Image, output_filename: str, tile: bool = False) -> Path:
        output_path = (self.dirs.t_overlay if tile else self.dirs.output) / f'{output_filename}'
        overlayed_image.save(output_path)
        return output_path

    def save_heatmap(self, image_path: Path, dataframe: pd.DataFrame, output_filename: str, title: str):
        data = [x for x in np.array_split(dataframe['labels'].replace(
            0, np.nan).tolist(), int(math.sqrt(dataframe.shape[0])))]

        sns.set(font_scale=1)
        _, ax = plt.subplots(figsize=(15, 15))
        ax.set_title(title)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.tick_params(left=False, bottom=False)
        sns.heatmap(data, annot=True, square=True, fmt='.5g',
                    alpha=0.3, zorder=2, cbar_kws={'shrink': 0.7}, ax=ax)

        with Image.open(image_path).convert("RGB") as image:
            ax.imshow(image, aspect=ax.get_aspect(),
                      extent=ax.get_xlim()+ax.get_ylim(), zorder=1)

        output_path = f'{self.dirs.output}/{output_filename}'

        plt.savefig(output_path, bbox_inches='tight')

        # Cleanup
        plt.clf()
        plt.cla()
        plt.close()

        return output_path
