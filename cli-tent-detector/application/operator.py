import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statistics import mean
from tqdm.auto import tqdm
from torch.optim import Optimizer
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from application.config import IOFormat
from application.utils import make_dirs


class Operator():
    """Makes it possible to work with models in a more generalized and scalable fashion."""

    def __init__(self, root_directory: Path, model: torch.nn.Module, loss_fn: torch.nn.Module, opt_fn: Optimizer, lr: float, metric_fns=None, **kwargs) -> None:
        self.device = kwargs.get('device', 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = model.to(self.device)

        # Only needed if training (TODO: change after we get this working; maybe include in training settings and pass training settings to Model?) #
        self.loss_fn = loss_fn
        self.opt_fn = opt_fn(self.model.parameters(),
                             lr=lr)
        self.metric_fns = metric_fns

        if self.metric_fns:
            for metric_fn in self.metric_fns:
                metric_fn.to(self.device)
        ###########################

        self.format = kwargs.get('format', IOFormat)
        self.dirs = make_dirs({'output':       root_directory,
                               'predictions':  root_directory / 'tiles',
                               'history':      root_directory / 'metrics'})

        # TODO: handle the case when metrics is none (it might break history); might
        # want to do the same with loss if we're not interested in training a new model.
        self.history = {
            't': {str(key): value for (key, value) in zip(['losses', *self.metric_fns], [[] for _ in [*self.metric_fns, '']])},
            'v': {str(key): value for (key, value) in zip(['losses', *self.metric_fns], [[] for _ in [*self.metric_fns, '']])}}

    def __str__(self) -> str:
        return str(self.model)

    def train(self, epochs, t_loader, v_loader, pbar: tqdm = None, **kwargs) -> pd.DataFrame:
        """Trains the model."""
        
        if pbar:
            pbar = pbar(range(epochs))
            pbar.set_description(f'Training {self.model}')
        for e in pbar if pbar else range(epochs):
            self.model.train()
            losses = []
            for x, y, _ in t_loader:
                x, y = (x.to(self.device), y.to(self.device))

                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                losses.append(loss.item())

                if loss.requires_grad:
                    self.opt_fn.zero_grad()
                    loss.backward()
                self.opt_fn.step()

                if self.metric_fns:
                    for metric_fn in self.metric_fns:
                        metric_fn.update(pred, y)

            self.history['t']['losses'].append(mean(losses))
            if self.metric_fns:
                for metric_fn in self.metric_fns:
                    self.history['t'][f'{metric_fn}'].append(
                        metric_fn.compute().cpu().detach().numpy().item())
                    metric_fn.reset()

            # Append the prediction metrics into history. We might not actually need a lambda for this.
            [(lambda k, v: [self.history['v'][k].append(p) for p in v])(k, v)
             for k, v in self.predict(v_loader)[2].items()]

            if pbar:
                pbar.set_description(
                    f'Epoch({e+1}/{epochs}) Training {self.model}, Training Loss: {self.history["t"]["losses"][-1]:.4f}, Validation Loss: {self.history["v"]["losses"][-1]:.4f}')

        if kwargs.get('save_model'):
            self.save_checkpoint()
        if kwargs.get('save_history'):
            self.save_training_history()
        return pd.DataFrame(self.history).fillna(np.nan)

    def predict(self, loader: DataLoader, pbar: tqdm = None, save_predictions: bool = False) -> tuple[list, pd.DataFrame | None, pd.DataFrame]:
        """Evaluates the model."""
        history = {str(key): value for (key, value) in zip(
            ['losses', *self.metric_fns], [[] for _ in [*self.metric_fns, '']])}
        predictions = []
        with torch.no_grad():
            self.model.eval()

            losses = []
            if pbar:
                pbar = pbar(loader)
                pbar.set_description(f'Evaluating {self.model}')
            for x, y, filename in pbar if pbar else loader:
                x, y = (x.to(self.device), y.to(self.device))

                p = self.model(x)

                if self.loss_fn:
                    loss = self.loss_fn(p, y)
                    losses.append(loss.item())
                if self.metric_fns:
                    for metric_fn in self.metric_fns:
                        metric_fn.update(p, y)

                for batch, mask in enumerate(p.cpu().detach()):
                    predictions.append({'name': filename[batch], 'mask': mask})

            if losses != []:
                history['losses'].append(mean(losses))
            if self.metric_fns:
                for metric_fn in self.metric_fns:
                    history[f'{metric_fn}'].append(
                        metric_fn.compute().cpu().detach().numpy().item())
                    metric_fn.reset()

        return predictions, self.__save_predictions(predictions) if save_predictions else None, pd.DataFrame(history).fillna(np.nan)

    def __save_predictions(self, predictions: list, **kwargs) -> pd.DataFrame:
        """Saves predictions to disk and outputs a dataframe with the names, paths, and labels."""
        saved_predictions = []
        for prediction in predictions:
            out_path = kwargs.get("directory_override", self.dirs.predictions) / prediction["name"]
            save_image(prediction['mask'], out_path)
            saved_predictions.append({'x_paths': None,
                                      'y_paths': out_path,
                                      'labels': self.__count_contours(prediction['mask'])})
        return pd.DataFrame(saved_predictions).sort_values('y_paths', ignore_index=True).fillna(np.nan)

    # TODO: either remove save_predictions_to_spreadsheet or set it up so we call it from __save_predictions
    # def save_predictions_to_spreadsheet(self, predictions_dataframe, **kwargs):
    #   output_path = f'{kwargs.get("directory_override", self.dirs.predictions)}/{kwargs.get("filename_override", f"labels.{self.format.spreadsheet}")}'
    #   predictions_dataframe.to_csv(output_path)
    #   return output_path

    def save_checkpoint(self, **kwargs) -> None:
        if kwargs.get('checkpoint', True):
            torch.save(self.model.state_dict(),
                       f'{kwargs.get("directory_override", self.dirs.output)}/{kwargs.get("filename_override", f"{self.model}.{self.format.model}")}')
        else:
            torch.save(self.model,
                       f'{kwargs.get("directory_override", self.dirs.output)}/{kwargs.get("filename_override", f"{self.model}.{self.format.model}")}')

    def save_training_history(self) -> tuple[str, list]:
        """Saves training history (loss and metric values per epoch) to disk."""
        model_name = str(self.model).replace('()', '')
        loss_fn_name = str(self.loss_fn).replace('()', '')
        loss_output_path = f'{self.dirs.history}/{model_name}_loss_{loss_fn_name}.{self.format.image}'
        metric_output_paths = []

        # Loss
        plt.plot(self.history['t']['losses'], label='training')
        plt.plot(self.history['v']['losses'], label='validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title(f'{model_name} Loss ({loss_fn_name})')
        plt.savefig(loss_output_path)

        # Cleanup
        plt.clf()
        plt.cla()
        plt.close()

        # Metrics
        if self.metric_fns:  # TODO: just iterate through self.history instead
            for metric_fn in self.metric_fns:
                metric_fn_name = str(metric_fn).replace('()', '')
                metric_output_path = f'{self.dirs.history}/{model_name}_metric_{metric_fn_name}.{self.format.image}'

                plt.plot(self.history['t'][f'{metric_fn}'], label='training')
                plt.plot(self.history['v'][f'{metric_fn}'], label='validation')
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.ylim([0, 1])
                plt.legend(loc='lower right')
                plt.title(f'{model_name} Metric ({metric_fn_name})')
                plt.savefig(metric_output_path)

                # Cleanup
                plt.clf()
                plt.cla()
                plt.close()

                metric_output_paths.append(metric_output_path)

        return loss_output_path, metric_output_paths

    def __save_prediction_performance(self, p_performance) -> str:
        """Saves the loss and metrics of the prediction if applicable."""
        model_name = str(self.model).replace('()', '')
        output_path = f'{self.dirs.history}/{model_name}_performance.{self.format.image}'
        bar_data = {}
        if self.loss_fn:
            bar_data[str(self.loss_fn).replace('()', '')
                     ] = mean(p_performance['losses'])
        if self.metric_fns:
            for metric_fn in self.metric_fns:
                bar_data[str(metric_fn).replace('()', '')] = mean(
                    p_performance[f'{metric_fn}'])
        if bar_data:
            plt.bar(list(bar_data.keys()), list(bar_data.values()))
            plt.ylim([0, 1])
            plt.title(f'{model_name} Mean Performance')
            plt.savefig(output_path)
            plt.close()
        return output_path

    def __contours(self, p) -> list:
        """Used to locate blobs in the prediction that correspond to tents."""
        # Adapted from https://stackoverflow.com/questions/48154642/how-to-count-number-of-dots-in-an-image-using-python-and-opencv
        img = p.numpy().T.astype(np.uint8).copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cnts = cv2.findContours(closing, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        max_area = 20
        xcnts = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < max_area:
                xcnts.append(cnt)
        return xcnts

    def __count_contours(self, p) -> int:
        """Count the number of blobs in the prediction mask (i.e., number of tents)"""
        return int(len(self.__contours(p)))
