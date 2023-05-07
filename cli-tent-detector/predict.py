# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from cog import BasePredictor, Input, Path
from application.unet import UNet
from main import main

OUTPUT_PATH = Path('output')
CHECKPOINT_PATH = Path('models/UNet.pth')


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.predictions_directory = Path('predictions')
        # self.tiles_directory = Path('tiles')
        # self.predictions_directory.mkdir(parents=True, exist_ok=True)
        # self.tiles_directory.mkdir(parents=True, exist_ok=True)
        self.device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = UNet()
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=self.device))

    def predict(
        self,
        image: Path = Input(description="Input image")
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)

        return main(image, OUTPUT_PATH, self.model, False, verbose=True)

        

        # Command to test this: cog predict -i image=@data/x_overview.png