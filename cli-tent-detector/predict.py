# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
import tempfile
from cog import BasePredictor, Input, Path
from PIL import Image
from application.unet import UNet
from main import main

OUTPUT_PATH = Path('output')
CHECKPOINT_PATH = Path('models/UNet.pth')


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = UNet()
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=self.device))

    def predict(
        self,
        image: Path = Input(description="Input image")
    ) -> Path:
        """Run a single prediction on the model"""
        
        output = Path(main(image, OUTPUT_PATH, self.model, False, verbose=True)[str(self.model)]['heatmap_path'])


        output_path = Path(tempfile.mkdtemp()) / f'result{output.suffix}'
        result = Image.open(output)
        result.save(output_path)

        return output_path
    
        # Command to test this locally: cog predict -i image=@data/x_overview.png