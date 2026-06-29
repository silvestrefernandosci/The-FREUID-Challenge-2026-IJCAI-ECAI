import pandas as pd
import os
from PIL import Image
import torch
from .region_detector import RegionDetector
from .crop_fields import CropFields

class FreuidSubmition:
    def __init__(self, filename):
        self._filename = filename
        self.crop_fields = CropFields()

    def run(self, test_path, model, transform):

        print("Submission running")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.eval()

        ids = []
        results = []
        region_detector = RegionDetector()
        regionids = {
            'BENIN/DL': 1,
            'EGYPT/DL': 2,
            'GUINEA/DL': 3,
            'MOZAMBIQUE/DL': 4,
            'MAURITIUS/ID': 5
        }

        with torch.no_grad():

            for file in os.listdir(test_path):

                image_path = os.path.join(test_path, file)

                img = Image.open(image_path).convert('RGB')
                region = region_detector.detect(img)
                regionid = regionids[region]
                img = self.crop_fields((img, regionid))

                x = transform(img)
                x = x.unsqueeze(0).to(device)

                output = model(x)

                # ajuste conforme seu modelo
                pred = torch.argmax(output, dim=1).item()

                img_id = file.rsplit('.',1)[0]

                ids.append(img_id)
                results.append(pred)

        df = pd.DataFrame({
            'id': ids,
            'label': results
        })

        df.to_csv(
            self._filename,
            index=False
        )

        print("Submission done")