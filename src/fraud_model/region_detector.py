import json
import numpy as np

class RegionDetector:
    def __init__(self, path='hists.json'):
        with open(path, 'r') as f:
            self.headers_hists = json.load(f)

    def detect(self, img)->str:
        return self.get_image_region(img)
    
    def get_header(self, img):
        w, h = img.size
        head = int(h * 0.8)
        return img.crop((0, 0, w , h - head))

    def get_image_region(self, img)->str:
        labels = {
            "benin": 'BENIN/DL',
            "egypt": 'EGYPT/DL',
            "guinea": 'GUINEA/DL',
            "mozambique": 'MOZAMBIQUE/DL',
            "mauritius": 'MAURITIUS/ID'
        }

        with open('hists.json', 'r') as f:
            data = json.loads(f.read())

        hist1 = np.array(self.get_header(img).histogram())
        histnorm1 = hist1 / np.linalg.norm(hist1)

        best_distance = float('inf')
        best_region = None
        
        for region, hist in data.items():
            hist2 = np.array(hist)

            histnorm2 = hist2 / np.linalg.norm(hist2)


            hist_distance = np.linalg.norm(histnorm1 - histnorm2)

            if hist_distance < best_distance:
                best_distance = hist_distance
                best_region = region

        if best_distance < 0.3:
            return labels[best_region]

        return labels['mozambique']