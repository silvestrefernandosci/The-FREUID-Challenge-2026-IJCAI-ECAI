from PIL import Image

class CropFields:
    def __call__(self, image_data):
        img, region_id = image_data
        signature_roi = img.crop(self.crop_region('signature', region_id))
        content_roi = img.crop(self.crop_region('content', region_id))
        photo_roi = img.crop(self.crop_region('photo', region_id))

        width = self.sum_roi_width_sizes([
            self.get_roi_size(signature_roi),
            self.get_roi_size(content_roi)
        ])

        height = self.sum_roi_height_sizes([
            self.get_roi_size(signature_roi),
            self.get_roi_size(photo_roi)
        ])

        canvas = Image.new('RGB', (width + 100, height + 100))
        canvas.paste(photo_roi, (30, 30))
        canvas.paste(content_roi, (signature_roi.size[0] + 60 , 30))
        canvas.paste(signature_roi, (50, photo_roi.size[1] + 50))

        return canvas

    
    def get_roi_size(self, roi):
        return roi.size
    
    def sum_roi_width_sizes(self, rois):
        w = 0
        for r in rois:
            w += r[0]
        
        return w

    def sum_roi_height_sizes(self, rois):
        h = 0
        for r in rois:
            h += r[1]
        
        return h
    

    def crop_region(self, section: str, region: int):
        crops_sections = {
            "photo" : {
                1: (9, 248, 425, 727),
                2: (9, 248, 370, 687),
                3: (65, 256, 495, 769),
                4: (47, 108, 283, 418),
                5: (65, 236, 535, 852)
            },

            "content" : {
                1: (430, 243, 1370, 900),
                2: (380, 263, 1000, 800),
                3: (467, 200, 1250, 800),
                4: (318, 113, 950, 470),
                5: (559, 181, 1100, 679)
            },

            "signature" : {
                1: (9, 868, 420, 982),
                2: (9, 747, 400, 862),
                3: (40, 820, 420, 982),
                4: (40, 419, 318, 470),
                5: (559, 710, 1100, 852)
            }
        }

        if section not in crops_sections:
            raise ValueError(f"Error: not found section {section}")
        
        return crops_sections[section][region]