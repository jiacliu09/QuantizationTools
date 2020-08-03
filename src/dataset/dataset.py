import numpy as np
import cv2, os

from ..utils import get_image_list, process_image_batch

class Dataset(object):
    pass

class CalibDataset(Dataset):
    def __init__(self, image, batch_size=1, transformer=None):
        # self.image_list = get_image_list(image_path)
        if os.path.isdir(image):
            self.image_list = get_image_list(image)
        else:
            self.image_list = image

        self.transformer = transformer
        self.image_batch_list = [self.image_list[i:i+batch_size] \
                for i in range(0, len(self.image_list), batch_size)]

    def __len__(self):
        return len(self.image_batch_list)

    def __getitem__(self, idx):
        imgs = []
        for img_path in self.image_batch_list[idx]:
            img = cv2.imread(img_path)
            if self.transformer is not None:
                img = self.transformer(img)
            imgs.append(img)

        try:
            imgs = np.array(imgs)
        except:
            pass

        return imgs
