import cv2
import json
import numpy as np

__all__ = ['SwapRGB', 'Resize', 'CenterCrop', 'Normalize',
        'Transpose', 'Compose', 'JsonTrans']

class Transform(object):

    def __call__(self, img):
        pass

class SwapRGB(Transform):

    def __init__(self):
        super(SwapRGB, self).__init__()

    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class Resize(Transform):
    def __init__(self, short_edge, keep_ratio=True):
        super(Resize, self).__init__()
        self.short_edge = short_edge
        self.keep_ratio = keep_ratio

    def __call__(self, img):
        img_h, img_w, _ = img.shape
        if isinstance(self.short_edge, int) and not self.keep_ratio:
            new_w, new_h = self.short_edge, self.short_edge
        if isinstance(self.short_edge, (tuple, list)):
            assert len(self.short_edge) == 2
            new_w, new_h = self.short_edge
        if isinstance(self.short_edge, int) and self.keep_ratio:

            scale = self.short_edge * 1.0 / min(img_h, img_w)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
        if new_h != img_h or new_w != img_w:
            img = cv2.resize(img, (new_w, new_h))
        return img

class CenterCrop(Transform):
    def __init__(self, size):
        super(CenterCrop, self).__init__()
        if isinstance(size ,(tuple, list)):
            self.w, self.h = size
        else:
            self.w, self.h = size, size

    def __call__(self, img):
        img_h, img_w, _ = img.shape
        assert(img_h >= self.h and img_w >= self.w)
        sy = (img_h-self.h)//2
        sx = (img_w-self.w)//2

        return img[sy:sy+self.h, sx:sx+self.w]

class Normalize(Transform):
    def __init__(self, mean, std, norm=True):
        super(Normalize, self).__init__()
        if len(mean) == 3:
            self.mean = np.array(mean).reshape(1,1,3)
        else:
            self.mean = mean

        if len(std) == 3:
            self.std = np.array(std).reshape(1,1,3)
        else:
            self.std = std
        self.norm = norm

    def __call__(self, img):
        if self.norm:
            img = np.float32(img) / 255.
        else:
            img = np.float32(img)
        img = (img - self.mean) / self.std
        return img

class Transpose(Transform):
    def __init__(self, axis):
        super(Transpose, self).__init__()
        self.axis = axis

    def __call__(self, img):
        return img.transpose(self.axis)

class Compose(Transform):

    def __init__(self, trans):
        super(Compose, self).__init__()
        self.trans =  trans

    def ___call__(self, img):
        for tran in self.trans:
            img = tran(img)
        return img

class JsonTrans(Transform):
    def __init__(self, obj):
        super(JsonTrans, self).__init__()
        if isinstance(obj, str):
            with open(obj, 'r') as f:
                infos = json.load(f)
        else:
            infos = obj

        self.trans = []
        for k, attrs in infos:
            self.trans.append(eval(k)(**attrs))

    def __call__(self, img):
        for tran in self.trans:
            # import pdb; pdb.set_trace()
            img = tran(img)
        return img

