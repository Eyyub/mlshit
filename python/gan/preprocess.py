import os
import numpy as np
from PIL import Image, ImageOps


def _image_preprocessing(filename, xsize, ysize):
    im = Image.open(filename)
    downsampled_im = ImageOps.fit(im, (xsize, ysize), method=Image.LANCZOS)
    norm_im = np.array(downsampled_im, dtype=np.float32) / 255.
    if len(norm_im.shape) == 2: # gray image 1 channel
        norm_im = np.repeat(norm_im[..., np.newaxis], 3, axis=2)
    downsampled_im.close()
    im.close()
    return norm_im

def _buid_imageset(path):
    imageset = dict()
    image_nb = 0
    for category in os.listdir(path):
        images_name = list(filter(lambda name : name[-4:] in ['.jpg', '.png', '.bmp'], os.listdir(os.path.join(path,  category))))
        if len(images_name) == 0:
            continue
        imageset[category] = images_name
        image_nb += len(images_name)
    return image_nb, imageset

def get_dataset(path, xsize, ysize, chan=3):
    image_nb, imageset = _buid_imageset(path)
    imageset_keys = list(imageset.keys())
    dataset_X = np.zeros((image_nb, xsize, ysize, chan))
    dataset_Y = np.zeros((image_nb, len(imageset_keys)))
    i = 0
    for k, v in imageset.items():
        for filename in v:
            if not v[-4:] in ['.jpg', '.png', '.bmp']:
                continue
            preprocessed_im = _image_preprocessing(os.path.join(path, k, filename), xsize, ysize)
            print('pre', preprocessed_im.shape)

            dataset_X[i] = preprocessed_im
            dataset_Y[i][imageset_keys.index(k)] = 1
            i += 1
    assert i == image_nb
    return dataset_X, dataset_Y, imageset_keys

if __name__ == '__main__':
    import sys
    dataset = get_dataset(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    np.save(sys.argv[1], dataset)
