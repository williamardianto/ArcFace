import glob
import os

from models.mtcnn import MTCNN
from PIL import Image
from utils import face_aligner
import numpy as np

import lfw
from PIL import Image
from align_face import align_face
from tqdm import tqdm

if __name__ == '__main__':
    aligned_dir = './data/aligned/lfw'
    raw_dir = './data/lfw/*/*.jpg'
    lfw_pairs = './data/pairs.txt'

    # align_face(raw_dir, aligned_dir)

    pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))
    paths, actual_issame = lfw.get_paths(os.path.expanduser(aligned_dir), pairs)

    print('num of paths:', len(paths))
    print('num of pairs: ', len(actual_issame))

    image_array = np.zeros([len(paths),112,112,3])

    for idx, path in enumerate(tqdm(paths)):
        image = np.asarray(Image.open(path))

        image_array[idx] = image

    np.save('data/lfw.npy', image_array)
    np.save('data/lfw_label.npy', actual_issame)




