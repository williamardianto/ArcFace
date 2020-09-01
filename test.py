import lfw
import os
import numpy as np
import torch

from PIL import Image
from utils import get_embedding
from models.arcface import ArcFace

if __name__ == '__main__':
    # lfw_dir = './data/lfw'
    lfw_dir = './data/lfw-deepfunneled'
    lfw_pairs = './data/pairs.txt'

    pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs)

    # print(paths)
    print('paths len:', len(paths))
    print('pairs len: ', len(actual_issame))

    embeddings = np.zeros([len(paths), 512])
    arcface = ArcFace(classnum=10)
    backbone = arcface.backbone.to('cuda')

    with torch.no_grad():
        for idx, path in enumerate(paths):
            print('process image no:', idx)
            img = Image.open(path)
            embedding = get_embedding(backbone, img, device='cuda')
            embeddings[idx] = embedding.cpu().numpy()

    np.save('temp2.npy', embeddings)

    # embeddings = np.load('temp.npy')

    tpr, fpr, accuracy = lfw.evaluate(embeddings,actual_issame)

    print('tpr:', tpr)
    print('fpr:', fpr)
    print('acc:', accuracy)
