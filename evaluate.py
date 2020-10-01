import os

import numpy as np
import torch
from torch.nn import functional as F

import lfw
from model import MobileFaceNet

def get_eval_dataset(image_dir,pairs):
    pairs = lfw.read_pairs(os.path.expanduser(pairs))
    paths, actual_issame = lfw.get_paths(os.path.expanduser(image_dir), pairs)
    return paths, actual_issame

def evaluate(model, dataloader, actual_issame, batch_size=64, embedding_size=512, device='cpu'):
    model.eval()

    # dataloader = lfw.dataloader(paths, batch_size=batch_size)

    embeddings = np.zeros((len(dataloader.dataset), embedding_size))

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            images = sample_batched.to(device)
            b_size = images.size()[0]
            current_idx = (i_batch * batch_size)

            embeddings[current_idx:current_idx + b_size] = F.normalize(model(images)).cpu().numpy()

    tpr, fpr, auc, accuracy, best_threshold = lfw.evaluate(embeddings, actual_issame)

    return tpr, fpr, auc, accuracy, best_threshold

if __name__ == '__main__':
    mobileFacenet = MobileFaceNet(512).to('cuda')
    mobileFacenet.load_state_dict(torch.load('mobilefacenet.pth'))

    lfw_dir = './data/aligned/lfw'
    # lfw_dir = './data/lfw-deepfunneled'
    lfw_pairs = './data/pairs.txt'

    pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs)

    tpr, fpr, auc, accuracy, best_threshold = evaluate(mobileFacenet, paths, actual_issame, device='cuda')
    print(auc, accuracy, best_threshold)
