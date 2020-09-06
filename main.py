import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch import nn
import numpy as np
from models.arcface import ArcFace
import lfw
import os
import evaluate

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

device = 'cuda'
classnum = 10575
batch_size = 512

transform_composed = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# data_path = './data/aligned/CASIA'
# dataset = datasets.ImageFolder(root=data_path, transform=transform_composed)

lfw_dir = './data/aligned/lfw'
lfw_pairs = './data/pairs.txt'

# Get the paths for the corresponding images
paths, actual_issame = evaluate.get_eval_dataset(lfw_dir,lfw_pairs)

def load_dataset(train_split=1):
    data_path = './data/aligned/CASIA'
    dataset = datasets.ImageFolder(root=data_path, transform=transform_composed)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0, shuffle=True)

    # return {'train_loader': train_loader, 'test_loader': test_loader}
    return {'train_loader': train_loader}


def trainer(dataset, model, optimizer, scheduler, epochs, checkpoint=None):

    epochs_range = range(epochs)
    if checkpoint is not None:
        checkpoint_state = torch.load(checkpoint)
        model.module.load_state_dict(checkpoint_state['model_state_dict'])
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint_state['scheduler_state_dict'])
        epochs_range = range(checkpoint_state['epoch'], epochs)


    for epoch in epochs_range:
        for train_batch_idx, (data, label) in enumerate(dataset['train_loader']):
            model.train()
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            label_onehot = torch.zeros([data.size()[0], classnum]).to(device)
            label_onehot.scatter_(1, label.view(-1, 1).long(), 1)
            thetas = model(data, label_onehot)
            loss = F.cross_entropy(thetas, label)
            loss.backward()

            optimizer.step()
            scheduler.step()

            if train_batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, train_batch_idx * len(data), len(dataset['train_loader'].dataset),
                           100. * train_batch_idx / len(dataset['train_loader']), loss.item()))

        # for test_batch_idx, (data, label) in enumerate(dataset['test_loader']):
        #     model.eval()
        #     data, label = data.to(device), label.to(device)
        #
        #     label_onehot = torch.zeros([data.size()[0], classnum]).to(device)
        #     label_onehot.scatter_(1, label.view(-1, 1).long(), 1)
        #
        #     thetas = model(data, label_onehot)
        #     test_loss = F.cross_entropy(thetas, label)
        #     if test_batch_idx % 5 == 0:
        #         print('Test Epoch: {} \tLoss: {:.6f}'.format(epoch, test_loss.item()))

        # evaluate on lfw
        tpr, fpr, auc, accuracy, best_threshold = evaluate.evaluate(model.module.backbone, paths, actual_issame, device=device)

        print('auc:', auc)
        print('accuracy:', accuracy)
        print('best threshold:', best_threshold)

        # torch.save(model.module.state_dict(), 'resnet50_{}_{:.5f}.pth'.format(epoch+1, auc))

        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'auc': auc,
            'accuracy': accuracy,
            'best_threshold': best_threshold
            }, 'resnet50_{}_{:.5f}.pth'.format(epoch+1, auc))





def main():

    arcface = ArcFace(classnum=classnum, backbone='shufflenetv2')
    arcface = nn.DataParallel(arcface)
    arcface.to(device)

    datasets = load_dataset()

    optimizer = optim.SGD(arcface.parameters(), lr=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(datasets['train_loader']), T_mult=1)
    epochs = 500


    checkpoint = 'resnet50_42_0.92730.pth'


    trainer(datasets, arcface, optimizer, scheduler, epochs, checkpoint)


if __name__ == '__main__':
    main()
