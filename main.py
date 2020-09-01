import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch import nn

from models.arcface import ArcFace

device = 'cuda'
classnum = 10
batch_size = 128

transform_composed = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

data_path = './data/aligned/CASIA'
dataset = datasets.ImageFolder(root=data_path, transform=transform_composed)

def load_dataset(train_split=0.8):
    data_path = './data/aligned/CASIA'
    dataset = datasets.ImageFolder(root=data_path, transform=transform_composed)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0, shuffle=True)

    return {'train_loader': train_loader, 'test_loader': test_loader}


def trainer(dataset, model, optimizer, epochs):
    for epoch in range(epochs):
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
            if train_batch_idx % 5 == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

        for test_batch_idx, (data, label) in enumerate(dataset['test_loader']):
            model.eval()
            data, label = data.to(device), label.to(device)

            label_onehot = torch.zeros([data.size()[0], classnum]).to(device)
            label_onehot.scatter_(1, label.view(-1, 1).long(), 1)

            thetas = model(data, label_onehot)
            test_loss = F.cross_entropy(thetas, label)
            if test_batch_idx % 5 == 0:
                print('Test Epoch: {} \tLoss: {:.6f}'.format(epoch, test_loss.item()))

def main():

    arcface = ArcFace(classnum=10)
    arcface = nn.DataParallel(arcface)
    arcface.to(device)


    optimizer = optim.SGD(arcface.parameters(), lr=0.001)
    epochs = 50

    datasets = load_dataset()



    trainer(datasets, arcface, optimizer, epochs)


if __name__ == '__main__':
    main()
