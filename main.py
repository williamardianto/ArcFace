import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms, datasets

from models.arcface import ArcFace
from models.mtcnn import MTCNN

mtcnn = MTCNN()
arcface = ArcFace()

transform_composed = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def load_dataset():
    data_path = './data/aligned/CASIA'
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transform_composed
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )
    return train_loader


def trainer(dataset, model, optimizer, epochs):
    for epoch in range(epochs):
        for batch_idx, (data, label) in enumerate(dataset):
            optimizer.zero_grad()

            thetas = model(data, label)
            loss = F.cross_entropy(thetas, label)
            loss.backward()

            optimizer.step()
            if batch_idx % 2 == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, loss.item()))


def main():
    # input_image = Image.open("./data/aligned/CASIA/0000099/001.jpg")
    #
    # tensor = transform_composed(input_image)
    # tensor = torch.unsqueeze(tensor, 0)
    #
    # output = resnet(tensor)
    #
    # print(output)
    # print(tensor.size())
    # print(tensor.mean(), tensor.std())
    #
    # for batch_idx, (data, target) in enumerate(load_dataset()):
    #     print(batch_idx)

    # print(list(arcface.parameters()))
    optimizer = optim.SGD(arcface.parameters(), lr=0.001)
    #
    trainer(load_dataset(), arcface, optimizer, 3)


if __name__ == '__main__':
    main()
