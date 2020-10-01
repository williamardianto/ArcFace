from torchvision import transforms, datasets
import torch

class Dataset():
    def __init__(self, train_data_path, val_data_path, train_batch_size, val_batch_size):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_transform_composed = transforms.Compose([
            transforms.RandomCrop(112),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.val_transform_composed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.train_dataset = datasets.ImageFolder(root=self.train_data_path, transform=self.train_transform_composed)
        self.val_dataset = datasets.ImageFolder(root=self.val_data_path, transform=self.val_transform_composed)

        self.train_classnum = len(self.train_dataset.classes)

    def load_train_dataset(self, train_split = 0.99):
        train_size = int(train_split * len(self.train_dataset))
        test_size = len(self.train_dataset) - train_size

        train_set, test_set = torch.utils.data.random_split(self.train_dataset, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batch_size, num_workers=4, shuffle=True,
                                                   pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.train_batch_size, num_workers=4, shuffle=False,
                                                  pin_memory=True)

        return {'train_loader': train_loader, 'test_loader': test_loader}

    def load_val_dataset(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False)

