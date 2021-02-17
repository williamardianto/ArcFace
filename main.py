import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import ImageFile, Image
from scipy import interpolate
from scipy.optimize import brentq
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms, datasets
from torchvision import transforms as trans

import evaluate
from models.arcface import ArcFace

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cuda'
classnum = 85742
batch_size = 128

torch.backends.cudnn.benchmark = True

transform_composed = transforms.Compose([
    # transforms.RandomCrop(112),
    # transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[128, 128, 128]),
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# data_path = './data/aligned/CASIA'
# dataset = datasets.ImageFolder(root=data_path, transform=transform_composed)
work_path = Path('work_space/')
model_path = work_path / 'models'
log_path = work_path / 'log'
save_path = work_path / 'save'

# lfw_dir = './data/aligned/lfw'
# lfw_pairs = './data/pairs.txt'

lfw_dir = 'D:\Datasets\insightface\image\lfw'
lfw_pairs = 'D:\Datasets\insightface\image\lfw_pairs.txt'

# Get the paths for the corresponding images
# paths, actual_issame = evaluate.get_eval_dataset(lfw_dir, lfw_pairs)
paths, actual_issame = evaluate.get_eval_dataset_insightface(lfw_dir, lfw_pairs)

writer = SummaryWriter(log_path)


def board_val(db_name, accuracy, best_threshold, roc_curve_tensor, val, val_std, far, auc, val_loss, eer, epoch):
    writer.add_scalar('{}_accuracy'.format(db_name), accuracy, epoch)
    writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, epoch)
    writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, epoch)
    writer.add_scalar('{}_val: True Acceptance Ratio'.format(db_name), val, epoch)
    writer.add_scalar('{}_val_std'.format(db_name), val_std, epoch)
    writer.add_scalar('{}_far: False Acceptance Ratio'.format(db_name), far, epoch)
    writer.add_scalar('{}_auc'.format(db_name), auc, epoch)
    writer.add_scalar('{}_val_loss'.format(db_name), val_loss, epoch)
    writer.add_scalar('{}_eer'.format(db_name), eer, epoch)


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


def load_dataset(train_split=0.99):
    data_path = 'D:\\Datasets\\insightface\\image\\emore'
    dataset = datasets.ImageFolder(root=data_path, transform=transform_composed)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True,
                                               pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, num_workers=2, shuffle=False, pin_memory=False)

    return {'train_loader': train_loader, 'test_loader': test_loader}
    # return {'train_loader': train_loader}


def trainer(dataset, model, optimizer, scheduler, epochs, checkpoint=None):
    epochs_range = range(epochs)
    num_iteration = 0
    if checkpoint is not None:
        checkpoint_state = torch.load(checkpoint)
        model.module.load_state_dict(checkpoint_state['model_state_dict'])
        # optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint_state['scheduler_state_dict'])
        num_iteration = checkpoint_state['iteration']
        # epochs_range = range(checkpoint_state['epoch'], epochs)

        print('Accuracy: %2.5f+-%2.5f' % (np.mean(checkpoint_state['accuracy']), np.std(checkpoint_state['accuracy'])))
        print('Area Under Curve (AUC): %1.3f' % checkpoint_state['auc'])
        print('Best Threshold:', checkpoint_state['best_threshold'])

    for epoch in epochs_range:
        train_loss = 0
        val_loss = 0
        for train_batch_idx, (data, label) in enumerate(dataset['train_loader']):
            model.train()
            num_iteration += 1
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            label_onehot = torch.zeros([data.size()[0], classnum]).to(device)
            label_onehot.scatter_(1, label.view(-1, 1).long(), 1)
            thetas = model(data, label_onehot)
            loss = F.cross_entropy(thetas, label)
            loss.backward()

            train_loss += loss.item()

            optimizer.step()
            scheduler.step()

            if num_iteration % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, train_batch_idx * len(data), len(dataset['train_loader'].dataset),
                    100. * train_batch_idx / len(dataset['train_loader']), loss.item()))

            # evaluation steps
            if num_iteration % 8500 == 0:

                for test_batch_idx, (data, label) in enumerate(dataset['test_loader']):
                    model.eval()
                    data, label = data.to(device), label.to(device)

                    label_onehot = torch.zeros([data.size()[0], classnum]).to(device)
                    label_onehot.scatter_(1, label.view(-1, 1).long(), 1)

                    thetas = model(data, label_onehot)
                    loss = F.cross_entropy(thetas, label)

                    val_loss += loss.item()

                    if test_batch_idx % 50 == 0:
                        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch + 1, test_batch_idx * len(data), len(dataset['test_loader'].dataset),
                            100. * test_batch_idx / len(dataset['test_loader']), loss.item()))

                # evaluate on lfw
                tpr, fpr, accuracy, val, val_std, far, auc, best_threshold = evaluate.evaluate(model.module.backbone,
                                                                                               paths, actual_issame,
                                                                                               device=device)
                eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)

                buf = gen_plot(fpr, tpr)
                roc_curve = Image.open(buf)
                roc_curve_tensor = trans.ToTensor()(roc_curve)

                val_loss = val_loss / len(dataset['test_loader'])

                board_val('lfw', np.mean(accuracy), np.mean(best_threshold), roc_curve_tensor, val, val_std, far, auc,
                          val_loss, eer, epoch)

                print('Learning Rate:', scheduler.get_last_lr()[0])
                print('Validation Loss:', val_loss)
                print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
                print('Validation Rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                print('Area Under Curve (AUC): %1.3f' % auc)
                print('Equal Error Rate (EER): %1.3f' % eer)
                print('Best Threshold:', best_threshold)

                torch.save({
                    'iteration': num_iteration,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'auc': auc,
                    'accuracy': np.mean(accuracy),
                    'best_threshold': np.mean(best_threshold)
                }, 'resnet_{}_{:.4f}_{:.4f}.pth'.format(num_iteration, np.mean(accuracy), auc))


def main():
    arcface = ArcFace(classnum=classnum)

    arcface = nn.DataParallel(arcface)
    arcface.to(device)

    datasets = load_dataset()

    optimizer = optim.SGD(arcface.parameters(), weight_decay=5e-4, lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(arcface.parameters(), weight_decay=5e-4, lr=0.001)

    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(datasets['train_loader']), T_mult=1)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.1
    scheduler = MultiStepLR(optimizer, [225000, 337500, 450000], gamma=0.1)
    epochs = 100

    checkpoint = None

    trainer(datasets, arcface, optimizer, scheduler, epochs, checkpoint)


if __name__ == '__main__':
    main()
