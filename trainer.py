import torch
from torch.nn import functional as F
import numpy as np
from scipy.optimize import brentq
from scipy import interpolate

def train(epoch, dataloader, model, optimizer, device):
    train_loss = 0
    for train_batch_idx, (data, label) in enumerate(dataloader):
        model.train()
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        label_onehot = torch.zeros([data.size()[0], len(dataloader.dataset.classes)]).to(device)
        label_onehot.scatter_(1, label.view(-1, 1).long(), 1)
        thetas = model(data, label_onehot)

        loss = F.cross_entropy(thetas, label)
        loss.backward()

        train_loss += loss.item()

        optimizer.step()

        if train_batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, train_batch_idx * len(data), len(dataloader.dataset),
                100. * train_batch_idx / len(dataloader), loss.item()))

def validate(epoch, dataloader, model, device):
    test_loss = 0
    for test_batch_idx, (data, label) in enumerate(dataloader):
        model.eval()
        data, label = data.to(device), label.to(device)

        label_onehot = torch.zeros([data.size()[0], len(dataloader.dataset.classes)]).to(device)
        label_onehot.scatter_(1, label.view(-1, 1).long(), 1)
        thetas = model(data, label_onehot)

        loss = F.cross_entropy(thetas, label)

        test_loss += loss.item()

        if test_batch_idx % 10 == 0:
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, test_batch_idx * len(data), len(dataloader.dataset),
                100. * test_batch_idx / len(dataloader), loss.item()))

def evaluate(dataloader, model, label, device):
    # evaluate on lfw
    tpr, fpr, accuracy, val, val_std, far, auc, best_threshold = evaluate.evaluate(model.module.backbone, dataloader,
                                                                                   actual_issame=label,
                                                                                   device=device)


def trainer(training_dataloaders, evaluation_dataloader, model, optimizer, scheduler, epochs, device, checkpoint=None):
    epochs_range = range(epochs)
    if checkpoint is not None:
        checkpoint_state = torch.load(checkpoint)
        model.module.load_state_dict(checkpoint_state['model_state_dict'])
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_state['scheduler_state_dict'])
        epochs_range = range(checkpoint_state['epoch'], epochs)

    for epoch in epochs_range:
        train_dataloader = training_dataloaders['train_loader']
        test_dataloader = training_dataloaders['test_loader']

        # val_loss = val_loss / len(test_loader)

        evaluate(evaluation_dataloader, model, )


        scheduler.step()

        print('learning_rate:', scheduler.get_last_lr()[0])
        # print('val_loss:', val_loss)
        # print('auc:', auc)
        # print('accuracy:', accuracy)
        # print('best threshold:', best_threshold)

        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        print('Area Under Curve (AUC): %1.3f' % auc)
        eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        print('Equal Error Rate (EER): %1.3f' % eer)
        print('best threshold:', best_threshold)


        # torch.save(model.module.state_dict(), 'resnet50_{}_{:.5f}.pth'.format(epoch+1, auc))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'auc': auc,
            'accuracy': accuracy,
            'best_threshold': best_threshold
        }, 'resnet50_{}_{:.5f}.pth'.format(epoch + 1, auc))