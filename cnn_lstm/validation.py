import torch
import numpy as np
from utils import AverageMeter, calculate_accuracy
from sklearn.metrics import confusion_matrix, roc_auc_score


def val_epoch(model, data_loader, criterion, device):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

    # show info
    print(
        'Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), losses.avg,
                                                                                   accuracies.avg * 100))
    return losses.avg, accuracies.avg


def test_epoch(model, data_loader, criterion, device):
    i = 0
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    distribution=[]
    with torch.no_grad():
        for (data, targets) in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            score, pred = outputs.topk(1, 1, True)
            pred = pred.t().squeeze()
            score = score.squeeze()

            if i == 0:
                score_auc = score
                targets_auc = targets
                pred_CM = pred
            else:
                if score.shape.__len__() == 0:
                    score = score.unsqueeze(0)
                    pred = pred.unsqueeze(0)
                score_auc = torch.cat((score_auc, score))
                targets_auc = torch.cat((targets_auc, targets))
                pred_CM = torch.cat((pred_CM, pred))
            i += 1

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

            outputArray = outputs.cpu().detach().numpy()
            targetArray = targets.cpu().detach().numpy().reshape((outputArray.shape[0], 1))
            predArray = pred.cpu().detach().numpy().reshape((outputArray.shape[0], 1))
            scoreArray = score.cpu().detach().numpy().reshape((outputArray.shape[0], 1))

            stat = np.concatenate((outputArray, targetArray, predArray, scoreArray), 1)
            distribution.append(stat)


        CM = confusion_matrix(targets_auc.cpu(), pred_CM.cpu())
        tn = CM[0][0]
        tp = CM[1][1]
        fp = CM[0][1]
        fn = CM[1][0]
        sensitivity = (tp / (tp + fn)) * 100
        specificity = (tn / (tn + fp)) * 100
        auc = roc_auc_score(targets_auc.cpu(), score_auc.cpu())
        precision = (tp / (tp + fp)) * 100
        recall = (tp / (tp + fn)) * 100

    # show info
    print('Test set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}\tsensitivity: {:.4f}\tspecificity: {:.4f}'
          '\tauc: {:.4f}\tprecision: {:.4f}\trecall: {:.4f}%'.format(len(data_loader.dataset), losses.avg,
                                                                     accuracies.avg * 100, sensitivity, specificity,
                                                                     auc, precision, recall))
    distribution = np.vstack(distribution)
    return losses.avg, accuracies.avg, sensitivity, specificity, CM, distribution