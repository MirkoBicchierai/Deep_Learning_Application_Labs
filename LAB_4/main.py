import argparse
import random
import numpy as np
from sklearn import metrics
from model import CNN, Autoencoder
from utils import str2bool, config_loggers, get_dataloaders, get_fake_loaders, test_AE, get_pred_CNN, max_logit, max_softmax, compute_scores
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker")

    args = parser.parse_args()

    return args

def test_fake_img(fake_dataloader, train_dataloader):

    for data in fake_dataloader:
        x, y = data
        img = x[0] # * 0.5 + 0.5 unnormalize
        img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
        plt.imshow(img)
        plt.title(f"Label: {train_dataloader.dataset.dataset.classes[y[0].item()]}")
        plt.show()
        break

    print(x.shape, train_dataloader.dataset.dataset.classes[y[0]])
    print(train_dataloader.dataset.dataset.classes)
    class_dict = {class_name: id_class for id_class, class_name in enumerate(train_dataloader.dataset.dataset.classes)}
    print(class_dict)

def main():
    args = get_parser()
    config_loggers(args)

    train_dataloader, val_dataloader, test_dataloader, num_classes, input_size = get_dataloaders("CIFAR10", args.batch_size, num_workers=args.num_workers)
    fake_dataloader = get_fake_loaders(args.batch_size, num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_fake_img(fake_dataloader, train_dataloader)

    # CNN - TEST with Fake

    modelCNN = CNN().to(device)
    modelCNN.load_state_dict(torch.load("Models/CNN_pretrain.pth"))

    y_gt, y_pred = get_pred_CNN(modelCNN, test_dataloader, device)

    y_pred_t = torch.cat(y_pred)
    y_gt_t = torch.cat(y_gt)

    accuracy = sum(y_pred_t == y_gt_t) / len(y_gt_t)
    print(f'Accuracy: {accuracy}')

    cm = metrics.confusion_matrix(y_gt_t.cpu(), y_pred_t.cpu())

    cmn = cm.astype(np.float32)
    cmn /= cmn.sum(1)

    cmn = (100 * cmn).astype(np.int32)
    disp = metrics.ConfusionMatrixDisplay(cmn, display_labels=test_dataloader.dataset.classes)
    disp.plot()
    plt.show()

    cmn = cm.astype(np.float32)
    cmn /= cmn.sum(1)
    print(f'Per class accuracy: {np.diag(cmn).mean():.4f}')

    for data in test_dataloader:
        x, y = data
        # plt.imshow(x[0,:].permute(1,2,0))
        break

    for data in fake_dataloader:
        x_fake, _ = data
        # plt.imshow(x[0,:].permute(1,2,0))
        break

    # WITH REAL DATA
    k = random.randint(0, x.shape[0])
    print(f'GT: {y[k]}, {test_dataloader.dataset.classes[y[k]]}')  # the corresponding label
    output = modelCNN(x.to(device))
    plt.bar(np.arange(10), output[k].detach().cpu())
    plt.title('logit')
    plt.show()
    T = 1
    plt.title(f'softmax t={T}')
    s = F.softmax(output / T, 1)
    plt.bar(np.arange(10), s[k].detach().cpu())
    plt.show()

    plt.imshow(x[k, :].permute(1, 2, 0))
    plt.show()

    # WITH FAKE DATA
    k = 0  # the kth sample of the batch
    output = modelCNN(x_fake.to(device))
    plt.bar(np.arange(10), output[k].detach().cpu())
    plt.title('logit')
    plt.show()
    T = 1
    plt.title(f'softmax t={T}')
    s = F.softmax(output / T, 1)
    plt.bar(np.arange(10), s[k].detach().cpu())
    plt.show()

    plt.imshow(x_fake[0, :].permute(1, 2, 0))
    plt.show()

    # temp = 1000
    # scores_test = compute_scores(testloader, lambda l: max_softmax(l, T=temp))
    # scores_fake = compute_scores(fakeloader, lambda l: max_softmax(l, T=temp))

    scores_test = compute_scores(modelCNN, test_dataloader, max_logit, device)
    scores_fake = compute_scores(modelCNN, fake_dataloader, max_logit, device)

    plt.plot(sorted(scores_test.cpu()), label='test')
    plt.plot(sorted(scores_fake.cpu()), label='fake')
    plt.legend()
    plt.show()

    plt.hist(scores_test.cpu(), density=True, alpha=0.5, bins=25, label='test')
    plt.hist(scores_fake.cpu(), density=True, alpha=0.5, bins=25, label='fake')
    plt.legend()
    plt.show()

    y_pred = torch.cat((scores_test, scores_fake))
    y_test = torch.ones_like(scores_test)
    y_fake = torch.zeros_like(scores_fake)
    y = torch.cat((y_test, y_fake))

    roc = metrics.RocCurveDisplay.from_predictions(y.cpu(), y_pred.cpu())
    pr_display = metrics.PrecisionRecallDisplay.from_predictions(y.cpu(), y_pred.cpu())

    fig = roc.figure_
    fig.show()

    fig = pr_display.figure_
    fig.show()

    # AE - TEST with Fake
    
    modelAE = Autoencoder().to(device)
    modelAE.load_state_dict(torch.load("Models/AE_pretrain.pth"))

    test_score, _ = test_AE(modelAE, test_dataloader, device)
    fake_score, _ = test_AE(modelAE, fake_dataloader, device)

    plt.plot(sorted(test_score.cpu()))
    plt.plot(sorted(fake_score.cpu()))
    plt.show()

    plt.hist(test_score.cpu(), density=True, alpha=0.5, bins=25)
    plt.hist(fake_score.cpu(), density=True, alpha=0.5, bins=25)
    plt.show()

    y_pred = torch.cat((test_score, fake_score))
    y_test = torch.ones_like(test_score)
    y_fake = torch.zeros_like(fake_score)

    y = torch.cat((y_test, y_fake))
    roc = metrics.RocCurveDisplay.from_predictions(y.cpu(), y_pred.cpu())
    pr_display = metrics.PrecisionRecallDisplay.from_predictions(y.cpu(), y_pred.cpu())

    fig = roc.figure_
    fig.show()

    fig = pr_display.figure_
    fig.show()


if __name__ == "__main__":
    main()