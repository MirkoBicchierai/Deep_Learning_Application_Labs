import argparse
import numpy as np
from sklearn import metrics
from model import CNN, Autoencoder, CNN2
from utils import str2bool, config_loggers, get_dataloaders, get_fake_loaders, test_AE, get_pred_CNN, max_logit, max_softmax, compute_scores
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def plot_confusion_matrix_accuracy(y_gt, y_pred, test_dataloader):
    y_pred_t = torch.cat(y_pred)
    y_gt_t = torch.cat(y_gt)

    accuracy = sum(y_pred_t == y_gt_t) / len(y_gt_t)
    print(f'Accuracy Test Set: {accuracy}')

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


def test_fake_img(fake_dataloader, dataloader):

    for data in fake_dataloader:
        x, y = data
        img = x[0] # * 0.5 + 0.5 unnormalize
        img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
        plt.imshow(img)
        plt.title(f"Label: {dataloader.dataset.dataset.classes[y[0].item()]}")
        plt.show()
        break

    print(x.shape, dataloader.dataset.dataset.classes[y[0]])
    print(dataloader.dataset.dataset.classes)
    class_dict = {class_name: id_class for id_class, class_name in enumerate(dataloader.dataset.dataset.classes)}
    print(class_dict)

def plot_score(scores_test, scores_fake):
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

def plot_logit_soft_max(x, k, model, device):

    output = model(x.to(device))
    plt.bar(np.arange(10), output[k].detach().cpu())
    plt.title('logit')
    plt.show()
    t = 1
    plt.title(f'softmax t={t}')
    s = F.softmax(output / t, 1)
    plt.bar(np.arange(10), s[k].detach().cpu())
    plt.show()

    plt.imshow(x[0, :].permute(1, 2, 0))
    plt.show()

def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker")

    # CNN Model
    parser.add_argument("--cnn", type=str2bool, default=True, help="If True test the CNN Model")
    parser.add_argument("--cnn_ty", type=str2bool, default=True,
                        help="If True use the CNN Model (more power), False use CNN2 Model")
    parser.add_argument("--model_path_cnn", type=str, default="Models/CNN2_pretrain.pth",
                        help="Path of the CNN model")

    parser.add_argument("--max_logit", type=str2bool, default=True,
                        help="If True use max_logit to predict scores of CNN, False use max_softmax with --temp ")
    parser.add_argument("--temp", type=float, default=1000,
                        help="Temperature for max_softmax to predict scores of CNN")

    # Autoencoder Model
    parser.add_argument("--ae", type=str2bool, default=True, help="f True test the AE Model")
    parser.add_argument("--model_path_ae", type=str, default="Models/AE_pretrain.pth", help="Path of the AE model")

    args = parser.parse_args()

    return args

def main():
    args = get_parser()

    train_dataloader, _, test_dataloader, _, _ = get_dataloaders("CIFAR10", args.batch_size, num_workers=args.num_workers)
    fake_dataloader = get_fake_loaders(args.batch_size, num_workers=args.num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_fake_img(fake_dataloader, train_dataloader)

    if args.cnn:

        if args.cnn_ty:
            modelCNN = CNN().to(device)
        else:
            modelCNN = CNN2().to(device)

        modelCNN.load_state_dict(torch.load(args.model_path_cnn))

        y_gt, y_pred = get_pred_CNN(modelCNN, test_dataloader, device)

        plot_confusion_matrix_accuracy(y_gt, y_pred, test_dataloader)

        x, y = next(iter(test_dataloader))
        x_fake, _ = next(iter(fake_dataloader))

        k = 0  # the kth sample of the batch
        plot_logit_soft_max(x, k, modelCNN, device) # Plot with real data
        plot_logit_soft_max(x_fake, k, modelCNN, device) # PLot with fake data

        if args.max_logit:
            scores_test = compute_scores(modelCNN, test_dataloader, max_logit, device)
            scores_fake = compute_scores(modelCNN, fake_dataloader, max_logit, device)
        else:
            scores_test = compute_scores(test_dataloader, lambda l: max_softmax(l, t = args.temp))
            scores_fake = compute_scores(fake_dataloader, lambda l: max_softmax(l, t = args.temp))

        plot_score(scores_test, scores_fake)

    if args.ae:
    
        modelAE = Autoencoder().to(device)
        modelAE.load_state_dict(torch.load(args.model_path_ae))

        test_score, _ = test_AE(modelAE, test_dataloader, device)
        fake_score, _ = test_AE(modelAE, fake_dataloader, device)

        plot_score(test_score, fake_score)


if __name__ == "__main__":
    main()