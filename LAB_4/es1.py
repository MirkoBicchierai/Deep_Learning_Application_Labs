import argparse
import os

import numpy as np
from sklearn import metrics
from model import CNN, Autoencoder, CNN2
from utils import str2bool, get_dataloaders, get_fake_loaders, test_AE, get_pred_CNN, max_logit, max_softmax, compute_scores
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def plot_confusion_matrix_accuracy(y_gt, y_pred, test_dataloader, model_name, save_path="plot/es1/"):
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
    plt.title("Confusion Matrix - " + model_name)
    plt.savefig(save_path + model_name + "_confusion_matrix.png")
    plt.close()

    cmn = cm.astype(np.float32)
    cmn /= cmn.sum(1)
    print(f'Per class accuracy: {np.diag(cmn).mean():.4f}')

def plot_score(scores_test, scores_fake, model_name, save_path="plot/es1/", score_fun=""):
    plt.plot(sorted(scores_test.cpu()), label='test')
    plt.plot(sorted(scores_fake.cpu()), label='fake')
    plt.legend()
    plt.title("Score - " + model_name + " - " + score_fun)
    plt.savefig(save_path + model_name + "_score_"+score_fun+".png")
    plt.close()

    plt.hist(scores_test.cpu(), density=True, alpha=0.5, bins=25, label='test')
    plt.hist(scores_fake.cpu(), density=True, alpha=0.5, bins=25, label='fake')
    plt.legend()
    plt.title("Histogram - " + model_name + " - " + score_fun)
    plt.savefig(save_path + model_name + "_score_hist_"+score_fun+".png")
    plt.close()

    y_pred = torch.cat((scores_test, scores_fake))
    y_test = torch.ones_like(scores_test)
    y_fake = torch.zeros_like(scores_fake)
    y = torch.cat((y_test, y_fake))

    roc = metrics.RocCurveDisplay.from_predictions(y.cpu(), y_pred.cpu())
    fig = roc.figure_
    roc_ax = fig.axes[0]
    roc_ax.set_title("ROC Curve - " + model_name + " - " + score_fun)
    fig.savefig(save_path + model_name + "_roc_curve_"+score_fun+".png")

    pr = metrics.PrecisionRecallDisplay.from_predictions(y.cpu(), y_pred.cpu())
    fig = pr.figure_
    pr_ax = fig.axes[0]
    pr_ax.set_title("Precision-Recall Curve - " + model_name + " - " + score_fun)
    fig.savefig(save_path + model_name + "_precision_recall_curve_"+score_fun+".png")

def plot_logit_soft_max(x, k, model, device, model_name, save_path="plot/es1/", ty=""):

    output = model(x.to(device))
    plt.bar(np.arange(10), output[k].detach().cpu())
    plt.title('Logit - ' + model_name + " - " + ty)
    plt.savefig(save_path + model_name + "_logit_" + ty + ".png")
    plt.close()

    plt.title('Softmax - ' + model_name + " - " + ty)
    s = F.softmax(output, 1)
    plt.bar(np.arange(10), s[k].detach().cpu())
    plt.savefig(save_path + model_name + "_softmax_" + ty + ".png")
    plt.close()

    plt.imshow(x[0, :].permute(1, 2, 0))
    plt.savefig(save_path + ty + ".png")
    plt.close()

def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker")

    # CNN Model
    parser.add_argument("--cnn", type=str2bool, default=True, help="If True test the CNN Model")
    parser.add_argument("--cnn_ty", type=str2bool, default=False,
                        help="If True use the CNN Model (more power), False use CNN2 Model")
    parser.add_argument("--model_path_cnn", type=str, default="Models/CNN2_pretrain.pth",
                        help="Path of the CNN model")

    parser.add_argument("--temp", type=float, default=1000, help="Temperature for max_softmax to predict scores of CNN")

    # Autoencoder Model
    parser.add_argument("--ae", type=str2bool, default=False, help="f True test the AE Model")
    parser.add_argument("--model_path_ae", type=str, default="Models/AE_pretrain.pth", help="Path of the AE model")

    args = parser.parse_args()

    return args

def main():
    args = get_parser()

    train_dataloader, _, test_dataloader, _, _ = get_dataloaders("CIFAR10", args.batch_size, num_workers=args.num_workers)
    fake_dataloader = get_fake_loaders(args.batch_size, num_workers=args.num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.cnn:

        if args.cnn_ty:
            modelCNN = CNN().to(device)
        else:
            modelCNN = CNN2().to(device)

        modelCNN.load_state_dict(torch.load(args.model_path_cnn))

        y_gt, y_pred = get_pred_CNN(modelCNN, test_dataloader, device)

        model_name = os.path.splitext(os.path.basename(args.model_path_cnn))[0]
        plot_confusion_matrix_accuracy(y_gt, y_pred, test_dataloader, model_name, save_path="plot/es1/")

        x, y = next(iter(test_dataloader))
        x_fake, _ = next(iter(fake_dataloader))

        k = 0  # the kth sample of the batch
        plot_logit_soft_max(x, k, modelCNN, device, model_name, save_path="plot/es1/", ty="Real data") # Plot with real data
        plot_logit_soft_max(x_fake, k, modelCNN, device, model_name, save_path="plot/es1/", ty="Fake data") # PLot with fake data

        # Using logit
        scores_test = compute_scores(modelCNN, test_dataloader, max_logit, device)
        scores_fake = compute_scores(modelCNN, fake_dataloader, max_logit, device)
        plot_score(scores_test, scores_fake, model_name, save_path="plot/es1/", score_fun="max_softmax")

        # Using max_softmax
        scores_test = compute_scores(modelCNN, test_dataloader, lambda l: max_softmax(l, t = args.temp), device)
        scores_fake = compute_scores(modelCNN, fake_dataloader, lambda l: max_softmax(l, t = args.temp), device)
        plot_score(scores_test, scores_fake, model_name, save_path="plot/es1/", score_fun="max_softmax")

    if args.ae:
    
        modelAE = Autoencoder().to(device)
        modelAE.load_state_dict(torch.load(args.model_path_ae))

        model_name = os.path.splitext(os.path.basename(args.model_path_ae))[0]

        test_score, _ = test_AE(modelAE, test_dataloader, device)
        fake_score, _ = test_AE(modelAE, fake_dataloader, device)

        plot_score(test_score, fake_score, model_name, save_path="plot/es1/")


if __name__ == "__main__":
    main()