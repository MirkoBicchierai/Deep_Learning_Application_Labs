import argparse
import os
import numpy as np
from sklearn import metrics
from model import CNN, Autoencoder, CNN2
from utils import str2bool, get_dataloaders, get_fake_loaders, test_AE, get_pred_CNN, max_logit, max_softmax, compute_scores
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

"""
Plots and saves the confusion matrix for a CNN or CNN2 model, and prints overall and per-class accuracy.

Args:
    y_gt (list of Tensors): Ground truth labels for each batch.
    y_pred (list of Tensors): Predicted labels for each batch.
    test_dataloader (DataLoader): DataLoader used for obtaining class labels.
    model_name (str): Name of the model to include in the plot filename.
    save_path (str): Directory path where the confusion matrix plot will be saved.

The confusion matrix is normalized and saved as a PNG file named '{model_name}_confusion_matrix.png'.
"""
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


"""
Plots and saves score-related visualizations for an autoencoder or CNN model.

Generates and saves the following plots:
- Sorted score line plot for test and fake samples
- Histogram of scores for test and fake samples
- ROC curve based on the scores
- Precision-Recall curve based on the scores

Args:
    scores_test (Tensor): Scores for the test (real) samples.
    scores_fake (Tensor): Scores for the fake (or negative) samples.
    model_name (str): Name of the model (used in plot titles and filenames).
    save_path (str): Directory path to save the plots.
    score_fun (str): Name of the scoring function used (e.g., "max_logit", "max_softmax") only if the model is a CNN.

All plots are saved as PNG files in `save_path` with filenames incorporating the model name and scoring function.
"""
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


"""
Plots the output logits and corresponding softmax probabilities of a CNN model for a specific sample,
and saves the plots along with the input image.

Args:
    x (Tensor): Input batch of images.
    k (int): Index of the sample in the batch to plot.
    model (nn.Module): The CNN model.
    device (torch.device): Device on which the model and data reside.
    model_name (str): Name of the model, used in plot titles and filenames.
    save_path (str): Directory to save the plots.
    ty (str): Optional suffix to add to filenames and titles.
"""
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
    parser = argparse.ArgumentParser(description="Hyperparameter and model configuration settings")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for data loading and training")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker threads for data loading")

    # CNN Model options
    parser.add_argument("--cnn", type=str2bool, default=True, help="Whether to test the CNN model")
    parser.add_argument("--cnn_ty", type=str2bool, default=False,help="If True, use the more powerful CNN model; if False, use the CNN2 model")
    parser.add_argument("--model_path_cnn", type=str, default="Models/CNN2_pretrain.pth", help="File path to the pretrained CNN model")

    parser.add_argument("--temp", type=float, default=1000,  help="Temperature parameter for max_softmax score calculation in CNN prediction")

    # Autoencoder Model options
    parser.add_argument("--ae", type=str2bool, default=False, help="Whether to test the autoencoder (AE) model")
    parser.add_argument("--model_path_ae", type=str, default="Models/AE_pretrain.pth", help="File path to the pretrained autoencoder model")

    args = parser.parse_args()

    return args

def main():
    args = get_parser()

    train_dataloader, _, test_dataloader, _, _ = get_dataloaders("CIFAR10", args.batch_size, num_workers=args.num_workers)
    fake_dataloader = get_fake_loaders(args.batch_size, num_workers=args.num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test on CNN or CNN2 model
    if args.cnn:

        if args.cnn_ty:
            modelCNN = CNN().to(device)
        else:
            modelCNN = CNN2().to(device)

        modelCNN.load_state_dict(torch.load(args.model_path_cnn))  # load model specified with the parser

        y_gt, y_pred = get_pred_CNN(modelCNN, test_dataloader, device) # get the prediction of the CNN

        model_name = os.path.splitext(os.path.basename(args.model_path_cnn))[0]
        plot_confusion_matrix_accuracy(y_gt, y_pred, test_dataloader, model_name, save_path="plot/es1/") # Plot the confusion matrix accuracy

        x, y = next(iter(test_dataloader))
        x_fake, _ = next(iter(fake_dataloader))

        k = 0  # the kth sample of the batch
        plot_logit_soft_max(x, k, modelCNN, device, model_name, save_path="plot/es1/", ty="Real data") # Plot with real data
        plot_logit_soft_max(x_fake, k, modelCNN, device, model_name, save_path="plot/es1/", ty="Fake data") # PLot with fake data

        # Using logit
        scores_test = compute_scores(modelCNN, test_dataloader, max_logit, device)
        scores_fake = compute_scores(modelCNN, fake_dataloader, max_logit, device)
        plot_score(scores_test, scores_fake, model_name, save_path="plot/es1/", score_fun="max_logit")

        # Using max_softmax
        scores_test = compute_scores(modelCNN, test_dataloader, lambda l: max_softmax(l, t = args.temp), device)
        scores_fake = compute_scores(modelCNN, fake_dataloader, lambda l: max_softmax(l, t = args.temp), device)
        plot_score(scores_test, scores_fake, model_name, save_path="plot/es1/", score_fun="max_softmax")

    # Test the autoencoder model
    if args.ae:
    
        modelAE = Autoencoder().to(device)
        modelAE.load_state_dict(torch.load(args.model_path_ae)) # load model specified with the parser

        model_name = os.path.splitext(os.path.basename(args.model_path_ae))[0]

        test_score, _ = test_AE(modelAE, test_dataloader, device)
        fake_score, _ = test_AE(modelAE, fake_dataloader, device)

        plot_score(test_score, fake_score, model_name, save_path="plot/es1/")


if __name__ == "__main__":
    main()