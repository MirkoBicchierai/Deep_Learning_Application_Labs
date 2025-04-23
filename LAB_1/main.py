import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
from cnn_model import CNN
from mlp_model import MLP
from utils import get_dataloaders, str2bool, train, test, config_loggers_ex1
import argparse
import matplotlib.pyplot as plt

"""
Computes and visualizes the gradient norms of each layer's weights and biases 
for a single minibatch.

This is useful for diagnosing vanishing or exploding gradients during training.
Gradients are computed via backpropagation on one batch from the provided dataloader.
A bar plot of the L2 norms of gradients is generated and logged to Weights & Biases (wandb).
"""

def gradient_norm(model, dataloader, device):

    data, labels = next(iter(dataloader))
    data = data.to(device)
    labels = labels.to(device)
    model.zero_grad()
    logits = model(data)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    grad_weights = {}
    grad_biases = {}
    for name, param in model.named_parameters():
        print(name)
        if param.grad is not None:
            if "weight" in name:
                grad_weights[name] = param.grad.norm().item()
            elif "bias" in name:
                grad_biases[name] = param.grad.norm().item()

    sorted_weight_layers = sorted(grad_weights.keys())
    sorted_bias_layers = sorted(grad_biases.keys())

    weight_norms = [grad_weights[layer] for layer in sorted_weight_layers]
    bias_norms = [grad_biases[layer] for layer in sorted_bias_layers]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(sorted_weight_layers)), weight_norms, color="#4c72b0", label="Weights", alpha=1)
    plt.bar(range(len(sorted_bias_layers)), bias_norms, color="#dd8452", label="Biases", alpha=0.8)

    plt.xlabel("Layers")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm / Layer")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    wandb.log({"Gradient Norm Plot": wandb.Image(plt)})


def get_parser():

    parser = argparse.ArgumentParser(description="Hyperparameter settings")
    parser.add_argument("--epochs", type=int, default=50, help="Number of train epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of the dataloaders")

    #General Model Parameters
    parser.add_argument("--MLP", type=str2bool, default=True, help="If True use MLP, if False use CNN")
    parser.add_argument("--residual", type=str2bool, default=True, help="If True use skip connection")
    parser.add_argument("--scheduler", type=str2bool, default=True, help="If True use a cosine scheduler")

    #MLP parameters
    parser.add_argument("--normalization", type=str2bool, default=True, help="If True use normalization layers (this choice is only available for the MLP model)")
    parser.add_argument("--depth", type=int, default=40, help="Number of layers in the MLP")
    parser.add_argument("--width", type=int, default=64, help="Number of hidden dimension for all the layer in the MLP")

    #CNN parameters, [2, 2, 2, 2] like a ResNet18, [3, 4, 6, 3] like a resnet34, [5, 6, 8, 5] like a resnet50
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 2, 2, 2], help="Number of layer pattern for the CNN Model")

    #Dataset
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10"], default="MNIST", help="Choose dataset from MNIST, CIFAR10")

    args = parser.parse_args()

    return args


def main():

    args = get_parser()
    model_name_save = config_loggers_ex1(args) # Config the wandb parameters see the utils.py file

    model_type = "MLP" if args.MLP else "CNN"
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # Get the train/val/test dataloaders and the num_classes and the input_size of the data (see utils.py)
    train_dataloader, val_dataloader, test_dataloader, num_classes, input_size = get_dataloaders(args.dataset, args.batch_size, num_workers = 12)

    if args.MLP:
        model = MLP(input_size=input_size, depth=args.depth, width=args.width, classes=num_classes, residual=args.residual, normalization=args.normalization).to(device)
    else:
        model = CNN(args.layers, num_classes=num_classes, residual=args.residual).to(device)

    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Train loop
    train_bar = tqdm(range(args.epochs), desc=f"[Training epochs]")
    for epoch in train_bar:
        loss = train(model, train_dataloader, opt, device, epoch, args.epochs)
        accuracy, _, val_loss = test(model, val_dataloader, device)

        if args.scheduler:
            scheduler.step()

        wandb.log({"Train-" + model_type + "-" + args.dataset: {"Loss": loss, "epoch": epoch}, "Validation-" + model_type + "-" + args.dataset: {"Loss": val_loss, "Accuracy":accuracy, "epoch": epoch}})
        train_bar.set_postfix(epoch_loss=f"{loss:.4f}")

    accuracy, _, _ = test(model, test_dataloader, device)
    wandb.log({"Test-" + model_type + "-" + args.dataset: {"Accuracy": accuracy}})
    print("Test accuracy: {}".format(accuracy))

    gradient_norm(model, train_dataloader, device)

    torch.save(model.state_dict(), "Models/"+model_name_save+".pth")

if __name__ == "__main__":
    main()