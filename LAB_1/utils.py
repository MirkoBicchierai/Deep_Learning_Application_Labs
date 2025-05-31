import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
from sklearn.metrics import accuracy_score

"""
    Loads and prepares data loaders for the specified dataset.
    Supports MNIST, CIFAR10, and CIFAR100. Applies appropriate normalization and
    transforms to each dataset. Splits the training set into a training and 
    validation subset (validation set has 5000 samples).
    Returns:
        - train_dataloader: DataLoader for training data.
        - val_dataloader: DataLoader for validation data (subset of training set).
        - test_dataloader: DataLoader for test data.
        - num_classes (int): Number of output classes for the dataset.
        - input_size (int): Flattened size of a single input sample.
"""

def get_dataloaders(name, batch_size, num_workers):
    ds_train, ds_test, num_classes, input_size = None, None, 0, 0
    if name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds_train = MNIST(root='./data', train=True, download=True, transform=transform)
        ds_test = MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_size = 28 * 28

    if name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        ds_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
        ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_size = 32 * 32 * 3

    if name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        ds_train = CIFAR100(root='./data', train=True, download=True, transform=transform)
        ds_test = CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
        input_size = 32 * 32 * 3

    # Split train into train and validation.
    val_size = 5000
    indices = np.random.permutation(len(ds_train))
    ds_val = Subset(ds_train, indices[:val_size])
    ds_train = Subset(ds_train, indices[val_size:])

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, num_classes, input_size

"""
Evaluates the model on a test or validation dataset.
Performs inference, computes cross-entropy loss, and calculates top-1 and top-5 accuracy.

Returns:
    - top1_accuracy (float): Standard classification accuracy (top-1).
    - top5_accuracy (float): Top-5 accuracy (percentage of samples where the 
      correct class is among the top 5 predicted).
    - avg_loss (float): Average cross-entropy loss over the dataset.
"""
def test(model, dataloader, device):
    model.eval()
    predictions = []
    gts = []
    losses = []

    top5_correct = 0
    total_samples = 0

    test_bar = tqdm(dataloader, desc=f"[Test/Validation]", leave=False)
    with torch.no_grad():
        for data, labels in test_bar:
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss = F.cross_entropy(logits, labels)
            prob = F.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            # Top-5 accuracy calculation
            top5 = torch.topk(prob, k=5, dim=1).indices
            top5_correct += sum([labels[i].item() in top5[i] for i in range(labels.size(0))])
            total_samples += labels.size(0)

            gts.append(labels.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            losses.append(loss.item())
            test_bar.set_postfix(minibatch_loss=f"{loss.item():.4f}")

    top5_accuracy = top5_correct / total_samples
    top1_accuracy = accuracy_score(np.hstack(gts), np.hstack(predictions))
    avg_loss = np.mean(losses)

    return top1_accuracy, top5_accuracy, avg_loss

"""

Trains the model for one epoch on the given data.
Performs a forward pass, computes cross-entropy loss, backpropagates the error,
and updates model parameters using the provided optimizer. Tracks and returns
the average training loss over the epoch. 

Returns:
    float: Average training loss over the epoch.
    
"""
def train(model, dataloader, opt, device, epoch, epochs):
    model.train()
    model = model.to(device)
    losses = []
    train_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", leave=False)
    for data, labels in train_bar:
        data = data.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        train_bar.set_postfix(minibatch_loss=f"{loss.item():.4f}")
    return np.mean(losses)

"""
Function used to define the type of an attribute inside of the parser for the main, to parse for example '--Residual true'
"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False

    return None

"""
Function to define the wandb parameters for the main.py (Exercise 1)
"""
def config_loggers_ex1(args):

    normalization = "with Normalization Layers" if args.normalization else ""
    model_type = str(args.depth) + "-" + str(args.width) + "-" + normalization + "-MLP" if args.MLP else str(sum(np.array(args.layers))) + "-CNN"
    res = "Residual" if bool(args.residual) else "No Residual"
    sch = "Scheduler" if args.scheduler else "No Scheduler"
    formatted_lr = f"{args.lr:.0e}".replace("e+00", "e+0").replace("e-00", "e-0")

    model_name_save = model_type + "-" + res + "-" + sch

    if args.use_wandb:
        wandb.init(
            project="DLA_LAB_1",
            name=model_type + "-" + args.dataset + "-" + res + "-" + formatted_lr + "-" + sch,
            config=vars(args),
            group=res
        )

    return model_name_save

"""
Function to define the wandb parameters for the main2.py (Exercise 2)
"""

def config_loggers_ex2(args):

    model_type = str(sum(np.array(args.layers))) + "-CNN"
    res = "Residual" if bool(args.residual) else "No Residual"
    sch = "Scheduler" if args.scheduler else "No Scheduler"
    formatted_lr = f"{args.lr:.0e}".replace("e+00", "e+0").replace("e-00", "e-0")

    if args.use_wandb:
        wandb.init(
            project="DLA_LAB_1",
            name = model_type + "-" + res + "-" + formatted_lr + "-" + sch + "-" + args.optimizer + "-FLayer" + str(args.num_layers),
            config=vars(args),
            group="Fine Tuning"
        )