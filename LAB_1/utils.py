import torch
from tqdm import tqdm

import torch.nn.functional as F
import wandb
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
from sklearn.metrics import accuracy_score


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
            transforms.Normalize((0.5,), (0.5,))
        ])
        ds_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
        ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        input_size = 32 * 32 * 3

    if name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
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


def test(model, dataloader, device):
    model.eval()
    predictions = []
    gts = []
    losses = []
    test_bar = tqdm(dataloader, desc=f"[Test/Validation]", leave=False)
    with torch.no_grad():
        for data, labels in test_bar:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            loss = F.cross_entropy(logits, labels)
            prob = F.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            gts.append(labels.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            losses.append(loss.item())
            test_bar.set_postfix(minibatch_loss=f"{loss.item():.4f}")

    return accuracy_score(np.hstack(gts), np.hstack(predictions)), np.mean(losses)


def train(model, dataloader, opt, device, epoch, epochs):
    model.train()
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False

    return None


def config_loggers(args):

    normalization = "with Normalization Layers" if args.normalization else ""
    model_type = str(args.depth) + "-" + str(args.width) + "-" + normalization + "-MLP" if args.MLP else str(sum(np.array(args.layers))) + "-CNN"
    res = "Residual" if bool(args.residual) else "No Residual"
    sch = "Scheduler" if args.scheduler else "No Scheduler"
    formatted_lr = f"{args.lr:.0e}".replace("e+00", "e+0").replace("e-00", "e-0")

    model_name_save = model_type + "-" + res + "-" + sch

    wandb.init(
        project="DLA_LAB_1",
        name=model_type + "-" + args.dataset + "-" + res + "-" + formatted_lr + "-" + sch,
        config=vars(args),
        group=res
    )

    return model_name_save
