
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FakeData
from tqdm import tqdm
import torch.nn.functional as F
import wandb


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False

    return None

def get_fake_loaders(batch_size, num_workers):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    fake_dataset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
    fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return fake_dataloader

def get_dataloaders(name, batch_size, num_workers):
    ds_train, ds_test, num_classes, input_size = None, None, 0, 0

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
Function to define the wandb parameters for the main.py 
"""
def config_loggers(args):

    exp_name = "_New"

    wandb.init(
        project="DLA_LAB_4",
        config=vars(args),
        name=exp_name,
    )

