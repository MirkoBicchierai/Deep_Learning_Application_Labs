import argparse

import wandb
from tqdm import tqdm

from model import CNN
from utils import str2bool, config_loggers, get_dataloaders, train, test, get_fake_loaders
import torch
import torchvision
from torchvision.datasets import FakeData
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
from sklearn import metrics

def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    #Dataset choice
    parser.add_argument("--dataset", type=str, default="rotten_tomatoes", help="Possible choose stanfordnlp/sst2 or rotten_tomatoes")

    # Exercise 1.3
    parser.add_argument("--check_baseLine", type=str2bool, default=False, help="If True compute the baseline using a linear SVM (Exercise 1.3)")

    args = parser.parse_args()

    return args

def main():
    args = get_parser()
    config_loggers(args)

    batch_size = 256
    num_workers = 12

    train_dataloader, val_dataloader, test_dataloader, num_classes, input_size = get_dataloaders("CIFAR10", batch_size, num_workers=num_workers)
    fake_dataloader = get_fake_loaders(batch_size, num_workers=num_workers)

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN().to(device)

    epochs = 200
    lr = 1e-4
    opt = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Train loop
    train_bar = tqdm(range(epochs), desc=f"[Training epochs]")
    for epoch in train_bar:
        loss = train(model, train_dataloader, opt, device, epoch, epochs)
        accuracy, _, val_loss = test(model, val_dataloader, device)
        scheduler.step()

        wandb.log({"Train-ES1": {"Loss": loss, "epoch": epoch},
                   "Validation-ES1": {"Loss": val_loss, "Accuracy": accuracy, "epoch": epoch}})

        train_bar.set_postfix(epoch_loss=f"{loss:.4f}")

    accuracy, _, _ = test(model, test_dataloader, device)
    wandb.log({"Test-ES1": {"Accuracy": accuracy}})
    print("Test accuracy: {}".format(accuracy))

    torch.save(model.state_dict(), "Models/CNN_pretrain.pth")



if __name__ == "__main__":
    main()