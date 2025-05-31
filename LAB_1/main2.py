import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import wandb
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm
from cnn_model import CNN
from utils import get_dataloaders, str2bool, train, test, config_loggers_ex2
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    parser.add_argument("--epochs", type=int, default=75, help="Number of train epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of the dataloaders")

    parser.add_argument("--num_layers", type=int, default=0, help="Number of the early layers to unfreeze, if 0 is a classical linear evaluation protocol")
    parser.add_argument("--scheduler", type=str2bool, default=True, help="If True activate the cosine scheduler")

    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"], default="SGD", help="Choose optimizer from SGD or ADAM")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for the optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")

    parser.add_argument("--path", type=str, default="Models/8-CNN-Residual-Scheduler.pth", help="Path to the pretrained model of the CNN on CIFAR10")

    # CNN parameters
    parser.add_argument("--residual", type=str2bool, default=True, help="True if the pretrained model use the skip connection trick.")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 2, 2, 2], help="Layer pattern of the pretrained model loaded.") #[5, 6, 8, 5]

    # Use wandb
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="If True use wandb")

    args = parser.parse_args()

    return args

"""
Function that return all the data of a dataloader as a feature of a model, using the specific method model.get_feature(data) 
that return the output of the last layer before the final fully connect layer, avgpool layer feature.
"""
def feature_extractor(model, dataloader, device):
    model.eval()
    extractor_bar = tqdm(dataloader, desc=f"[Feature extractor]")
    features = []
    true_labels = []

    for data, labels in extractor_bar:
        data = data.to(device)
        with torch.no_grad():
            logits = model.get_feature(data)
            features.append(logits)
            true_labels.append(labels)

    features = torch.cat(features, dim=0)
    true_labels = torch.cat(true_labels, dim=0)

    return features.detach().cpu(), true_labels.detach().cpu()

"""
Function that perform a Linear SVM from sklearn as baseline
"""
def base_test(train_features, train_labels, val_features, val_labels, test_features, test_labels):

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    val_features = np.array(val_features)
    val_labels = np.array(val_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_val_scaled = scaler.transform(val_features)
    X_test_scaled = scaler.transform(test_features)

    # Train SVM
    clf = SVC(kernel='linear', max_iter=1000) # Set a max-iteration to 100 to don't have a time explosion
    clf.fit(X_train_scaled, train_labels)

    val_preds = clf.predict(X_val_scaled)
    test_preds = clf.predict(X_test_scaled)

    return  accuracy_score(val_labels, val_preds), accuracy_score(test_labels, test_preds)

"""
This function configures a pre-trained model for fine-tuning on a new classification task.
    1. It first freezes all layers in the model by setting `requires_grad` to False.
    2. Then, if `num_layers` > 0, it selectively unfreezes the first `num_layers` layers of the model's backbone
       (useful for partial fine-tuning).
    3. It replaces the final fully connected (fc) layer with a new one with output dimension 'num_classes'.
    4. The new fc layer's weights are initialized with a normal distribution (mean=0, std=0.01),
       and its biases are initialized to zero.
    5. The new fc layer is set to require gradients, enabling it to be trained.
"""
def setup_model(model, num_classes, num_layers):
    for param in model.parameters():
        param.requires_grad = False

    if num_layers > 0:
        backbone_layers = list(model.children())
        for i, layer in enumerate(backbone_layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = True
                print(f"Unfrozen layer {i}: {layer}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.weight = nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    model.fc.bias = nn.init.zeros_(model.fc.bias)
    model.fc.requires_grad = True


def main():

    args = get_parser()
    config_loggers_ex2(args) # Config the wandb parameters see the utils.py file
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    train_dataloader, val_dataloader, test_dataloader, num_classes, input_size = get_dataloaders("CIFAR100", args.batch_size, num_workers=12)

    # Load the pretrained CNN model trained on CIFAR10 (10 classes)
    model = CNN(args.layers, num_classes=10, residual=True).to(device)
    model.load_state_dict(torch.load(args.path, map_location=device))

    # Extract the feature of the all split of the dataset using the pretrained model
    train_features, train_labels = feature_extractor(model, train_dataloader, device)
    validation_features, validation_labels = feature_extractor(model, val_dataloader, device)
    test_features, test_labels = feature_extractor(model, test_dataloader, device)

    #Get the accuracy on the validation/test set using a Linear SVM on the extracted feature
    val_accuracy, test_accuracy = base_test(train_features, train_labels, validation_features, validation_labels , test_features, test_labels)
    print(f"Validation Accuracy: {val_accuracy}") # 0.2062
    print(f"Test Accuracy: {test_accuracy}") # 0.1951

    setup_model(model, num_classes, num_layers=args.num_layers)

    #Optimizer can be selected with the parser
    opt = None
    if args.optimizer == "SGD":
        opt = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.optimizer == "Adam":
        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Loop to fine tune the pretrained model
    train_bar = tqdm(range(args.epochs), desc=f"[Fine tuning epochs]")
    for epoch in train_bar:
        loss = train(model, train_dataloader, opt, device, epoch, args.epochs)
        top1, top5, val_loss = test(model, val_dataloader, device)

        if args.scheduler:
            scheduler.step()

        if args.use_wandb:
            wandb.log({"Exercise2/Validation": {"Top1-Accuracy": top1, "Top5-Accuracy": top5, "Loss": val_loss, "Epoch": epoch},"Exercise2/Train": {"Loss": loss, "Epoch": epoch}})
        train_bar.set_postfix(epoch_loss=f"{loss:.4f}")

    top1, top5, _ = test(model, test_dataloader, device)

    if args.use_wandb:
        wandb.log({"Exercise2/Test": {"Top1-Accuracy": top1, "Top5-Accuracy": top5}})
    print("Test accuracy: {}".format(top1))

if "__main__" == __name__:
    main()