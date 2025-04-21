import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch import device
from tqdm import tqdm
from cnn_model import CNN
from utils import get_dataloaders, str2bool, train, test
import argparse

def get_parser():

    parser = argparse.ArgumentParser(description="Hyperparameter settings")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3) #  1e-2, 1e-3
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--residual", type=str2bool, default=True) # True/False
    parser.add_argument("--scheduler", type=str2bool, default=True) # True/False

    #CNN parameters
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 2, 2, 2]) # [2, 2, 2, 2], [3, 4, 6, 3], [5, 6, 8, 5]

    args = parser.parse_args()

    return args


def feature_extractor(model, dataloader):
    model.eval()
    extractor_bar = tqdm(dataloader, desc=f"[Feature extractor]")
    features = []
    labels = []
    for data, labels in extractor_bar:
        data = data.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(data)
            features.append(logits)
            labels.append(labels)

    features = torch.cat(features, dim = 0)
    labels = torch.cat(labels, dim=0)

    return features, labels

def knn_test(train_features, train_labels, test_features, test_labels):

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Normalize features (important for KNN)
    train_features_mean = train_features.mean(axis=0)
    train_features_std = train_features.std(axis=0)
    train_features_normalized = (train_features - train_features_mean) / train_features_std
    val_features_normalized = (test_features - train_features_mean) / train_features_std

    knn = KNeighborsClassifier(n_neighbors=200)
    knn.fit(train_features_normalized, train_labels)

    # Predict on validation set
    val_predictions = knn.predict(val_features_normalized)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, val_predictions)

    return accuracy


def main():

    args = get_parser()
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    train_dataloader, val_dataloader, test_dataloader, num_classes, input_size = get_dataloaders("CIFAR100", args.batch_size, num_workers=12)

    model = CNN(args.layers, num_classes=10, residual=True).to(device)
    model.load_state_dict(torch.load("Models/model.pth"))

    train_features, train_labels = feature_extractor(model, train_dataloader)
    validation_features, validation_labels = feature_extractor(model, val_dataloader)
    test_features, test_labels = feature_extractor(model, test_dataloader)

    val_accuracy = knn_test(train_features, train_labels, validation_features, validation_labels)
    test_accuracy = knn_test(train_features, train_labels, test_features, test_labels)

    print(f"Validation accuracy: {val_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_bar = tqdm(range(args.epochs), desc=f"[Fine tuning epochs]")
    for epoch in train_bar:
        loss = train(model, train_dataloader, opt, device, epoch, args.epochs)
        accuracy, val_loss = test(model, val_dataloader, device)

        if args.scheduler:
            scheduler.step()

        train_bar.set_postfix(epoch_loss=f"{loss:.4f}")