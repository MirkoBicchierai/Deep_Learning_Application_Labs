import argparse
import wandb
from tqdm import tqdm
from model import CNN, Autoencoder, CNN2
from utils import str2bool, config_loggers, get_dataloaders, train_CNN, test_CNN, train_AE, test_AE
import torch

def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    #Pretrain parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # Pretrain CCN
    parser.add_argument("--train_cnn", type=str2bool, default=True, help="If True Train the CNN Model")
    parser.add_argument("--cnn_ty", type=str2bool, default=True, help="If True use the CNN Model (more power), False use CNN2 Model")
    # Pretrain AutoEncoder
    parser.add_argument("--train_AE", type=str2bool, default=False, help="If True Train the AE Model")

    args = parser.parse_args()

    return args

def main():
    args = get_parser()
    config_loggers(args)

    train_dataloader, val_dataloader, test_dataloader, num_classes, input_size = get_dataloaders("CIFAR10", args.batch_size, num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_cnn:

        if args.cnn_ty:
            model = CNN().to(device)
            model_name = "CNN_pretrain"
        else:
            model = CNN2().to(device)
            model_name = "CNN2_pretrain"

        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        # Train loop
        train_bar = tqdm(range(args.epochs), desc=f"[Training epochs]")
        for epoch in train_bar:
            loss = train_CNN(model, train_dataloader, opt, device, epoch, args.epochs)
            accuracy, _, val_loss = test_CNN(model, val_dataloader, device)
            scheduler.step()

            wandb.log({"Train-CNN": {"Loss": loss, "epoch": epoch},
                       "Validation-CNN": {"Loss": val_loss, "Accuracy": accuracy, "epoch": epoch}})

            train_bar.set_postfix(epoch_loss=f"{loss:.4f}")

        accuracy, _, _ = test_CNN(model, test_dataloader, device)
        wandb.log({"Test-CNN": {"Accuracy": accuracy}})
        print("Test accuracy CNN: {}".format(accuracy))

        torch.save(model.state_dict(), "Models/" + model_name + ".pth")


    if args.train_AE:
        model = Autoencoder().to(device)
        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        # Train loop
        train_bar = tqdm(range(args.epochs), desc=f"[Training epochs]")
        for epoch in train_bar:
            loss = train_AE(model, train_dataloader, opt, device, epoch, args.epochs)
            _, val_loss = test_AE(model, val_dataloader, device)
            scheduler.step()

            wandb.log({"Train-AE": {"Loss": loss, "epoch": epoch},
                       "Validation-AE": {"Loss": val_loss, "epoch": epoch}})

            train_bar.set_postfix(epoch_loss=f"{loss:.4f}")

        torch.save(model.state_dict(), "Models/AE_pretrain.pth")


if __name__ == "__main__":
    main()