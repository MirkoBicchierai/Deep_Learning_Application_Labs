import argparse
import wandb
from tqdm import tqdm
from model import CNN, Autoencoder, CNN2
from utils import str2bool, config_loggers, get_dataloaders, train_CNN, test_CNN, train_AE, test_AE
import torch

def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    #wandb
    parser.add_argument("--exp_name", type=str, default="_New", help="Name of the experiment for Weights & Biases logging")

    # Pretraining parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, default=200, help="Total number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # FGSM attack as data augmentation during training
    parser.add_argument("--aug_fgsm", type=str2bool, default=True, help="Enable FGSM-based data augmentation during training")
    parser.add_argument("--rand_epsilon", type=str2bool, default=False, help="Use a random epsilon value between 0.01 and 0.2 for FGSM")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Fixed epsilon value for FGSM (ignored if --rand_epsilon is True)")

    # CNN pretraining
    parser.add_argument("--train_cnn", type=str2bool, default=True, help="Enable training of the CNN model")
    parser.add_argument("--cnn_ty", type=str2bool, default=False, help="If True, use the more powerful CNN model; if False, use CNN2")

    # Autoencoder pretraining
    parser.add_argument("--train_AE", type=str2bool, default=False, help="Enable training of the autoencoder model")

    args = parser.parse_args()

    return args

def main():
    args = get_parser()
    config_loggers(args)

    train_dataloader, val_dataloader, test_dataloader, num_classes, input_size = get_dataloaders("CIFAR10", args.batch_size, num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    add_name = "rand" if args.rand_epsilon else str(args.epsilon)

    # Train the CNN
    if args.train_cnn:

        if args.cnn_ty:
            model = CNN().to(device)
            model_name = "CNN_pretrain"
        else:
            model = CNN2().to(device)
            model_name = "CNN2_pretrain"

        if args.aug_fgsm:
            model_name = model_name + "_aug_" + add_name

        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        # Train loop
        train_bar = tqdm(range(args.epochs), desc=f"[Training epochs]")
        for epoch in train_bar:
            loss_clean, loss_adv = train_CNN(model, train_dataloader, opt, device, epoch, args)
            accuracy, _, val_loss = test_CNN(model, val_dataloader, device)
            scheduler.step()

            wandb.log({"Train-CNN": {"Loss": loss_clean, "Loss_adv": loss_adv, "epoch": epoch},
                       "Validation-CNN": {"Loss": val_loss, "Accuracy": accuracy, "epoch": epoch}})

            train_bar.set_postfix(epoch_loss_clean=f"{loss_clean:.4f}", epoch_loss_adv=f"{loss_adv:.4f}")

        accuracy, _, _ = test_CNN(model, test_dataloader, device)
        wandb.log({"Test-CNN": {"Accuracy": accuracy}})
        print("Test accuracy CNN: {}".format(accuracy))

        torch.save(model.state_dict(), "Models/" + model_name + ".pth")

    # Train the autoencoder
    if args.train_AE:

        model_name = "AE_pretrain"
        if args.aug_fgsm:
            model_name = model_name + "_aug_" + add_name

        model = Autoencoder().to(device)
        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        # Train loop
        train_bar = tqdm(range(args.epochs), desc=f"[Training epochs]")
        for epoch in train_bar:
            loss_clean, loss_adv = train_AE(model, train_dataloader, opt, device, epoch, args)
            _, val_loss = test_AE(model, val_dataloader, device)
            scheduler.step()

            wandb.log({"Train-AE": {"Loss": loss_clean, "Loss_adv": loss_adv, "epoch": epoch},
                       "Validation-AE": {"Loss": val_loss, "epoch": epoch}})

            train_bar.set_postfix(epoch_loss=f"{loss_clean:.4f}", epoch_loss_adv=f"{loss_adv:.4f}")

        torch.save(model.state_dict(), "Models/"+model_name+".pth")


if __name__ == "__main__":
    main()