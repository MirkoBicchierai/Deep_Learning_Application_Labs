import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from model import CNN, CNN2, Autoencoder
from utils import get_dataloaders, str2bool, denorm, fgsm_attack


def test( model, device, test_loader, epsilon, mean, std):
    correct = 0
    adv_examples = []
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.cross_entropy(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(data, mean, std, device)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize(mean, std)(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1

        # Save some adv examples for visualization later
        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    return final_acc, adv_examples


def test_autoencoder(model, device, test_loader, epsilon, mean, std):
    total_loss = 0
    adv_examples = []
    for data, _ in tqdm(test_loader):
        data = data.to(device)
        data.requires_grad = True

        # Forward pass
        _, x_rec = model(data)
        loss = F.mse_loss(x_rec, data)
        model.zero_grad()
        loss.backward()

        # Gradient on input
        data_grad = data.grad.data

        # Denormalize for attack
        data_denorm = denorm(data, mean, std, device)

        # Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Re-normalize for model input
        perturbed_data_normalized = transforms.Normalize(mean, std)(perturbed_data)

        # Forward pass with perturbed data
        _, rec_adv = model(perturbed_data_normalized)

        # Calculate reconstruction loss
        loss_adv = F.mse_loss(rec_adv, perturbed_data_normalized)
        total_loss += loss_adv.item()

        # Save some adv examples for visualization later
        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((loss.item(), loss_adv.item(), adv_ex))

    avg_loss = total_loss / len(test_loader)
    print(f"Epsilon: {epsilon}\tAvg Reconstruction Loss = {avg_loss}")

    return avg_loss, adv_examples


def plot_result(epsilons, examples, metric, ty=None):
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, metric, "*-")
    plt.title("Reconstruction Loss vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("MSE Loss")
    plt.show()

    cnt = 0
    plt.figure(figsize=(12, 12))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)

            if ty=="CNN":
                orig, adv, ex = examples[i][j]
                plt.title(f"{orig} -> {adv}")
                plt.imshow(np.transpose(ex, (1, 2, 0)))
            if ty == "AE":
                orig_loss, adv_loss, ex = examples[i][j]
                plt.title(f"MSE: {orig_loss:.3f}->{adv_loss:.3f}")
                plt.imshow(np.transpose(ex, (1, 2, 0)))

    plt.tight_layout()
    plt.show()


def test_targeted_fgsm(model, args, test_loader, epsilon, target_class, mean, std, device):
    correct = 0
    targeted_success = 0
    adv_examples = []

    for data, target in tqdm(test_loader):
        data, target = data.to(args.device), target.to(args.device)

        # Skip batch if any of the samples are already predicted as the target class
        with torch.no_grad():
            output = model(data)
            pred = output.max(1)[1]
            if (pred == target_class).any():
                continue

        data.requires_grad = True

        # Replace ground truth label with target label
        target_labels = torch.full_like(target, target_class)

        output = model(data)
        loss = F.cross_entropy(output, target_labels)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        data_denorm = denorm(data, mean, std, device)

        # Generate adversarial example targeting `target_class`
        perturbed_data = fgsm_attack(data_denorm, -epsilon, data_grad)  # Use -ε for targeted attack
        perturbed_data_norm = transforms.Normalize(mean, std)(perturbed_data)

        output = model(perturbed_data_norm)
        final_pred = output.max(1)[1]

        # Count correctly classified original examples
        correct += (final_pred == target).sum().item()
        targeted_success += (final_pred == target_labels).sum().item()

        # Save some examples
        for i in range(len(data)):
            if len(adv_examples) < 5:
                adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                adv_examples.append((target[i].item(), final_pred[i].item(), adv_ex))

    total = len(test_loader.dataset)
    final_acc = correct / float(total)
    targeted_acc = targeted_success / float(total)

    print(f"Epsilon: {epsilon}")
    print(f"Standard Accuracy: {correct}/{total} = {final_acc:.4f}")
    print(
        f"Targeted Attack Success Rate (class {target_class}): {targeted_success}/{total} = {targeted_acc:.4f}"
    )

    return final_acc, targeted_acc, adv_examples

def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker")

    # Pretrain AE
    parser.add_argument("--ae", type=str2bool, default=False, help="If True use the AE Model")
    parser.add_argument("--model_path_ae", type=str, default="Models/AE_pretrain.pth", help="Path of the AE model pretrained")

    # Pretrain CCN
    parser.add_argument("--cnn", type=str2bool, default=True, help="If True use CNN Model")
    parser.add_argument("--cnn_ty", type=str2bool, default=True, help="If True use the CNN Model (more power), False use CNN2 Model")
    parser.add_argument("--model_path_cnn", type=str, default="Models/CNN_pretrain_aug_rand.pth", help="Path of the CNN model pretrained")

    args = parser.parse_args()

    return args

def main():
    args = get_parser()

    _, _, test_dataloader, _, _ = get_dataloaders("CIFAR10", 1, num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    if args.cnn:

        if args.cnn_ty:
            modelCNN = CNN().to(device)
        else:
            modelCNN = CNN2().to(device)

        modelCNN.load_state_dict(torch.load(args.model_path_cnn))

        accuracies = []
        examples = []
        epsilons = [0, .05, .075, .1, .125, 1.5]

        for eps in tqdm(epsilons):
            acc, ex = test(modelCNN, device, test_dataloader, eps, mean, std)
            accuracies.append(acc)
            examples.append(ex)

        plot_result(epsilons, examples, accuracies, ty="CNN")

    if args.ae:
        modelAE = Autoencoder().to(device)
        modelAE.load_state_dict(torch.load(args.model_path_ae))

        losses = []
        examples = []
        epsilons = [0, .05, .075, .1, .125, .15, .175, .2]

        for eps in tqdm(epsilons):
            loss, ex = test_autoencoder(modelAE, device, test_dataloader, eps, mean, std)
            losses.append(loss)
            examples.append(ex)

        plot_result(epsilons, examples, losses, ty="AE")


if __name__ == "__main__":
    main()