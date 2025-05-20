import argparse
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from model import CNN, CNN2, Autoencoder
from utils import get_dataloaders, str2bool, denorm, fgsm_attack


"""
Plots the model accuracy or reconstruction loss under FGSM attack across a range of epsilon values,
and displays a grid of example adversarial images for each epsilon.

Args:
    epsilons (list or array): Sequence of epsilon values used for the FGSM attack.
    examples (list): Nested list of example tuples for each epsilon. Each example is either:
                     - For CNN: (original_label, adversarial_label, adversarial_image)
                     - For AE: (original_loss, adversarial_loss, adversarial_image)
    metric (list or array): Accuracy or reconstruction loss values corresponding to each epsilon.
    model_name (str): Name of the model (used for plot titles and saving files).
    save_path (str, optional): Directory to save the plots. Default is "plot/".
    ty (str, optional): Type of model — "CNN" or "AE". Determines plot labels and example captions based on the model used.
"""
def plot_result(epsilons, examples, metric, model_name, save_path="plot/", ty=None):

    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, metric, "*-")

    if ty == "CNN":
        plt.title("Accuracy vs Epsilon Model:" + model_name)
    else:
        plt.title("Reconstruction Loss vs Epsilon Model:" + model_name)

    plt.xlabel("Epsilon")
    plt.ylabel("MSE Loss")
    plt.savefig(save_path + 'FGSM_eps_'+model_name+'.png')

    cnt = 0
    plt.figure(figsize=(12, 12))
    plt.suptitle("Model: " + model_name, fontsize=16)
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
    plt.savefig(save_path + 'FGSM_EXAMPLE_IMG_' + model_name + '.png')


"""
Performs a test of a CNN model’s robustness against FGSM adversarial attacks.

Args:
    model (nn.Module): The CNN model to test.
    device (torch.device): Device to run the computations on.
    test_loader (DataLoader): DataLoader for the test dataset.
    epsilon (float): Perturbation magnitude for the FGSM attack.
    mean (list or tuple): Mean used for data normalization.
    std (list or tuple): Standard deviation used for data normalization.

Returns:
    final_acc (float): Accuracy of the model on the adversarially perturbed test set.
    adv_examples (list): A list of up to 5 adversarial examples in the form (original_label, predicted_label, perturbed_image).
    
This implementation is based on https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html with some modifications
"""
def test( model, device, test_loader, epsilon, mean, std):
    correct = 0
    adv_examples = []
    tqdm_bar = tqdm(test_loader, total=len(test_loader), desc = "[FGSM attack epsilon:" + str(epsilon) + "]", leave=False)
    for data, target in tqdm_bar:
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

    return final_acc, adv_examples


"""
Performs a test of a CNN model’s robustness against FGSM adversarial attacks.

Args:
    model (nn.Module): The CNN model to test.
    device (torch.device): Device to run the computations on.
    test_loader (DataLoader): DataLoader for the test dataset.
    epsilon (float): Perturbation magnitude for the FGSM attack.
    mean (list or tuple): Mean used for data normalization.
    std (list or tuple): Standard deviation used for data normalization.

Returns:
    avg_loss (float): Mean of the Reconstruction error (MSELoss) over all perturbed data in the test_loader
    adv_examples (list): A list of up to 5 adversarial examples in the form (original_loss, adversarial_loss, perturbed_image).

"""
def test_autoencoder(model, device, test_loader, epsilon, mean, std):
    total_loss = 0
    adv_examples = []
    tqdm_bar = tqdm(test_loader, total=len(test_loader), desc="[FGSM attack epsilon:" + str(epsilon) + "]", leave=False)
    for data, _ in tqdm_bar:
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

    return avg_loss, adv_examples


"""
Performs a targeted FGSM attack on a CNN model aiming to misclassify inputs into a specific target class.

Args:
    model (torch.nn.Module): The CNN model to attack.
    test_loader (DataLoader): DataLoader for the test dataset.
    epsilon (float): Magnitude of the FGSM perturbation (positive value). The attack uses -epsilon to push towards the target class.
    target_class (int): The target class index for the targeted attack (exercise 3.3).
    mean (list or tensor): Mean used for input normalization.
    std (list or tensor): Standard deviation used for input normalization.
    device (torch.device): Device to perform computations on (CPU or GPU).

Returns:
    float: The targeted attack success rate, i.e., the fraction of samples successfully misclassified as the target class.
"""
def test_targeted_fgsm(model, test_loader, epsilon, target_class, mean, std, device):

    targeted_success = 0
    tqdm_bar = tqdm(test_loader, total=len(test_loader), desc="[Targeted FGSM epsilon:"+str(-epsilon)+" , target_class:"+str(target_class)+"]", leave=False)
    for data, target in tqdm_bar:
        data, target = data.to(device), target.to(device)

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

        targeted_success += (final_pred == target_labels).sum().item()


    total = len(test_loader.dataset)
    targeted_acc = targeted_success / float(total)

    return targeted_acc

"""
Plots the targeted FGSM attack success rate across different epsilon values and saves the plot.

Args:
    attack_success_rate (list or array): List of attack success rates corresponding to each epsilon.
    epsilons (list or array): List of epsilon values used for the FGSM attack.
    target_class (int): The target class index for the attack.
    model_name (str): Name of the model being evaluated.
    save_path (str, optional): Directory path where the plot image will be saved. Defaults to "plot/".

Saves:
    A plot image named 'FGSM_SUCCESS_RATE_TARGET_<target_class>_<model_name>.png' in the specified save_path.
"""
def plot_target_attack(attack_success_rate, epsilons, target_class, model_name, save_path="plot/"):

    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, attack_success_rate, marker="o")
    plt.xlabel("Epsilon")
    plt.ylabel("Attack Success Rate")
    plt.title("Attack Success Rate per Epsilon - Target class:" + str(target_class) + "-Model:" + model_name)
    plt.ylim([0, 1])
    plt.savefig(save_path + 'FGSM_SUCCESS_RATE_TARGET_'+str(target_class)+'_' + model_name + '.png')

def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter and model configuration settings")

    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for data loading")

    # Autoencoder (AE) settings
    parser.add_argument("--ae", type=str2bool, default=False, help="If True, use the Autoencoder (AE) model")
    parser.add_argument("--model_path_ae", type=str, default="Models/AE_pretrain.pth", help="Path to the pretrained AE model")

    # CNN settings
    parser.add_argument("--cnn", type=str2bool, default=True, help="If True, use the CNN model")
    parser.add_argument("--cnn_ty", type=str2bool, default=True, help="If True, use the more powerful CNN model; if False, use the CNN2 model")
    parser.add_argument("--model_path_cnn", type=str, default="Models/CNN_pretrain.pth", help="Path to the pretrained CNN model")

    parser.add_argument("--target_class", type=int, default=0, help="Target class index for exercise 3.3")

    args = parser.parse_args()

    return args

def main():
    args = get_parser()

    # Get CIFAR10 dataloaders (see utilis.py)
    _, _, test_dataloader, _, _ = get_dataloaders(1, num_workers=args.num_workers) # Use batch size 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    # Test the CNN or CNN2 model
    if args.cnn:

        if args.cnn_ty:
            modelCNN = CNN().to(device)
        else:
            modelCNN = CNN2().to(device)

        modelCNN.load_state_dict(torch.load(args.model_path_cnn))

        accuracies = []
        examples = []
        attack_success_rate = []

        epsilons = [0, .05, .075, .1, .125, .15]

        tqdm_bar = tqdm(epsilons, total = len(epsilons), desc="[All Epsilon]")
        for eps in tqdm_bar:
            acc, ex = test(modelCNN, device, test_dataloader, eps, mean, std) # Attack the CNN/CNN2 model
            targeted_acc = test_targeted_fgsm(modelCNN, test_dataloader, eps, args.target_class, mean, std, device) # Exercise 3.3
            attack_success_rate.append(targeted_acc)
            accuracies.append(acc)
            examples.append(ex)
            tqdm_bar.set_postfix(epsilon=f"{eps}", test_accuracy=f"{acc:.4f}", targeted_attack_success_rate=f"{targeted_acc:.4f}")

        model_name = os.path.splitext(os.path.basename(args.model_path_cnn))[0] # get the model name by the path of the model
        plot_result(epsilons, examples, accuracies, model_name, save_path="plot/es2/", ty="CNN")

        # Exercise 3.3 plot
        plot_target_attack(attack_success_rate, epsilons, args.target_class, model_name, save_path="plot/es3/")
        print("All plot are saved in " + "plot/es2/" + "***_" + model_name)

    # Test the autoencoder model
    if args.ae:
        modelAE = Autoencoder().to(device)
        modelAE.load_state_dict(torch.load(args.model_path_ae))

        losses = []
        examples = []
        epsilons = [0, .05, .075, .1, .125, .15, .175, .2]

        tqdm_bar = tqdm(epsilons, total=len(epsilons), desc="[All Epsilon]")
        for eps in tqdm_bar:
            loss, ex = test_autoencoder(modelAE, device, test_dataloader, eps, mean, std) # Attack the Autoencoder model
            losses.append(loss)
            examples.append(ex)
            tqdm_bar.set_postfix(epsilon=f"{eps}", avg_reconstruction_loss=f"{loss:.4f}")

        model_name = os.path.splitext(os.path.basename(args.model_path_ae))[0] # get the model name by the path of the model
        plot_result(epsilons, examples, losses, model_name, save_path="plot/es2/", ty="AE")
        print("All plot are saved in " + "plot/es2/" + "***_" + model_name)

if __name__ == "__main__":
    main()