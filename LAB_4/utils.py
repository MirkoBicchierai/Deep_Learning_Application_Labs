import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, FakeData
from tqdm import tqdm
import torch.nn.functional as F
import wandb

"""
Performs an FGSM (Fast Gradient Sign Method) attack on an input image using a specified epsilon value.

This implementation follows the official PyTorch tutorial:
https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

Args:
    image (Tensor): The input image tensor.
    epsilon (float): The perturbation magnitude.
    data_grad (Tensor): The gradient of the loss w.r.t the input image.

Returns:
    Tensor: The perturbed image, clipped to the [0, 1] range.
"""
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

"""
Restores a batch of normalized tensors to their original scale using the provided mean and standard deviation.

This is the inverse operation of normalization commonly applied to image tensors.
"""
def denorm(batch, mean, std, device):
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

"""
Function used to define the type of an attribute inside of the parser for the main, to parse for example '--cnn true'
"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False

    return None

def get_fake_loaders(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    fake_dataset = FakeData(size=1000, image_size=(3, 32, 32), transform=transform)
    fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return fake_dataloader

"""
    Loads and prepares data loaders for the specified dataset.
    Applies appropriate normalization and transforms to dataset. Splits the training set into a training and 
    validation subset (validation set has 5000 samples).
    Returns:
        - train_dataloader: DataLoader for training data.
        - val_dataloader: DataLoader for validation data (subset of training set).
        - test_dataloader: DataLoader for test data.
        - num_classes (int): Number of output classes for the dataset.
        - input_size (int): Flattened size of a single input sample.
"""
def get_dataloaders(batch_size, num_workers):

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

    # Split train into train and validation.
    val_size = 5000
    indices = np.random.permutation(len(ds_train))
    ds_val = Subset(ds_train, indices[:val_size])
    ds_train = Subset(ds_train, indices[val_size:])

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=False, num_workers=num_workers,pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, num_classes, input_size

"""
Evaluates a CNN model on a given dataloader and returns Top-1 accuracy, Top-5 accuracy, and average loss.

This function computes:
- Top-1 accuracy using the predicted class with highest probability
- Top-5 accuracy by checking if the ground truth is among the top 5 predicted classes
- Mean cross-entropy loss across all batches

Note: This is the same function used in `utils.py` from LAB_1, reused here for clarity.
"""
def test_CNN(model, dataloader, device):
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
Returns predictions and ground truth labels for all samples in the dataloader using a CNN model (CNN or CNN2, in model.py).

The model's predictions are obtained using argmax over the output logits.
"""
def get_pred_CNN(model, dataloader, device):
    model.eval()
    y_gt, y_pred = [], []
    for it, data in enumerate(dataloader):
        x, y = data
        x, y = x.to(device), y.to(device)

        yp = model(x)

        y_pred.append(yp.argmax(1))
        y_gt.append(y)

    return y_gt, y_pred

"""
Trains the CNN and CNN2 model defined in `pretrain.py`.

- If `args.aug_fgsm` is False, the model is trained using standard CNN and CNN2 training.
- If `args.aug_fgsm` is True, the model is trained using FGSM-based adversarial examples as data augmentation.
  - If `args.rand_epsilon` is also True, a random epsilon value between 0.01 and 0.15 is used for FGSM generation.

The FGSM-based training procedure follows Algorithm 1 from the paper:
https://arxiv.org/abs/1806.02782
"""

def train_CNN(model, dataloader, opt, device, epoch, args):
    model.train()
    model = model.to(device)

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    losses_clean = []
    losses_adv = []

    train_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs} [Training]", leave=False)
    for data, labels in train_bar:
        data, labels = data.to(device), labels.to(device)

        if args.aug_fgsm:
            data.requires_grad = True

        logits_clean = model(data)
        loss_clean = F.cross_entropy(logits_clean, labels)

        if args.aug_fgsm:

            # Backward to clean forward
            model.zero_grad()
            loss_clean.backward()
            opt.step()

            data_grad = data.grad.data
            data_denorm = denorm(data, mean, std, device) # Denormalize for fgsm

            if args.rand_epsilon:
                epsilon = random.uniform(0.01, 1.5)
                data_adv = fgsm_attack(data_denorm, epsilon, data_grad) # Augmentation with fgsm with random epsilon
            else:
                data_adv = fgsm_attack(data_denorm, args.epsilon, data_grad) # Augmentation with fgsm with static epsilon

            data_adv = transforms.Normalize(mean, std)(data_adv) # Re-normalize for model input

            # Forward pass on adversarial examples
            logits_adv = model(data_adv)
            loss_adv = F.cross_entropy(logits_adv, labels)

            # Backward to adv forward
            opt.zero_grad()
            loss_adv.backward()
            opt.step()

            losses_clean.append(loss_clean.item())
            losses_adv.append(loss_adv.item())

            train_bar.set_postfix(minibatch_loss_clean=f"{loss_clean.item():.4f}", minibatch_loss_adv=f"{loss_adv.item():.4f}")

        else:

            opt.zero_grad()
            loss_clean.backward()
            opt.step()
            losses_clean.append(loss_clean.item())
            train_bar.set_postfix(minibatch_loss_clean=f"{loss_clean.item():.4f}")


    return np.mean(losses_clean), np.mean(losses_adv) if len(losses_adv) > 0 else 0

"""
Computes the maximum logit value for each element in the input batch.

Returns the maximum value across classes (dimension 1) for each input sample.
"""
def max_logit(logit):
    s = logit.max(dim=1)[0] #get the max for each element of the batch
    return s

"""
Computes the maximum softmax score from logits using a specified temperature `t`.

Applies the softmax function to the input logits scaled by temperature `t`, then returns the maximum softmax value for each element in the batch.
"""
def max_softmax(logit, t = 1.0):
    s = F.softmax(logit/t, 1)
    s = s.max(dim=1)[0] #get the max for each element of the batch
    return s

"""
Computes and returns the scores of a CNN model using a provided dataloader and scoring function.

The `score_fun` parameter can be a scoring function such as `max_softmax` or `max_logit`.
The function iterates over the dataloader, applies the model and scoring function to each batch, and returns the concatenated scores.
"""
def compute_scores(model, dataloader, score_fun, device):
    scores = []
    with torch.no_grad():
        tqdm_bar = tqdm(dataloader, desc="[Testing (Val/Test/Fake)]", leave=False)
        for data in tqdm_bar:
            x, y = data
            output = model(x.to(device))
            s = score_fun(output)
            scores.append(s)
        scores_t = torch.cat(scores)
        return scores_t

def test_AE(model, dataloader, device):
    model.eval()
    # use negative MSE since higher error means OOD
    loss = nn.MSELoss(reduction='none')
    scores = []
    losses = []
    tqdm_bar = tqdm(dataloader, desc="[Testing (Val/Test/Fake)]", leave=False)
    with torch.no_grad():
        for data in tqdm_bar:
            x, y = data
            x = x.to(device)
            z, xr = model(x)
            l = loss(x, xr)
            score = l.mean([1, 2, 3])

            losses.append(score)
            scores.append(-score)

    scores = torch.cat(scores)
    losses = torch.mean(torch.cat(losses))
    return  scores, losses.item()

"""
Trains the autoencoder model defined in `pretrain.py`.

- If `args.aug_fgsm` is False, the model is trained using standard autoencoder training.
- If `args.aug_fgsm` is True, the model is trained using FGSM-based adversarial examples as data augmentation.
  - If `args.rand_epsilon` is also True, a random epsilon value between 0.01 and 0.15 is used for FGSM generation.

The FGSM-based training procedure follows Algorithm 1 from the paper:
https://arxiv.org/abs/1806.02782
"""
def train_AE(model, dataloader, opt, device, epoch, args):
    model.train()
    model = model.to(device)

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    losses_clean = []
    losses_adv = []

    train_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs} [Training]", leave=False)
    for data, _ in train_bar:
        data = data.to(device)

        if args.aug_fgsm:
            data.requires_grad = True

        opt.zero_grad()

        z, x_rec = model(data)
        loss_clean = F.mse_loss(data, x_rec)

        if args.aug_fgsm:

            # Backward to get gradients wrt input
            model.zero_grad()
            loss_clean.backward()
            opt.step()

            data_grad = data.grad.data
            data_denorm = denorm(data, mean, std, device) # Denormalize for fgsm

            if args.rand_epsilon:
                epsilon = random.uniform(0.01, 0.2)
                data_adv = fgsm_attack(data_denorm, epsilon, data_grad) # Augmentation with fgsm with random epsilon
            else:
                data_adv = fgsm_attack(data_denorm, args.epsilon, data_grad) # Augmentation with fgsm with static epsilon

            data_adv = transforms.Normalize(mean, std)(data_adv) # Re-normalize for model input

            # Forward pass on adversarial examples
            z_adv, x_rec_adv = model(data_adv)
            loss_adv =  F.mse_loss(data, x_rec_adv)

            # Backward to adv forward
            opt.zero_grad()
            loss_adv.backward()
            opt.step()

            losses_clean.append(loss_clean.item())
            losses_adv.append(loss_adv.item())

            train_bar.set_postfix(minibatch_loss_clean=f"{loss_clean.item():.4f}", minibatch_loss_adv=f"{loss_adv.item():.4f}")

        else:

            opt.zero_grad()
            loss_clean.backward()
            opt.step()
            losses_clean.append(loss_clean.item())
            train_bar.set_postfix(minibatch_loss_clean=f"{loss_clean.item():.4f}")


    return np.mean(losses_clean), np.mean(losses_adv) if len(losses_adv) > 0 else 0

"""
Function to define the wandb parameters for the pretrain.py 
"""
def config_loggers(args):

    if args.use_wandb:
        wandb.init(
            project="DLA_LAB_4",
            config=vars(args),
            name=args.exp_name,
        )