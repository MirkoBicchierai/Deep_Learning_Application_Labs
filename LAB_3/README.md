# Deep Learning Transformer Lab

This repository contains experiments on fine-tuning DistilBERT using LoRA (Low-Rank Adaptation) for sentiment analysis on the Rotten Tomatoes dataset.

## Experiment Tracking

All experiments can be viewed on Weights & Biases at the following link:
https://wandb.ai/AI-UNIFI/DLA_LAB_3?nw=nwusermirkobicchierai

## Running Experiments

To run the experiments, use the following command:

```bash
python main.py [parameters]
```

### Key Parameters

#### Exercise 1.3 - Baseline
- `--check_baseLine`: If set to True, compute the baseline using a linear SVM (Exercise 1.3)

#### Dataset
- `--dataset`: "stanfordnlp/sst2" or "rotten_tomatoes"

#### LoRA Parameters (Exercise 3.1)
- `--lora`: If True, use LoRA to fine-tune the model
- `--lora_alpha`: LoRA alpha parameter (default: 32)
- `--lora_rank`: LoRA rank parameter (default: 8)

#### Basic Optimizer Parameters
- `--lr`: Learning rate for fine-tuning DistilBERT (default: 2e-5)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Training batch size (default: 24)

## Experiment Setup

All experiments are configured in the bash file:
- `experiments_1.sh` (Experiments on Rotten Tomatoes dataset)
- `experiments_2.sh` (Experiments on stanfordnlp/sst2 dataset)

## Exercise 3.1 - LoRA Fine-tuning

For Exercise 3, I chose Exercise 3.1, which involved finding an intelligent way to fine-tune DistilBERT on the Rotten Tomatoes dataset using LoRA through the PEFT library.

Unlike fine-tuning the entire DistilBERT model as in Exercise 2, the LoRA approach results in:
- Faster inference (as shown in the graphs on Weights & Biases)
- More stable and easier training compared to fine-tuning the entire model
- Raw accuracy that is much more stable with LoRA
- Lower memory usage, allowing for larger batch sizes and thus faster training (though for fair comparison, the same batch size was maintained)

## Results Organization

On Weights & Biases, the experiments are divided into 3 groups:
- "lora": for models where LoRA was used (on rotten_tomatoes dataset)
- "Full": for models where LoRA was not used (on rotten_tomatoes dataset)
- "stanfordnlp/sst2": for models where LoRA was used  (on stanfordnlp/sst2 dataset) 

For LoRA implementation, I found similar work on Hugging Face about fine-tuning DistilBERT, which indicated that targeting the modules "q_lin", "k_lin", "v_lin", "out_lin", "lin1", and "lin2" with LoRA produces the best results.

# Rotten Tomatoes dataset

### Baseline Performance

As a baseline, I used a linear SVM with a maximum number of iterations set to 1000, which achieved:
- 80% accuracy on the validation set
- 79% accuracy on the test set

### Fine-tuned Model Performance

The models specifically fine-tuned with LoRA achieved slightly better results, approximately 83% accuracy on the test set and 86% on the validation set.

![loss rott.png](img/loss%20rott.png)
![acc rott.png](img/acc%20rott.png)


# stanfordnlp/sst2

This dataset is significantly larger than Rotten Tomatoes, containing 67,349 training samples, 872 validation samples, and 1,821 test samples.
For this reason, I used it only to fine-tune DistilBERT with LoRA and to observe how performance changes as the dataset complexity increases (also considering the much longer training times, 5 epochs instead of 10 (10 hours)).

One additional note: the test set does not include labels, so I was unable to use it for evaluation.

### Baseline Performance

As a baseline, I used a linear SVM with a maximum number of iterations set to 1000, which achieved:
- 65% accuracy on the validation set
- Test set of this dataset don't have the label

### Fine-tuned Model Performance

I fine-tuned this model on the dataset using only LoRA. Fully fine-tuning the entire model was not feasible due to resource constraints.
Overall, the model achieves around 90% accuracy on the validation set (the test set has no labels), significantly outperforming the SVM baseline, which achieved approximately 65% accuracy.

![W&B Chart 28_04_2025, 11_14_40.png](img/W%26B%20Chart%2028_04_2025%2C%2011_14_40.png)
![W&B Chart 28_04_2025, 11_13_25.png](img/W%26B%20Chart%2028_04_2025%2C%2011_13_25.png)