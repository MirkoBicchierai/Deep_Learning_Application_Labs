import argparse
import numpy as np
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    TrainingArguments, Trainer
import torch
from sklearn.preprocessing import StandardScaler
from utils import str2bool, config_loggers


def get_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter settings")

    #Dataset choice
    parser.add_argument("--dataset", type=str, default="rotten_tomatoes", help="Possible choose stanfordnlp/sst2 or rotten_tomatoes")

    # Exercise 1.3
    parser.add_argument("--check_baseLine", type=str2bool, default=False, help="If True compute the baseline using a linear SVM (Exercise 1.3)")

    # Lora
    parser.add_argument("--lora", type=str2bool, default=False, help="If True use LorA to fine tune all model (exercise 3.1)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha")
    parser.add_argument("--lora_rank", type=int, default=8, help="Lora rank")

    # basic optimizer parameters
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate to fine tune distillibert")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size to train for")

    # Use wandb
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="If True use wandb")

    args = parser.parse_args()

    return args


# Function that perform a test of the pretrained model output on the dataset stanfordnlp/sst2 or rotten_tomatoes
def test_output_model(dataset_name, tokenizer, model):
    dataset = load_dataset(dataset_name, split="train")

    if dataset_name == "stanfordnlp/sst2":
        sample_texts = dataset.select(range(3))["sentence"]
    else:
        sample_texts = [dataset[i]["text"] for i in range(3)]

    print("Example of 3 text in the train dataset",sample_texts)

    # Tokenize the sample texts
    inputs = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    print("Keys in model output:", outputs.keys())
    print("Last hidden state shape:", outputs.last_hidden_state.shape)


# Function to compute the metric during the training, compute accuracy, precision, recall and f1_score using sklearn library method
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Function to extract [CLS] embeddings
def extract_cls_embeddings(text_list, tokenizer, model):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token = first token
    return cls_embeddings.numpy()


# Extract features for one split "train", "test" and "val"
def get_split_embeddings(dataset_split, dataset, tokenizer, model):
    features = []
    labels = []
    for example in tqdm(dataset[dataset_split]):
        text = example.get("text", example.get("sentence")) # In stanfordnlp/sst2 dataset the text is inside the key 'sentence' instead of 'text' for the other dataset
        label = example["label"]
        emb = extract_cls_embeddings([text], tokenizer, model)[0]
        features.append(emb)
        labels.append(label)
    return np.array(features), np.array(labels)


def main():
    args = get_parser()

    # EXERCISE 1.1
    dataset = load_dataset(args.dataset) # stanfordnlp/sst2 , rotten_tomatoes
    print("\nDataset structure:")
    print(dataset)

    # Print dataset statistics
    print("\nDataset statistics:")
    for split in dataset:
        print(f"Split: {split}, Size: {len(dataset[split])}")
        labels = [example["label"] for example in dataset[split]]
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        print(f"  Positive: {pos_count} ({pos_count/len(labels):.2%})")
        print(f"  Negative: {neg_count} ({neg_count/len(labels):.2%})")


    # EXERCISE 1.2
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    test_output_model(args.dataset, tokenizer, model)

    # EXERCISE 1.3, compute the baseline of accuracy on the dataset selected in the parser with a SCV linear
    if args.check_baseLine:

        x_train, y_train = get_split_embeddings("train", dataset, tokenizer, model)
        x_val, y_val = get_split_embeddings("validation", dataset, tokenizer, model)

        scaler = StandardScaler()  # Define a scaler obj to scale the x
        # The other dataset stanfordnlp/sst2 don't have the label on the test set, do only for rotten_tomatoes dataset
        if args.dataset == "rotten_tomatoes":
            x_test, y_test = get_split_embeddings("test", dataset, tokenizer, model)
            x_test_scaled = scaler.transform(x_test)

        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        clf = SVC(kernel='linear', max_iter=1000) # I use max_iter=1000, i try higher value but was computationally unsustainable
        clf.fit(x_train_scaled, y_train)

        val_pred = clf.predict(x_val_scaled)
        print("Validation Accuracy (Base Line):", accuracy_score(y_val, val_pred))  # 0.8095 rotten_tomatoes, 0.6523 stanfordnlp/sst2

        # The other dataset stanfordnlp/sst2 don't have the label on the test set
        if args.dataset == "rotten_tomatoes":
            test_pred = clf.predict(x_test_scaled)
            print("Test Accuracy (Base Line):", accuracy_score(y_test, test_pred))  # 0.7917 rotten_tomatoes

    # EXERCISE 2 (.1 .2 .3) and 3.1

    # Tokenization function
    def tokenize_function(example):
        text = example.get("text", example.get("sentence")) # In stanfordnlp/sst2 dataset the text is inside the key 'sentence' instead of 'text' for the other dataset
        return tokenizer(
            text,
            padding="max_length",  # For batching later
            truncation=True,
            max_length=512  # DistilBERT max length
        )

    # Apply tokenizer to all splits
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Format check
    print(tokenized_datasets["train"].features)
    print(tokenized_datasets["train"][0])

    config_loggers(args)  # Config wandb experiments, see utils.py

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # load the model distilbert-base-uncased with the classification task has 2 classes (binary classification).
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "negative", 1: "positive"},  # mappings labels
        label2id={"negative": 0, "positive": 1}  # mappings labels
    )

    # Define the trainer arguments
    training_args = TrainingArguments(
        output_dir="./results",
        run_name="distilbert-classification-experiment",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        report_to = "wandb" if args.use_wandb else None,
    )

    for name, param in model.named_parameters():
            print(f"parameter: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")

    # IF args.lora is true use lora to fine tune the model else fine tune the all model
    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_rank,  # rank of the low-rank decomposition
            lora_alpha=args.lora_alpha,  # scaling factor
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"], # which layers to inject LoRA into (works well for DistilBERT, i found online this target modules in another work published on hugging face)
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        # instantiate the lora model using peft library
        model = get_peft_model(model, lora_config)

        # Print detailed information about LoRA layers (is a simple check)
        print("=== LoRA LAYERS DETAILS ===")
        for name, param in model.named_parameters():
            if 'lora' in name:
                print(f"LoRA parameter: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")

        # Count total parameters vs trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")
        print("===========================")

    # Trainer configuration
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Fine tune the model
    trainer.train()

    # The other dataset stanfordnlp/sst2 don't have the label on the test set, do it only for rotten_tomatoes dataset
    if args.dataset == "rotten_tomatoes":
        # Evaluate on test set
        test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        print(test_results)
        if args.use_wandb:
            wandb.log({
                "test/accuracy": test_results["eval_accuracy"],
                "test/precision": test_results["eval_precision"],
                "test/recall": test_results["eval_recall"],
                "test/f1": test_results["eval_f1"]
            })

if __name__ == "__main__":
    main()
