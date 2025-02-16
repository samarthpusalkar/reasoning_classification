from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
# from transformers import pipeline
from datasets import load_dataset
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import autograd
autograd.set_detect_anomaly(mode=True)
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()
import wandb
import os
from variables import TOKENIZED_DATASET_PATH, TRAIN_DATASET_PATH
from transformers import PreTrainedTokenizerFast
# Set CUDA device and optimization flags
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device
torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere

# Check CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

fn1 = nn.Softmax(dim=-1)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1)[:, np.newaxis])
    return e_x / e_x.sum(axis=-1)[:,np.newaxis]

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        model_output = model(**inputs)
        logits = model_output.logits
        labels = inputs.get("labels")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, model_output) if return_outputs else loss

def compute_metrics(p):
    """
    Compute the accuracy, precision, recall, and F1 score from logits.

    Args:
    logits (np.array): Logits output by the model.
    labels (list or np.array): True labels.

    Returns:
    dict: A dictionary containing the computed metrics.
    """
    # Convert logits to predicted labels
    logits = p.predictions
    labels = p.label_ids
    # p.predictions, references=p.label_ids 
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def preprocess_function(examples, tokenizer):
    # Tokenize with longer sequence length if GPU memory allows
    tokenized_inputs = tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        # truncation_strategy = "max_length",
        max_length=512,  # Adjust based on your GPU memory
        return_tensors="pt"
    )
    tokenized_inputs['labels'] = examples['label']
    return tokenized_inputs

# Load and prepare data
MODEL_PATH = "answerdotai/ModernBERT-base"#"answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
id2label = {0: "non-reasoning", 1: "reasoning"}
label2id = {v:k for k,v in id2label.items()}

# Clean up memory before loading model
import gc
gc.collect()
torch.cuda.empty_cache()

# Load model and move to GPU
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(id2label),
    id2label=id2label, label2id=label2id
).to(device)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Process dataset with larger batch size
try:
    if os.path.exists(TOKENIZED_DATASET_PATH):
        encoded_dataset = load_dataset(TOKENIZED_DATASET_PATH)
    else:
        raise "Tokenized Dataset not found"
except Exception as e:
    dataset = load_dataset(TRAIN_DATASET_PATH)
    encoded_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=32,  # Adjust based on your CPU memory
        fn_kwargs={"tokenizer": tokenizer}
    )
    encoded_dataset.save_to_disk(TOKENIZED_DATASET_PATH)
tran_eval = encoded_dataset['test'].train_test_split(test_size=0.1, shuffle=True, stratify_by_column='label')

run = wandb.init(
    project="ReasoningClassificationFull",
    name="experiment-2025-2-16_",
)
config = {
"model_architecture": 'ModernBert-Base',
"learning_rate": 3e-5,
"per_device_train_batch_size": 704,
"per_device_eval_batch_size": 512,
"num_train_epochs": 2,
"gradient_accumulation_steps":4,
"dataloader_num_workers":4,
"weight_decay": 0.001,
"warmup_ratio": 0.03,
"logging_steps": 50,
"evaluation_strategy": "steps",
"eval_steps": 100,
"save_strategy": "steps",
"save_steps": 200,
"load_best_model_at_end": True,
"metric_for_best_model": "eval_loss",
"gradient_checkpointing": True,
"fp16": True
}
with wandb.init(config=config) as run:
    run.config.update(config)
# Optimize training arguments for CUDA
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=config['per_device_train_batch_size'],  # Increase if GPU memory allows
    per_device_eval_batch_size=config['per_device_eval_batch_size'],   # Can be larger than training batch size
    num_train_epochs=config['num_train_epochs'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    # max_grad_norm = 1.0,  # Add
    gradient_checkpointing=config['"gradient_checkpointing"'],
    fp16=config['fp16'],                       # Enable mixed precision training
    # fp16_opt_level="O2",            # Aggressive mixed precision
    dataloader_num_workers=config['dataloader_num_workers'],        # Parallel data loading
    dataloader_pin_memory=True,      # Pin memory for faster data transfer to GPU
    learning_rate=config['learning_rate'],
    weight_decay=config['weight_decay'],
    warmup_ratio=config['warmup_ratio'],
    logging_steps=config['logging_steps'],
    evaluation_strategy=config['evaluation_strategy'],
    eval_steps=config['eval_steps'],
    save_strategy=config['save_strategy'],
    save_steps=config['save_steps'],
    load_best_model_at_end=config['load_best_model_at_end'],
    metric_for_best_model=config["metric_for_best_model"],
    report_to = "wandb",
    run_name="experiment-2025-2-16_"
)

# Initialize trainer with optimized settings
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=tran_eval["test"],
    compute_metrics = compute_metrics
)

# Train with CUDA optimization
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(eval_results)
run.finish()
# Save the model
trainer.save_model("./modern-bert-finetuned")
tokenizer.save_pretrained("./modern-bert-finetuned")

# Clean up memory after training
del model
del trainer
gc.collect()
torch.cuda.empty_cache()
