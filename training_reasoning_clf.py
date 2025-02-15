from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
# from transformers import pipeline
from datasets import load_dataset
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
torch.autograd.set_detect_anomaly(True)
from datasets import load_dataset

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
        labels = inputs.get("labels").float()
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, model_output) if return_outputs else loss

def compute_metrics(logits, labels):
    """
    Compute the accuracy, precision, recall, and F1 score from logits.

    Args:
    logits (np.array): Logits output by the model.
    labels (list or np.array): True labels.

    Returns:
    dict: A dictionary containing the computed metrics.
    """
    # Convert logits to predicted labels
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


def preprocess_function(examples, tokenizer=None):
    # Tokenize with longer sequence length if GPU memory allows
    tokenized_inputs = tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=512,  # Adjust based on your GPU memory
        return_tensors="pt"
    )
    tokenized_inputs['labels'] = examples['label']
    return tokenized_inputs

# Load and prepare data
MODEL_PATH = "./modern-bert-finetuned"#"answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
dataset = load_dataset('CodeIsAbstract/reasoning_dataset')
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
encoded_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=32,  # Adjust based on your CPU memory
    fn_kwargs={"tokenizer": tokenizer}
)

# Optimize training arguments for CUDA
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,  # Increase if GPU memory allows
    per_device_eval_batch_size=4,   # Can be larger than training batch size
    num_train_epochs=3,
    gradient_accumulation_steps=1,
    # max_grad_norm = 1.0,  # Add
    # gradient_checkpointing=True,
    # fp16=True,                       # Enable mixed precision training
    # fp16_opt_level="O2",            # Aggressive mixed precision
    dataloader_num_workers=4,        # Parallel data loading
    dataloader_pin_memory=True,      # Pin memory for faster data transfer to GPU
    learning_rate=2e-6,
    weight_decay=0.001,
    warmup_ratio=0.1,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Initialize trainer with optimized settings
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["eval"],
    compute_metrics = compute_metrics
)

# Train with CUDA optimization
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(eval_results)

# Save the model
trainer.save_model("./modern-bert-finetuned")
tokenizer.save_pretrained("./modern-bert-finetuned")

# Clean up memory after training
del model
del trainer
gc.collect()
torch.cuda.empty_cache()
