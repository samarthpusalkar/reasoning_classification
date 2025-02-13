from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from datasets import load_dataset
import torch
from torch import nn
import os
import evaluate
import numpy as np
torch.autograd.set_detect_anomaly(True)

fn1 = nn.Softmax(dim=-1)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1)[:, np.newaxis])
    return e_x / e_x.sum(axis=-1)[:,np.newaxis]

accuracy = evaluate.load("accuracy")
id2label = {0: "not_category", 1: "reasoning_category", 2: "self_questioning_category"}
label2id = {v:k for k,v in id2label.items()}

# Set CUDA device and optimization flags
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device
torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere

# Check CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # Your custom loss calculation logic here

        # Example: using a weighted cross-entropy loss
        model_output = model(**inputs)
        logits = model_output.logits
        # fn1 = nn.Softmax(dim=-1)
        preds = fn1(logits)
        # preds = preds[:, 1]
        labels = inputs.get("labels").float()
        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss = loss_fn(model_output.logits, labels)
        loss_fct = nn.MSELoss()
        loss = loss_fct(preds, labels)

        return (loss, model_output) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = softmax(predictions)
    predictions = np.round(preds)
    cat1_pred = predictions[:,0]
    cat2_pred = predictions[:,1]
    cat3_pred = predictions[:,2]
    return {"not_category": accuracy.compute(predictions=cat1_pred, references=labels[:,0]), "reasoning_category":accuracy.compute(predictions=cat2_pred, references=labels[:,1]), "self_questioning_category":accuracy.compute(predictions=cat3_pred, references=labels[:,2])}


def preprocess_function(examples):
    # Tokenize with longer sequence length if GPU memory allows
    tokenized_inputs = tokenizer(
        examples['text_segment'],
        padding="max_length",
        truncation=True,
        max_length=512,  # Adjust based on your GPU memory
        return_tensors="pt"
    )
    
    labels = []
    for i in range(len(examples['text_segment'])):
        sum_ = examples['not_category'][i]+ examples['reasoning_category'][i]+ examples['self_questioning_category'][i]
        temp_label = [examples['not_category'][i]/sum_, examples['reasoning_category'][i]/sum_, examples['self_questioning_category'][i]/sum_]
        labels.append(temp_label)
        # if examples['not_category'][i] == 1:
        #     labels.append(0)
        # elif examples['reasoning_category'][i] == 1:
        #     labels.append(0)
        # elif examples['self_questioning_category'][i] == 0:
        #     labels.append(1)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Load and prepare data
MODEL_PATH = "./modern-bert-finetuned"#"answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
dataset = load_dataset('csv', data_files={'train': 'dataset-full1.csv', 'test': 'test.csv', 'eval': 'eval.csv'})

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
    batch_size=32  # Adjust based on your CPU memory
)

# Optimize training arguments for CUDA
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,  # Increase if GPU memory allows
    per_device_eval_batch_size=4,   # Can be larger than training batch size
    num_train_epochs=10,
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
# trainer.train()

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
