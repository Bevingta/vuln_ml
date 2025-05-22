import torch
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score, 
    f1_score,
    classification_report,
)
from tqdm import tqdm


# 1. Setup device, load model + tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = "./codet5_binary_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
model.to(device)
model.eval()

# 2. Load your validation DataFrame
with open("../data/test.json", "r") as f:
    data = json.load(f)
val_df = pd.DataFrame(data) 
val_df = val_df[['func', 'target']].dropna()
texts = val_df['func'].tolist()
labels = val_df['target'].astype(int).tolist()

# 3. Tokenize (all at once or in chunks)
encodings = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
input_ids = encodings.input_ids
attention_mask = encodings.attention_mask
labels_tensor = torch.tensor(labels)

# 4. Create a DataLoader for batching
dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
loader  = DataLoader(dataset, batch_size=8)

# 5. Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating"):
        ids, mask, labs = [t.to(device) for t in batch]
        outputs = model(input_ids=ids, attention_mask=mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labs.cpu().tolist())

n_preds = len(all_preds)
n_truth = len(val_df)

# 6. Metrics
acc     = accuracy_score(all_labels, all_preds)
prec    = precision_score(all_labels, all_preds)
rec     = recall_score(all_labels, all_preds)
f1      = f1_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=4)

with open("binary_test_log", "w") as log:
    log.write(f"Processed {n_preds} examples, expected {n_truth}.\n\n")
    log.write(f"Accuracy : {acc:.4f}\n")
    log.write(f"Precision: {prec:.4f}\n")
    log.write(f"Recall   : {rec:.4f}\n")
    log.write(f"F1 Score : {f1:.4f}\n\n")
    log.write("Per-class report:\n")
    log.write(report)