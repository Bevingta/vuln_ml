import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, logging
import time
from tqdm import tqdm

logging.set_verbosity_error()

df = pd.read_json("../data/train.json")

# Keep only the columns needed and drop any NaNs
df = df[['func', 'target']].dropna(subset=['func', 'target'])

# Ensure target is integer
df['target'] = df['target'].astype(int)

# an explicit EOS token:
# df['func'] = df['func'].astype(str) + tokenizer.eos_token

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)


# 2. Tokenizer & model
model_name = "Salesforce/codet5-base"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()


# 3. Training setup
batch_size  = 4
epochs      = 700
#700
max_steps   = 125
#125
optimizer   = torch.optim.AdamW(model.parameters(), lr=3e-6)
total_steps = epochs * max_steps
warmup      = max_steps // 10
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup,
    num_training_steps=total_steps,
)


# 4. Training loop using DataFrame batches
with open("final_binary.txt", "w") as log:
    log.write("Starting training...\n")
    log.write(f"Batch size: {batch_size}\n")

    start_idx = 0  

    for epoch in tqdm(range(epochs), desc="Epochs"):
        step       = 0
        epoch_loss = 0.0
        start      = time.time()
        log.write(f"\nEpoch {epoch+1}/{epochs}\n")
        
        # iterate over df in batch-size chunks
        with tqdm(total=min(max_steps, (len(df) - start_idx + batch_size - 1) // batch_size),
                  desc=f"Epoch {epoch+1} Batches", leave=False) as pbar:
            while step < max_steps and start_idx < len(df):
                batch_start = start_idx
                end_idx = min(start_idx + batch_size, len(df))
                batch   = df.iloc[start_idx:end_idx]
                start_idx = end_idx 

                code_batch  = batch['func'].tolist()
                label_batch = batch['target'].tolist()

                # tokenize
                enc = tokenizer(
                    code_batch,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                input_ids      = enc.input_ids.to(device)
                attention_mask = enc.attention_mask.to(device)
                labels         = torch.tensor(label_batch, dtype=torch.long).to(device)

                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                except ValueError as e:
                    msg = str(e)
                    if "same number of <eos> tokens" in msg:
                        log.write(f"Skipping batch {batch_start}-{end_idx} due to eos mismatch: {msg}\n")
                        continue
                    else:
                        raise

                loss   = outputs.loss
                logits = outputs.logits

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                preds = logits.argmax(dim=-1)
                acc   = (preds == labels).float().mean().item()
                step += 1
                pbar.update(1)

                log.write(f"\nStep {step} — loss {loss.item():.4f}, acc {acc:.4f}\n")

                pairs = [
                    (int(p.item()), int(t.item()))
                    for p, t in zip(preds, labels)
                ]
                log.write(f"(Prediction, output): {pairs}\n")

                if step >= max_steps:
                    break

        duration = time.time() - start
        avg_loss = epoch_loss / step if step > 0 else 0.0
        log.write(
            f"\nEpoch done in {duration:.1f}s, average loss: {avg_loss:.4f}\n\n"
        )

    log.write("Training complete.\n")


# Save the fine-tuned model 

model.save_pretrained("./codet5_binary_model")
tokenizer.save_pretrained("./codet5_binary_model")