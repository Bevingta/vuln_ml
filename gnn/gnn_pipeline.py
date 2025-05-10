import os
import sys
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data.GraphDataset import GraphDataset
from model import GNNModel
from tqdm import tqdm
import json
from data.data_visualization import plot_confusion_matrix, plot_training_history, plot_roc_curve
from sklearn.metrics import classification_report, roc_curve
import argparse
from data.w2v.train_word2vec import train_w2v
from gensim.models import Word2Vec
import numpy as np
from data.data_processing import subsample_and_split, print_split_stats, load_huggingface_datasets, preprocess_graphs, load_seengraphs, save_seengraphs, remove_duplicates
from utils.util_funcs import load_configs, load_w2v_from_huggingface, early_stopping
from utils.FocalLoss import FocalCrossEntropyLoss
from utils.json_functions import load_json_array
import shutil
from datetime import datetime
from utils.logging import create_logger
from data.graph_gen.graph_generator import generate_one_graph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
W2V_PATH = os.path.join(BASE_DIR, "data/w2v/word2vec_code.model")

# Training loop
def train(model, train_loader, val_loader, optimizer, model_save_path, criterion, device, epochs, patience, logger, run_history_save_path, scheduler=None, roc_implementation = True, losses_file_path="training_losses.json"):
    losses = {'epoch_loss': [], 'batch_loss': []}
    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    }
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        batch_losses = []  # List to store batch losses for the current epoch

        # Create a tqdm progress bar for the training batches
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as batch_progress:
            for data in batch_progress:
                data = data.to(device) # So i can use either CPU or GPU depending on machine
                optimizer.zero_grad()  # Zero gradients
                out = model(data)  # Forward pass
                
                # Calculate loss
                labels = data.y
                loss = criterion(out, labels)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                
                running_loss += loss.item()
                batch_losses.append(loss.item())

                # Get predictions
                _, predicted = torch.max(out, 1)
                correct += (predicted == labels.view(-1)).sum().item()
                total += data.y.size(0)
                
                # Update the progress bar with loss and accuracy info
                batch_progress.set_postfix(loss=loss.item(), accuracy=correct/total)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # STORE LOSSES
        losses['epoch_loss'].append(epoch_loss)
        losses['batch_loss'].append(batch_losses)
        # Validation evaluation
        if not roc_implementation:
            val_accuracy, val_loss, val_preds, val_labels, _ = evaluate(model, val_loader, criterion, device)
            prec, rec, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average="binary", zero_division=0)
        else:
            # Validation evaluation with ROC analysis
            val_accuracy, val_loss, val_preds, val_labels, val_probs, _ = evaluate(model, val_loader, criterion, device, roc_implementation)

            # ROC thresholding

            fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
            plot_roc_curve(val_labels, val_probs, dataset_name=f"Val_Epoch{epoch+1}", save_path="visualizations/roc_val") #! Change save_path
            youden_index = tpr - fpr
            best_thresh = thresholds[np.argmax(youden_index)]

            # Apply adjusted threshold
            adjusted_val_preds = (np.array(val_probs) >= best_thresh).astype(int)

            # Metrics with thresholded predictions
            prec, rec, f1, _ = precision_recall_fscore_support(val_labels, adjusted_val_preds, average="binary", zero_division=0)

            # Optionally: print it
            print(f"📈 Epoch {epoch+1}: Val F1 (thresholded @ {best_thresh:.3f}) = {f1:.4f}")
            print(f"Validation accuracy: {val_accuracy}")

        # Log metrics
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(prec)
        history["val_recall"].append(rec)
        history["val_f1"].append(f1)

        logger.info(f"Validation stats after epoch {epoch}: accuracy: {val_accuracy}, recall: {rec}, precision: {prec}, F1: {f1}")

        stop_early, ewi = early_stopping(best_val_f1, f1, patience, epochs_without_improvement)
        epochs_without_improvement = ewi
        if stop_early:
            break

        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved at epoch {epoch+1} (Val F1: {f1:.4f})")

        if scheduler:
            # Pass validation loss to the scheduler
            scheduler.step(val_loss=val_loss)
        

    # Save final loss and metrics
    with open(losses_file_path, 'w') as f:
        json.dump(losses, f, indent=2)

    with open(f"{run_history_save_path}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("✅ Training history saved to training_history.json")


    with open(losses_file_path, 'w') as f:
        json.dump(losses, f, indent=2)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Evaluate the model

def evaluate(model, loader, criterion, device, roc_implementation=False,):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    predictions_dicts = []

    with tqdm(loader, desc="Testing model accuracy", unit="batch") as batch_progress:
        with torch.no_grad():
            for data in batch_progress:
                data = data.to(device)
                out = model(data)
                labels = data.y
                loss = criterion(out, labels)
                running_loss += loss.item()

                probs = torch.softmax(out, dim=1)[:, 1]  # Class 1 probabilities
                _, preds = torch.max(out, dim=1)  # shape: [batch_size]
                correct += (preds == labels).sum().item()
                total += data.y.size(0)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

                for i in range(len(labels)):
                    predictions_dicts.append({
                        "idx": int(data.idx[i].cpu().item()),
                        "prediction": int(preds[i].cpu().item()),
                        "target": int(labels[i].cpu().item())
                    })

    accuracy = correct / total
    avg_loss = running_loss / len(loader)

    if roc_implementation:
        return accuracy, avg_loss, all_preds, all_labels, all_probs, predictions_dicts
    return accuracy, avg_loss, all_preds, all_labels, predictions_dicts

def preprocess_pipeline(args, batch_size, downsample_factor):
    if (args.do_data_splitting == False) and args.download_presplit_datasets:
        load_huggingface_datasets(args.dataset_link)

    elif not (os.path.exists("data/split_datasets/train.json") and os.path.exists("data/split_datasets/test.json") and os.path.exists("data/split_datasets/valid.json")):    
        print("🚧 Splitting dataset into train/val/test...")
        with open(args.in_dataset, 'r') as f:
            full_data = json.load(f)
        subsample_and_split(full_data, "data/split_datasets", upsample_vulnerable=args.upsample_vulnerable, downsample_safe=args.downsample_safe, safe_ratio=args.vul_to_safe_ratio, downsample_factor=downsample_factor)

    # Load train/test/valid data arrays
    train_array = load_json_array(args.train_dataset)
    test_array = load_json_array(args.test_dataset)
    valid_array = load_json_array(args.valid_dataset)
    
    print_split_stats("Train", train_array)
    print_split_stats("Validation", valid_array)
    print_split_stats("Test", test_array)

    if args.generate_dataset_only:
        sys.exit()
    
    # Save graph for entry 9 in val_dataset
    test_graph = generate_one_graph(valid_array, 9, True)

    '''# === STANDARDIZE CODE === #
    for i in tqdm(range(len(train_array)), desc="Standardizing train functions", unit="function"):
        old_func = train_array[i]["func"]
        new_func = standardize_code(old_func)
        train_array[i]["func"] = new_func

    for i in tqdm(range(len(test_array)), desc="Standardizing test functions", unit="function"):
        old_func = test_array[i]["func"]
        new_func = standardize_code(old_func)
        test_array[i]["func"] = new_func

    for i in tqdm(range(len(valid_array)), desc="Standardizing valid functions", unit="function"):
        old_func = valid_array[i]["func"]
        new_func = standardize_code(old_func)
        valid_array[i]["func"] = new_func'''

    combined = remove_duplicates(train_array) + test_array + valid_array

    # LOAD or CREATE w2v
    if not os.path.exists(W2V_PATH):
        if args.download_w2v:
            print("Loading w2v from huggingface...")
            load_w2v_from_huggingface(args.w2v_link)
            w2v = Word2Vec.load(W2V_PATH)
        else:
            print("Training new w2v model")
            train_w2v(combined)
            w2v = Word2Vec.load(W2V_PATH)
    else:
        print("Word2Vec exists. Loading pretrained model...")
        w2v = Word2Vec.load(W2V_PATH)

    #* Preprocess graphs
    if os.path.exists("data/preprocessed_data/seen_graphs.pkl"):
        print("Loading seen graphs dict")
        seen_graphs = load_seengraphs()
    else:
        # Preprocess graphs for speed later
        seen_graphs = preprocess_graphs(train_array, test_array, valid_array)
        save_seengraphs(seen_graphs)
    
    train_dataset = GraphDataset(train_array, w2v, seen_graphs, args.save_memory)
    val_dataset = GraphDataset(valid_array, w2v, seen_graphs, args.save_memory)
    test_dataset = GraphDataset(test_array, w2v, seen_graphs, args.save_memory)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataset, val_dataset, test_dataset, train_loader, test_loader, val_loader

def main():
    #* STEP 1: LOAD CONFIGS
    input_dim, hidden_dim, output_dim, dropout, l2_reg, batch_size, learning_rate, epochs, downsample_factor, patience = load_configs()
    '''
        load_existing_model: boolean value, decides whether we load saved .pth model or train a new one
        save_graphs: boolean value, decide if we save graphs to computer (for faster runtime), or do not save (for better space efficiency)
        model_save_path: path to where we save our model at the end of training
        visualization_save_path: path to where we store graphs
        losses_file_path: path to where we store loss data
    '''

    #* ARGUMENT PARSING
    parser = argparse.ArgumentParser(description="Train and evaluate GNN model")
    parser.add_argument("--in-dataset", type=str, default="data/databases/complete_dataset.json", help="Path to the complete dataset (default: complete_dataset.json)")
    parser.add_argument("--train-dataset", type=str, default="data/split_datasets/train.json", help="Path to the training dataset split (default: train.json)")
    parser.add_argument("--test-dataset", type=str, default="data/split_datasets/test.json", help="Name of the testing dataset split (default: test.json)")
    parser.add_argument("--valid-dataset", type=str, default="data/split_datasets/valid.json", help="Name of the validation dataset split (default: test.json)")    
    parser.add_argument("--upsample-vulnerable", type=str, default=False, help="Upsample vulnerable entries (default: False)")
    parser.add_argument("--downsample-safe", type=str, default=False, help="Downsample safe entries (default: False)")
    parser.add_argument("--do-data-splitting", type=bool, default=False, help="Does data need to be split or is it already split? (default: False)")
    parser.add_argument("--download-presplit-datasets", type=bool, default=False, help="Option to download pre-split datasets from Huggingface (default: False)") 
    parser.add_argument("--dataset-link", type=str, default="alexv26/GNNVulDatasets", help="Link to download dataset (default: alexv26/GNNVulDatasets)")
    parser.add_argument("--download-w2v", type=bool, default=False, help="Option to download w2v from Huggingface (default: False)") 
    parser.add_argument("--w2v-link", type=str, default="alexv26/complete_dset_pretrained_w2v", help="Link to download dataset (default: alexv26/GNNVulDatasets)")
    parser.add_argument("--do-lr-scheduling", type=bool, default=True, help="Adjust learning rate after validation loss plateaus (default: True)")
    parser.add_argument("--vul-to-safe-ratio", type=int, default=None, help="Ratio between vulnerable to safe code: 1:n vul/safe (default: 3)")
    parser.add_argument("--generate-dataset-only", type=bool, default=False, help="Only generate dataset splits, do not run model (default: False)")
    parser.add_argument("--load-existing-model", type=bool, default=False, help="Load pre-trained model (default: False)")
    parser.add_argument("--roc-implementation", type=bool, default=True, help="Does the model use ROC curve based decision boundary adjustments? (default: True)")
    parser.add_argument("--architecture-type", type=str, default="rgcn", help="Architecture type (default: rgcn)")
    parser.add_argument("--save-memory", type=bool, default=False, help="Use less RAM by generating graphs every time instead of loading from seen_dict? (default: False)")
   
    args = parser.parse_args()

    if args.download_presplit_datasets and args.dataset_link is None:
        parser.error("--dataset-name is required when --download-presplit-datasets is set to True")

    '''
    The code below basically handles whether you need to do pre-splitting of data or not. If we do, we a pre-split of the data
    and try to keep it balanced between datasets of vuln/nonvuln.
    '''

    train_dataset, val_dataset, test_dataset, train_loader, test_loader, val_loader = preprocess_pipeline(args, batch_size, downsample_factor)

    input_dim = train_dataset[0].x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, model=args.architecture_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    if args.do_lr_scheduling:
        print("The model will adjust learning rate when validation loss plateau's.")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    else:
        scheduler = None

    
    # ==== SAVE RUN HISTORY ==== #

    os.makedirs("run_history", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_history_save_path = f"run_history/run_{timestamp}" 
    os.makedirs(run_history_save_path)
    # Save copy of configs to run_history_save_path
    shutil.copy2("configs.json", f"{run_history_save_path}/run_{timestamp}_configs.json")

    visualizations_save_path = f"{run_history_save_path}/visualizations"
    os.makedirs(visualizations_save_path, exist_ok=True)

    model_save_path = f"{run_history_save_path}/saved_model.pth"

    # ==== CREATE LOGGER ==== #
    logger = create_logger(run_history_save_path, timestamp)

    #* STEP 2: TRAIN MODEL
    if not args.load_existing_model:
        # SETTING WEIGHTS TO FIX VULN/NONVULN INBALANCE
        vuln, nonvuln = train_dataset.get_vuln_nonvuln_split()
        print(vuln, nonvuln)
        total = vuln + nonvuln

        # Weight inversely proportional to class frequency
        weight = torch.tensor([
            nonvuln / total,   # weight for class 0 (safe)
            vuln / total       # weight for class 1 (vulnerable)
        ], dtype=torch.float).to(device)
        
        criterion = FocalCrossEntropyLoss()

        train(model, train_loader, val_loader, optimizer, model_save_path=model_save_path, criterion=criterion, device=device, roc_implementation=args.roc_implementation, losses_file_path=f"{run_history_save_path}/training_losses.json", logger=logger, epochs=epochs, patience=patience, run_history_save_path=run_history_save_path) 
        print(train_dataset.get_skipped_embeddings_count())
    else:
        print("Loading existing model")
        model.load_state_dict(torch.load(model_save_path))

    #* STEP 3: TEST MODEL
    if not args.roc_implementation:
        test_accuracy, _, all_preds_test, all_labels_test, prediction_dicts = evaluate(model, test_loader, criterion, device)
        # Test results
        plot_confusion_matrix(all_labels_test, all_preds_test, dataset_name="Test", save_path=visualizations_save_path)
        print(classification_report(
            all_labels_test,
            all_preds_test,
            target_names=["Safe", "Vulnerable"],
            zero_division=0  # suppress warnings for undefined metrics
        ))
    else:
        test_accuracy, _, all_preds_test, all_labels_test, all_probs_test, prediction_dicts = evaluate(
        model, test_loader, criterion, device, args.roc_implementation
        )

        fpr, tpr, thresholds = roc_curve(all_labels_test, all_probs_test)
        plot_roc_curve(all_labels_test, all_probs_test, dataset_name="Test", save_path=visualizations_save_path)
        youden_index = tpr - fpr
        best_thresh = thresholds[np.argmax(youden_index)]
        print(f"📈 Best threshold (Youden's J): {best_thresh:.4f}")

        # Apply new threshold
        adjusted_preds = (np.array(all_probs_test) >= best_thresh).astype(int)

        # Save confusion matrix with adjusted threshold
        plot_confusion_matrix(all_labels_test, adjusted_preds, dataset_name="Test_ThresholdAdjusted", save_path=visualizations_save_path)

        # Report
        print("\n📊 Adjusted Threshold Performance:")
        print(classification_report(
            all_labels_test,
            adjusted_preds,
            target_names=["Safe", "Vulnerable"],
            zero_division=0
        ))

    #* STEP 4: SAVE PREDICTIONS AND LABELS TO JSON
    with open(f"{run_history_save_path}/all_test_entry_predictions.json", "w") as f:
        json.dump(prediction_dicts, f, indent=2)

    plot_training_history(history_file_path=f"{run_history_save_path}/training_history.json", save_dir=f"{run_history_save_path}/visualizations")

if __name__ == "__main__":
    main()