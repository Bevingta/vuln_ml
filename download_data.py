import argparse
from huggingface_hub import hf_hub_download
import os


def load_huggingface_datasets(repo_id):
    # LOAD DATASET
    train_name = "train.json"
    test_name = "test.json"
    valid_name = "valid.json"

    # Set your target directory
    target_dir = "data"

    # Create the folder if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Download train
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=train_name,
        local_dir=target_dir,
    )

    # Download test
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=test_name,
        local_dir=target_dir,
    )

    # Download valid
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=valid_name,
        local_dir=target_dir,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN model")
    parser.add_argument("--repo-id", type=str, default="alexv26/GNNVulDatasets", help="Repo id for huggingface directory (default: alexv26/GNNVulDatasets)") 
    args = parser.parse_args()
    load_huggingface_datasets(args.repo_id)