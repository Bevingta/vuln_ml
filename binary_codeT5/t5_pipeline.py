import subprocess
import os
from huggingface_hub import snapshot_download

def main():
    print("Downloading pretrained model...")
    target_directory = "codet5_binary_model"
    os.makedirs(target_directory, exist_ok=True)

    snapshot_download(
        repo_id="bryanokeefe/Trained_codeT5",
        repo_type="dataset",
        local_dir=target_directory,
        local_dir_use_symlinks=False  # ensures real files, not symlinks
    )

    print("Training model...")
    subprocess.run(["python", "T5_train_bin.py"])
    print("Testing model...")
    subprocess.run(["python", "T5_test_bin.py"])

if __name__ == "__main__":
    main()