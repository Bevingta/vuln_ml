import json
from huggingface_hub import hf_hub_download
import os

def load_configs():
    configs_file_path = "configs.json"
    with open(configs_file_path, 'r') as file:
        configs: dict = json.load(file)
    return (configs["input_dim"], configs["hidden_dim"], configs["output_dim"], 
            configs["dropout"], configs["l2_reg"], configs["batch_size"], configs["learning_rate"], 
            configs["epochs"], configs["downsample_factor"], configs["patience"])

def load_w2v_from_huggingface(repo_id):
    w2v_file1 = "word2vec_code.model"
    w2v_file2 = "word2vec_code.model.syn1neg.npy"
    w2v_file3 = "word2vec_code.model.wv.vectors.npy"

    target_dir = "data/w2v"

    os.makedirs(target_dir, exist_ok=True)

    # Download file 1
    hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=w2v_file1,
        local_dir=target_dir,
    )

    # Download file 2
    hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=w2v_file2,
        local_dir=target_dir,
    )

    # Download file 3
    hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=w2v_file3,
        local_dir=target_dir,
    )


def early_stopping(best_f1, val_f1, patience=5, epochs_without_improvement=0):
    if val_f1 > best_f1:
        best_f1 = val_f1
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"Epochs without improvement: {epochs_without_improvement}")
        if epochs_without_improvement >= patience:
            print("Early stopping...")
            return True, epochs_without_improvement
    return False, epochs_without_improvement


from tqdm import tqdm

def analyze_word2vec_coverage(dataset, w2v, embedding_dim):
    total_nodes = 0
    matched_nodes = 0
    unmatched_tokens = {}

    for i in tqdm(range(len(dataset)), desc="Analyzing W2V Coverage"):
        data = dataset[i]  # This calls __getitem__
        x = data.x  # shape: [num_nodes, total_dim]
        num_nodes = x.size(0)
        total_nodes += num_nodes

        # Check which rows are zero vectors in embedding portion
        # node_types = 7 → embeddings start at index 7
        for row in x[:, 7:]:
            if row.abs().sum() > 1e-6:  # non-zero embedding
                matched_nodes += 1

    print(f"📊 Word2Vec coverage: {matched_nodes}/{total_nodes} "
          f"({100 * matched_nodes / total_nodes:.2f}%)")
