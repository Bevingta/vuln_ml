import argparse
import torch
from model import GNNModel
from data.GraphDataset import GraphDataset
from data.graph_gen.graph_generator import generate_one_graph
from gensim.models import Word2Vec
from torch_geometric.loader import DataLoader
def predict_single_function(func_entry, model_path, w2v_model_path, model_type="rgcn", hidden_dim=128, output_dim=2, dropout=0.5, device=None, threshold=0.5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load w2v model
    w2v = Word2Vec.load(w2v_model_path)

    # make the function into a dataset-like entry
    dataset_like_entry = [{
        "idx": 0,
        "func": func_entry,
        "target": -1 # unknown
        }]

    data = GraphDataset(data=dataset_like_entry, w2v=w2v, seen_graphs={}, save_memory=True)

    # Instantiate and load model
    input_dim = 107
    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, model=model_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Add batch dimension manually
    loader = DataLoader(data, batch_size=1)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            probs = torch.softmax(out, dim=1)
            vuln_prob = probs[0][1].item()
            is_vulnerable = vuln_prob >= threshold

    return "Vulnerable" if is_vulnerable else "Safe"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a function for vulnerabilities through a pre-trained model")
    parser.add_argument("--func", type=str, required=True)
    parser.add_argument("--saved-model-path", type=str, default="saved_models/saved_model.pth")
    parser.add_argument("--w2v", type=str, default="data/w2v/word2vec_code.model")
    args = parser.parse_args()

    func, model_path, w2v = args.func, args.saved_model_path, args.w2v

    prediction = predict_single_function(func, model_path, w2v)
    print(f"This code is predicted to be {prediction}")