import argparse
import torch
from model import GNNModel
from data.GraphDataset import GraphDataset
from data.graph_gen.graph_generator import generate_one_graph
from gensim.models import Word2Vec
from torch_geometric.loader import DataLoader
import re
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


def extract_all_c_functions(file_path):
    with open(file_path, 'r') as f:
        code = f.read()

    # Strip comments (basic but sufficient here)
    import re
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)

    functions = []
    i = 0
    while i < len(code):
        # Skip until we find a likely function header (ends with ') {')
        header_match = re.search(r'[a-zA-Z_][\w\s\*\(\),\[\]]+\)\s*\{', code[i:])
        if not header_match:
            break

        start = i + header_match.start()
        i = start
        brace_count = 0
        in_string = False
        escape = False
        func_end = None

        while i < len(code):
            c = code[i]

            if c == '"' and not escape:
                in_string = not in_string
            elif c == '\\' and not escape:
                escape = True
                i += 1
                continue
            else:
                escape = False

            if in_string:
                i += 1
                continue

            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    func_end = i + 1
                    break
            i += 1

        if func_end:
            func_code = code[start:func_end].strip()
            functions.append(func_code)
            i = func_end
        else:
            # failed to close properly
            break

    return functions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a function for vulnerabilities through a pre-trained model")
    parser.add_argument("--func", type=str, default=None, required=False)
    parser.add_argument("--func-path", type=str, default=None, required=False)
    parser.add_argument("--saved-model-path", type=str, default="saved_models/saved_model.pth")
    parser.add_argument("--w2v", type=str, default="data/w2v/word2vec_code.model")
    args = parser.parse_args()

    func, func_path, model_path, w2v = args.func, args.func_path, args.saved_model_path, args.w2v

    if func and func_path:
        raise ValueError("Only one of 'func' or 'func_path' should be provided, not both.")

    if (func and func_path) or (not func and not func_path):
        raise ValueError("You must provide exactly one of 'func' or 'func_path'.")
    
    if func_path:
        funcs = extract_all_c_functions(func_path)
    if func:
        funcs = [func]

    for func in funcs:
        prediction = predict_single_function(func, model_path, w2v)
        print(func)
        print(f"This code in the above function is predicted to be {prediction}")