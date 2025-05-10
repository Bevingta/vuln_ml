from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import os

from .data_processing import split_name_into_subtokens
from .graph_gen.graph_generator import generate_one_graph as gengraph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "../../data/final_data")
W2V_PATH = os.path.join(BASE_DIR, "w2v/word2vec_code.model")


class GraphDataset(Dataset):
    def __init__(self, data, w2v, seen_graphs: dict, save_memory: bool):
        self.data = data
        self.w2v = w2v
        self.embedding_dim = self.w2v.vector_size
        self.seen_graphs = seen_graphs
        self.save_memory = save_memory
        self.skipped_embeddings = 0

    def __len__(self):
        return len(self.data)
    
    def get_vuln_nonvuln_split(self):
        vuln = 0
        nonvuln = 0
        for entry in self.data:
            label = int(entry["target"])
            if label == 1:
                vuln += 1
            else:
                nonvuln += 1
        return vuln, nonvuln
    
    def get_data(self):
        return self.data
    
    def get_skipped_embeddings_count(self):
        return self.skipped_embeddings
    
    def __getitem__(self, idx):

        #* STEP 1: LOAD GRAPH FROM PREPROCESSED GRAPHS
        global_idx = self.data[idx]["idx"]  # Get the global unique ID

        if not self.save_memory:
            G = self.seen_graphs[global_idx]
        elif self.save_memory:
            G = gengraph(self.data, idx, False)

        #* STEP TWO: CREATE GRAPH FEATURES
        # a: node feature matrix
        node_types = {'FunctionCall': 0, 'Variable': 1, 'ControlStructure_if': 2, 'ControlStructure_while': 3, 'ControlStructure_switch': 4, 'ControlStructure_for': 5, 'FunctionDefinition': 6}
 
        node_feature_matrix = torch.zeros((G.number_of_nodes(), len(node_types) + self.embedding_dim))
        node_to_idx = {}
        for index, (node, attrs) in enumerate(G.nodes(data=True)):

            node_to_idx[node] = index

            # One-hot type
            node_type = attrs.get('type', 'unknown')
            if node_type in node_types:
                node_feature_matrix[index, node_types[node_type]] = 1.0

            # Word2Vec embedding of node name subtokens
            subtokens = split_name_into_subtokens(str(node))
            embeddings = []

            for token in subtokens:
                token = token.lower()
                if token in self.w2v.wv:
                    embeddings.append(torch.tensor(self.w2v.wv[token]))
            if embeddings:
                mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
            else:
                # Use zero vector for unseen tokens
                mean_embedding = torch.zeros(self.embedding_dim)
                self.skipped_embeddings += 1
            node_feature_matrix[index, len(node_types):] = mean_embedding


        
        # b: edge index and edge type
        edge_type_map = {
            "declares": 0,
            "calls": 1,
            "contains": 2,
            "used_in_condition": 3,
            "used_as_parameter": 4,
            "used_in_body": 5
        }

        '''edge_index = []
        for u, v in G.edges():
            if u in node_to_idx and v in node_to_idx:
                edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()'''
        edge_index = []
        edge_attr = []

        for u, v, attrs in G.edges(data=True):
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_type_str = attrs.get('type', 'unknown')
            edge_attr.append(edge_type_map.get(edge_type_str, -1))  # handle unknowns safely
        if len(edge_index) == 0:
            # Add dummy self-loop if no edges
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([0], dtype=torch.long)  # default edge type
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        # LABEL HANDLING: Support both cvss_score (float) and target (already binary)
        label = torch.tensor(int(self.data[idx]["target"]), dtype=torch.long)
        
        flag_keys = [
            'uses_dangerous_function',
            'potential_buffer_overflow',
            'pointer_arithmetic',
            'memory_allocation',
            'format_string_vulnerability'
        ]

        flag_vector = torch.tensor(
            [1.0 if G.graph.get(flag, False) else 0.0 for flag in flag_keys],
            dtype=torch.float
        )

        data = Data(x=node_feature_matrix, edge_index=edge_index, edge_attr=edge_attr, y=label, graph_flags=flag_vector)
        return data
