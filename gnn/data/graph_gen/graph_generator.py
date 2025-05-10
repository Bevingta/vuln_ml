import json
import os
import networkx as nx
from .regex_based_graph_generator import RegexBasedGraphGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DB_DIR = os.path.join(BASE_DIR, "../graph_database")

def get_functions(filename, limit=None, start=None, end=None):
    """Load C functions from your JSON file format."""
    functions = {}
    with open(filename, "r") as f:
        data = json.load(f)
        line_number = 0
        for line in data:
            # Check if we've reached the limit (if specified)
            if limit is not None and len(functions) >= limit:
                break
                
            # Check if the current line is within the start-end range (if specified)
            if start is not None and end is not None:
                if line_number < start or line_number > end:
                    line_number += 1
                    continue
            
            # Store the function and any associated metadata
            functions[line_number] = line['func']
            # If vulnerability data is available, store it as well
            if 'cve_id' in line:
                functions[f'cve_{line_number}'] = line['cve_id']
            if 'cvss_score' in line:
                if line['cvss_score'] == 'None':
                    functions[f'cvss_{line_number}'] = 0
                else:
                    functions[f'cvss_{line_number}'] = line['cvss_score']
            if 'cvss_severity' in line:
                functions[f'severity_{line_number}'] = line['cvss_severity']
            line_number += 1
    return functions

def create_graph_database(input_json, output_dir="graph_database", num_functions=None, start=None, end=None):
    """Generate and save all graphs to create a database for GNN training."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading functions from {input_json}")
    functions = get_functions(input_json, num_functions, start, end)
    
    # Extract just the function code
    function_code = {}
    metadata = {}
    for key, value in functions.items():
        if isinstance(key, int):
            function_code[key] = value
        else:
            # Extract index from key (like 'cve_5' -> 5)
            parts = key.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                index = int(parts[1])
                if index not in metadata:
                    metadata[index] = {}
                metadata[index][parts[0]] = value
    
    print(f"Processing {len(function_code)} functions")
    
    # Initialize the graph generator
    generator = RegexBasedGraphGenerator()
    generator.set_functions(function_code)
    
    # Generate graphs
    print("Generating graphs...")
    graphs, function_to_graph = generator.generate_graphs()
    print(f"Successfully generated {len(graphs)} graphs")
    
    # Save all graphs in GraphML format
    print("Saving graphs to database...")
    graphml_dir = os.path.join(output_dir, "graphml")
    os.makedirs(graphml_dir, exist_ok=True)
    
    # Create a mapping file to help with data loading
    graph_mapping = {
        'total_graphs': len(graphs),
        'function_to_graph': {},
        'graph_to_function': {},
        'vulnerable_functions': []
    }
    
    # Save each graph with its metadata
    for func_idx, graph in function_to_graph.items():
        # Add vulnerability and other metadata if available
        if func_idx in metadata:
            for key, value in metadata[func_idx].items():
                graph.graph[key] = value
                if key == 'cve' and value:  # If it has a CVE, mark as vulnerable
                    graph_mapping['vulnerable_functions'].append(func_idx)
        
        # Save the graph
        graphml_path = os.path.join(graphml_dir, f"function_{func_idx}.graphml")
        generator.save_graph(graph, graphml_path)
        
        # Update mapping
        graph_idx = len(graph_mapping['function_to_graph'])
        graph_mapping['function_to_graph'][func_idx] = graph_idx
        graph_mapping['graph_to_function'][graph_idx] = func_idx
    
    # Save graph statistics for easy reference
    graph_stats = []
    for func_idx, graph in function_to_graph.items():
        stats = generator.extract_graph_features(graph)
        stats['function_id'] = func_idx
        
        # Add metadata
        if func_idx in metadata:
            for key, value in metadata[func_idx].items():
                stats[key] = value
        
        graph_stats.append(stats)
    
    # Save statistics to JSON
    with open(os.path.join(output_dir, "graph_stats.json"), "w") as f:
        # Ensure all values are JSON serializable
        for stats in graph_stats:
            if 'vulnerability_flags' in stats:
                stats['vulnerability_flags'] = dict(stats['vulnerability_flags'])
        json.dump(graph_stats, f, indent=2)
    
    # Save mapping file
    with open(os.path.join(output_dir, "graph_mapping.json"), "w") as f:
        json.dump(graph_mapping, f, indent=2)
    
    print(f"All {len(graphs)} graphs saved to {graphml_dir}")
    print(f"Graph statistics and mappings saved to {output_dir}")
    
    return graphs, function_to_graph, graph_stats

def generate_graphs(in_dataset, output=GRAPH_DB_DIR, num_functions=None):
    create_graph_database(in_dataset, output, num_functions)

def save_function(data: dict, debug_dir = "debug_functions"):
    os.makedirs(debug_dir, exist_ok=True)
    func_code = data["func"]
    entry_idx = data["idx"]
    file_location = f"{debug_dir}/function_{entry_idx}.c"
    with open(file_location, "w") as f:
        f.write(func_code)
    return file_location


# Create graph for ONE function
def generate_one_graph(in_dataset:list, idx:int, save_graphs, output_dir=GRAPH_DB_DIR):
    entry_idx = in_dataset[idx]["idx"]
    func_code = in_dataset[idx]["func"] # CODE IN FUNCTION
    graph_name = f"function_{entry_idx}.graphml"
    path_to_graph = os.path.join(GRAPH_DB_DIR, graph_name)
    #* CHECK IF GRAPH EXISTS. IF SO, FIND AND RETURN IT
    if (os.path.exists(path_to_graph)):
        G = nx.read_graphml(path_to_graph)
    else:
        #* STEP 1: CREATE DATA DICT
        data = {
            "idx": entry_idx,
            "func": func_code
        }
        #* STEP 2: SAVE DEBUG FUNCTION FILE
        path_to_debug_func = save_function(data)

        #* STEP 3: MAKE GRAPH
        generator = RegexBasedGraphGenerator()
        G = generator._build_graph_from_function(func_code)

        #* STEP 4: DELETE DEBUG FUNCTION FILE
        os.remove(path_to_debug_func)
        
        #* STEP 5: SAVE GRAPH
        if save_graphs:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            generator.save_graph(G, path_to_graph)

    return G

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create graph database for GNN training")
    parser.add_argument("--input", "-i", default="all_train_dataset.json", 
                        help="Input JSON file with C functions")
    parser.add_argument("--output", "-o", default="graph_database", 
                        help="Output directory for graph database")
    parser.add_argument("--num-functions", "-n", type=int, default=None, 
                        help="Number of functions to process (default: all)")
    parser.add_argument("--start", "-s", type=int, default=None,
                        help="Start index of the functions to grab")
    parser.add_argument("--end", "-e", type=int, default=None,
                        help="End index of the functions to grab")
    
    args = parser.parse_args()
    print("Start: ", args.start)
    print("End: ", args.end)

    if args.num_functions and (args.start or args.end):
        print("Cannot specify both num-functions and start/end")
    else:
        create_graph_database(
            args.input,
            args.output,
            args.num_functions,
            args.start,
            args.end
        )
'''