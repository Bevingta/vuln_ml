import json
import re
import os
import networkx as nx
import matplotlib.pyplot as plt

class RegexBasedGraphGenerator:
    def __init__(self, json_file_path=None):
        """Initialize with path to JSON file containing C functions."""
        self.json_file_path = json_file_path
        self.functions = []
        if json_file_path:
            self.functions = self._load_functions_from_json()
    
    def _load_functions_from_json(self):
        """Load C functions from JSON file."""
        functions = []
        with open(self.json_file_path, 'r') as f:
            data = json.load(f)
            for line_num, func_code in data.items():
                functions.append(func_code)
        return functions
    
    def set_functions(self, functions_dict):
        """Set functions directly from a dictionary."""
        self.functions = list(functions_dict.values())
    
    def _extract_variables(self, function_code):
        """Extract variables from function code using regex."""
        # Look for variable declarations
        var_pattern = r'(?:int|char|float|double|long|short|unsigned|void|uint\w+_t|int\w+_t|size_t|ssize_t|time_t|FILE|struct\s+\w+|enum\s+\w+|\w+_t)\s+\*?(\w+)(?:\s*\[\s*\w*\s*\])?(?:\s*=\s*[^;]+)?;'
        variables = re.findall(var_pattern, function_code)
        
        # Also find function parameters
        func_header = re.search(r'(\w+)\s*\(([^)]*)\)', function_code)
        if func_header:
            params_str = func_header.group(2)
            # Extract parameter names
            param_pattern = r'(?:const\s+)?(?:int|char|float|double|long|short|unsigned|void|uint\w+_t|int\w+_t|size_t|ssize_t|time_t|FILE|struct\s+\w+|enum\s+\w+|\w+_t)\s+\*?(\w+)(?:\s*\[\s*\w*\s*\])?'
            params = re.findall(param_pattern, params_str)
            variables.extend(params)
        
        return list(set(variables))  # Remove duplicates
    
    def _extract_function_calls(self, function_code):
        """Extract function calls from function code using regex."""
        # Pattern for function calls
        call_pattern = r'(\w+)\s*\([^;{]*\)'
        calls = re.findall(call_pattern, function_code)
        
        # Filter out control structures which can look like function calls
        control_keywords = ['if', 'for', 'while', 'switch', 'return']
        calls = [call for call in calls if call not in control_keywords]
        
        return list(set(calls))  # Remove duplicates
    
    def _extract_control_structures(self, function_code):
        """Extract control structures from function code."""
        structures = []
        
        # Find if statements
        if_pattern = r'if\s*\(([^)]*)\)'
        if_matches = re.finditer(if_pattern, function_code)
        for match in if_matches:
            structures.append(('if', match.group(1), match.start()))
        
        # Find for loops
        for_pattern = r'for\s*\(([^)]*)\)'
        for_matches = re.finditer(for_pattern, function_code)
        for match in for_matches:
            structures.append(('for', match.group(1), match.start()))
        
        # Find while loops
        while_pattern = r'while\s*\(([^)]*)\)'
        while_matches = re.finditer(while_pattern, function_code)
        for match in while_matches:
            structures.append(('while', match.group(1), match.start()))
        
        # Find switch statements
        switch_pattern = r'switch\s*\(([^)]*)\)'
        switch_matches = re.finditer(switch_pattern, function_code)
        for match in switch_matches:
            structures.append(('switch', match.group(1), match.start()))
        
        return structures
    
    def _find_variable_uses(self, function_code, variables):
        """Find where variables are used in the function."""
        variable_uses = {}
        
        for var in variables:
            if not var or len(var) < 2:  # Skip very short variable names to avoid false positives
                continue
                
            # Pattern to find variable uses that are not declarations
            # This is a simple approximation and won't catch all uses
            use_pattern = r'(?<![a-zA-Z0-9_])' + re.escape(var) + r'(?![a-zA-Z0-9_])'
            
            uses = []
            for match in re.finditer(use_pattern, function_code):
                # Skip if it's part of a declaration
                if not re.search(r'(?:int|char|float|double|long|short|unsigned|void|uint\w+_t|int\w+_t|size_t|ssize_t|time_t|FILE|struct|enum|\w+_t)\s+\*?' + re.escape(var), function_code[max(0, match.start()-40):match.start()]):
                    uses.append(match.start())
            
            if uses:
                variable_uses[var] = uses
        
        return variable_uses
    
    def _extract_function_name(self, function_code):
        """Extract the function name from function code."""
        # Try to find function name from the signature
        func_header = re.search(r'(\w+)\s*\(', function_code)
        if func_header:
            return func_header.group(1)
        return "unknown_function"
    
    def _build_graph_from_function(self, function_code):
        """Build a graph representation from function code using regex patterns."""
        G = nx.DiGraph()
        
        try:
            # Extract function name
            function_name = self._extract_function_name(function_code)
            G.add_node(function_name, type='FunctionDefinition')
            
            # Extract variables
            variables = self._extract_variables(function_code)
            for var in variables:
                if var:  # Skip empty string matches
                    G.add_node(var, type='Variable')
                    G.add_edge(function_name, var, type='declares')
            
            # Extract function calls
            function_calls = self._extract_function_calls(function_code)
            for call in function_calls:
                if call != function_name:  # Skip recursive calls for simplicity
                    G.add_node(call, type='FunctionCall')
                    G.add_edge(function_name, call, type='calls')
            
            # Extract control structures
            control_structures = self._extract_control_structures(function_code)
            for i, (struct_type, condition, position) in enumerate(control_structures):
                node_id = f"{struct_type}_{i}"
                G.add_node(node_id, type=f'ControlStructure_{struct_type}', condition=condition)
                G.add_edge(function_name, node_id, type='contains')
                
                # Connect variables to control structures if they appear in the condition
                for var in variables:
                    if var and var in condition:
                        G.add_edge(var, node_id, type='used_in_condition')
            
            # Find variable uses to establish data flow
            variable_uses = self._find_variable_uses(function_code, variables)
            
            # Add data flow edges between variables
            for var, uses in variable_uses.items():
                # Connect variables to function calls if they are used in calls
                for call in function_calls:
                    call_pattern = call + r'\s*\([^)]*' + re.escape(var) + r'[^)]*\)'
                    if re.search(call_pattern, function_code):
                        G.add_edge(var, call, type='used_as_parameter')
                
                # Connect variables to control structures if they are used in the body
                for i, (struct_type, condition, position) in enumerate(control_structures):
                    node_id = f"{struct_type}_{i}"
                    
                    # Find where the control structure body starts and ends
                    body_start = function_code.find('{', position)
                    if body_start != -1:
                        # Simple heuristic to find matching closing brace
                        # This won't work perfectly for nested structures
                        depth = 1
                        body_end = body_start + 1
                        while depth > 0 and body_end < len(function_code):
                            if function_code[body_end] == '{':
                                depth += 1
                            elif function_code[body_end] == '}':
                                depth -= 1
                            body_end += 1
                        
                        # Check if variable is used in the body
                        body = function_code[body_start:body_end]
                        var_in_body_pattern = r'(?<![a-zA-Z0-9_])' + re.escape(var) + r'(?![a-zA-Z0-9_])'
                        if re.search(var_in_body_pattern, body):
                            G.add_edge(var, node_id, type='used_in_body')
            
            # Add static analysis flags for common vulnerability patterns
            vuln_flags = self._check_vulnerability_patterns(function_code, G)
            for flag, exists in vuln_flags.items():
                if exists:
                    G.graph[flag] = True
            
            return G
        
        except Exception as e:
            print(f"Error building graph: {e}")
            # Return a minimal graph even if there was an error
            if G.number_of_nodes() == 0:
                G.add_node("error", type="ParseError", error=str(e))
            return G
    
    def _check_vulnerability_patterns(self, function_code, graph):
        """Check for common vulnerability patterns."""
        flags = {
            'uses_dangerous_function': False,
            'potential_buffer_overflow': False,
            'pointer_arithmetic': False,
            'memory_allocation': False,
            'format_string_vulnerability': False
        }
        
        # Check for dangerous functions
        dangerous_funcs = ['strcpy', 'strcat', 'gets', 'sprintf', 'scanf', 'vsprintf']
        for func in dangerous_funcs:
            if re.search(r'(?<![a-zA-Z0-9_])' + re.escape(func) + r'\s*\(', function_code):
                flags['uses_dangerous_function'] = True
                break
        
        # Check for potential buffer overflow (array access without bounds checking)
        if re.search(r'\[\s*(\w+)\s*\]', function_code) and not re.search(r'if\s*\([^)]*<[^)]*\)', function_code):
            flags['potential_buffer_overflow'] = True
        
        # Check for pointer arithmetic
        if re.search(r'\w+\s*\+\+|\+\+\s*\w+|\w+\s*\+=|\w+\s*=\s*\w+\s*\+', function_code) and re.search(r'\*\w+', function_code):
            flags['pointer_arithmetic'] = True
        
        # Check for memory allocation
        if re.search(r'(?<![a-zA-Z0-9_])(malloc|calloc|realloc|alloca)\s*\(', function_code):
            flags['memory_allocation'] = True
        
        # Check for format string vulnerability
        if re.search(r'(?<![a-zA-Z0-9_])(printf|sprintf|fprintf|snprintf)\s*\([^,)]*,\s*([^,")]*)\)', function_code):
            flags['format_string_vulnerability'] = True
        
        return flags
    
    def generate_graphs(self):
        """Generate graphs for all functions."""
        graphs = []
        function_to_graph = {}
        
        for i, func_code in enumerate(self.functions):
            print(f"Processing function {i+1}/{len(self.functions)}")
            
            # Debug: Write function to temp file for inspection
            debug_dir = "debug_functions"
            os.makedirs(debug_dir, exist_ok=True)
            with open(f"{debug_dir}/function_{i}.c", "w") as f:
                f.write(func_code)
            
            graph = self._build_graph_from_function(func_code)
            
            if graph.number_of_nodes() > 1:  # At least more than just the function node
                graphs.append(graph)
                function_to_graph[i] = graph
                print(f"✓ Successfully processed function {i+1} (nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()})")
            else:
                print(f"✗ Generated minimal graph for function {i+1}")
        
        return graphs, function_to_graph
    
    def visualize_graph(self, graph, output_path):
        """Visualize a graph and save to output path."""
        if graph.number_of_nodes() <= 1:
            print(f"Cannot visualize graph with {graph.number_of_nodes()} nodes")
            return
        
        plt.figure(figsize=(14, 10))
        
        # Create position layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Draw nodes with different colors based on node type
        node_colors = []
        node_sizes = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', '')
            if 'FunctionDefinition' in node_type:
                node_colors.append('red')
                node_sizes.append(1000)  # Larger size for function nodes
            elif 'FunctionCall' in node_type:
                node_colors.append('orange')
                node_sizes.append(700)
            elif 'ControlStructure' in node_type:
                node_colors.append('blue')
                node_sizes.append(700)
            elif 'Variable' in node_type:
                node_colors.append('green')
                node_sizes.append(500)
            else:
                node_colors.append('gray')
                node_sizes.append(300)
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Draw edges with different styles based on edge type
        data_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                      if d.get('type') in ['used_as_parameter', 'used_in_condition', 'used_in_body']]
        call_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                     if d.get('type') == 'calls']
        other_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                      if d.get('type') not in ['used_as_parameter', 'used_in_condition', 'used_in_body', 'calls']]
        
        nx.draw_networkx_edges(graph, pos, edgelist=data_edges, 
                              edge_color='red', style='dashed', alpha=0.7, width=1.5)
        nx.draw_networkx_edges(graph, pos, edgelist=call_edges, 
                              edge_color='blue', style='solid', alpha=0.7, width=2)
        nx.draw_networkx_edges(graph, pos, edgelist=other_edges, 
                              edge_color='black', alpha=0.5, width=1)
        
        # Add labels
        labels = {node: node for node in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels, font_size=10)
        
        # Add vulnerability flags in the title if any are present
        title = "C Function Graph"
        vuln_flags = [flag for flag, exists in graph.graph.items() if exists]
        if vuln_flags:
            title += "\nPotential issues: " + ", ".join(vuln_flags)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    def extract_graph_features(self, graph):
        """Extract basic features from a graph for use in ML models."""
        features = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'num_variables': len([n for n, attrs in graph.nodes(data=True) if attrs.get('type') == 'Variable']),
            'num_function_calls': len([n for n, attrs in graph.nodes(data=True) if attrs.get('type') == 'FunctionCall']),
            'num_control_structures': len([n for n, attrs in graph.nodes(data=True) if 'ControlStructure' in attrs.get('type', '')]),
            'vulnerability_flags': {flag: exists for flag, exists in graph.graph.items() if flag.startswith('uses_') or flag.startswith('potential_')}
        }
        
        # Calculate average degree
        if graph.number_of_nodes() > 0:
            features['avg_degree'] = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        else:
            features['avg_degree'] = 0
        
        return features
    
    def save_graph(self, graph, output_path):
        """Save graph to file in GraphML format for later use."""
        # Make a copy to avoid modifying the original
        G_copy = nx.DiGraph()
        
        # Copy nodes and attributes
        for node, attrs in graph.nodes(data=True):
            G_copy.add_node(node, **attrs)
        
        # Copy edges and attributes
        for u, v, attrs in graph.edges(data=True):
            G_copy.add_edge(u, v, **attrs)
        
        # Copy graph attributes
        for key, value in graph.graph.items():
            G_copy.graph[key] = value
        
        # Save the graph
        nx.write_graphml(G_copy, output_path)
