# VulnML
This is the final project for _Topics in Computational Intelligence: Machine Learning Projects_ with Professor Bento. \

The goal of this project was to identify the most effective machine learning approach for detecting security vulnerabilities in source code. To establish a performance baseline, we first developed a static vulnerability analyzer that uses pattern-matching and simple text-recognition heuristics. Building upon this, we explored three machine learning architectures: a Graph Neural Network (GNN) that leverages the structural semantics of code, a Random Forest classifier using manually engineered features, and a Transformer-based model (CodeT5) that applies pre-trained deep learning to code understanding.

## Setup
**1) Clone the repository** \
`$ git clone https://github.com/Bevingta/vuln_ml.git` \
**2) Install required libraries** \
After navigating to the project directory, run: `$ pip install -r requirements.txt` \

## Run Static Analyzer
`$ cd rule_based_benchmark` \
`$ python run_rule_based_benchmark.py`

## Run GNN
Navigate to the gnn directory: `$ cd gnn` \
**1) Dataset setup** 
