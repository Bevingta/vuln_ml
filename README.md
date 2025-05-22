# VulnML
This is the final project for _Topics in Computational Intelligence: Machine Learning Projects_ with Professor Bento at Boston College. 

The goal of this project was to identify the most effective machine learning approach for detecting security vulnerabilities in source code. To establish a performance baseline, we first developed a static vulnerability analyzer that uses pattern-matching and simple text-recognition heuristics. Building upon this, we explored three machine learning architectures: a Graph Neural Network (GNN) that leverages the structural semantics of code, a random forest classifier using manually engineered features, and a transformer-based model (CodeT5) that applies pre-trained deep learning to code understanding.

## Setup
**1) Clone the repository** \
`$ git clone https://github.com/Bevingta/vuln_ml.git` \
**2) Install required libraries** \
After navigating to the project directory, run: `$ pip install -r requirements.txt`  

## Data Preparation
The dataset file should be contained in the directory named _data_. You can either use your own dataset, or download one of our pre-made datasets. If you use your own dataset, you entries in the dataset need to follow the following format:
```
{
    "idx": 253185,
    "func": "...",
    "target": 0 or 1,
    "cwe": [
      "CWE-###"
    ],
    "cve": "CVE-####-####",
    "database_origin": "bigvul"
  }
```
All of the entries should be contained in an array in a json file. If you would like to use one of our pre-made datasets, you can use one of the following datasets: [PrimeVul Upsampled](https://huggingface.co/datasets/alexv26/PrimeVulOversampled), [BigVul Upsampled](https://huggingface.co/datasets/alexv26/BigVulOversampled), or our [Combined Dataset](https://huggingface.co/datasets/alexv26/GNNVulDatasets). To download a database from huggingface, you can use the following command: \
`$ python download_data.py --repo-id repo_id` \
where `repo_id` is a repo id such as _username/dataset_name_. 

## Run Static Analyzer
`$ cd rule_based_benchmark` \
`$ python run_rule_based_benchmark.py`

## Run GNN
Navigate to the `gnn` directory. \
**1) Dataset setup** \
Ensure datasets are downloaded in the `vuln_ml/data` directory.

**2) Set configs** \
In the configs.json file, you can change the configs for the execution, including the number of epochs, patience (how many epochs without improvement until model stops early), L2 regularization rate, dropout rate, etc.

**3) Run the pipeline with propper arguments**
To run the code, use the following command: `$ python gnn_pipeline.py` \
You can use any of the following arguments:
```
--in-dataset PATH                   # Path to the complete dataset (default: data/databases/complete_dataset.json)
--train-dataset PATH                # Path to the training dataset split (default: data/split_datasets/train.json)
--test-dataset PATH                 # Path to the testing dataset split (default: data/split_datasets/test.json)
--valid-dataset PATH                # Path to the validation dataset split (default: data/split_datasets/valid.json)
--upsample-vulnerable True/False    # Whether to upsample vulnerable entries (default: False)
--downsample-safe True/False        # Whether to downsample safe entries (default: False)
--do-data-splitting True/False      # Whether to split data into train/val/test (default: False)
--download-presplit-datasets NAME   # HuggingFace repo name to download pre-split datasets (e.g., alexv26/GNNVulDatasets)
--download-w2v NAME                 # HuggingFace repo name to download pretrained Word2Vec (e.g., alexv26/complete_dset_pretrained_w2v)
--do-lr-scheduling True/False       # Use learning rate scheduler during training (default: True)
--vul-to-safe-ratio N               # Ratio of vulnerable to safe samples (e.g., 3 for 1:3, default: None)
--generate-dataset-only True/False  # Only generate and save dataset splits, skip model training (default: False)
--load-existing-model True/False    # Load a previously trained model from disk (default: False)
--roc-implementation True/False     # Enable ROC-based threshold tuning (default: True)
--architecture-type gcn/gat/rgcn    # Type of GNN model architecture (default: rgcn)
--save-memory True/False            # Regenerate graphs on-the-fly to reduce RAM use (default: False)
```

**4) Evaluation** \
After full execution of the pipeline, the model will save model history and visualizations in a subfolder in the run_history directory. If you want to test the model's performance on a single function, you can run the following command: \
`$ python predict_single_function.py --func func_text --saved-model-path path_to_saved_model` \
where `path_to_saved_model` is a path to one of the saved models in run_history and `func_text` is a function in string format. \
Additionally, if you have a .c file containing multiple functions, you can run the following command: \
`$ python predict_single_function.py --func-path func_path --saved-model-path path_to_saved_model` \
where `func_path` is the path to the .c file you wish to analyze.

## Run Random Forest
Navigate to the `vuln_ml/random_forest` directory. \
**1) Dataset setup** \
Ensure datasets are downloaded in the `vuln_ml/data` directory.
**2) Execute code** \
Run the following command: \
`python rf_pipeline.py`

## Run CodeT5 Transformer
Navigate to the `vuln_ml/binary_codeT5` directory. \
**1) Dataset setup** \
Ensure datasets are downloaded in the `vuln_ml/data` directory.
**2) Execute code** \
Run the following command: \
`python t5_pipeline.py`

# Literature
Please see our paper to read about our methodology and process: [VulnML Paper](https://www.dropbox.com/scl/fi/kexa4ehuy04yr1z89lk2z/VulnML.pdf?rlkey=6lhr5w6en2f6hpnk31lbphkv7&st=m6k9tthb&dl=0)
