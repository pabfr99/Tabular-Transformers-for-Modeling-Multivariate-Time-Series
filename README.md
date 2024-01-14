# Tabular Transformers for Multivariate Time Series Analysis

## Project Description
This project is a reimplementation of the paper "Tabular Transformers for Modeling Multivariate Time Series", originally presented in [arXiv:2011.01843](https://arxiv.org/abs/2011.01843).

## Original Implementation
The original code of the paper is hosted by IBM and can be found at [IBM/TabFormer GitHub repository](https://github.com/IBM/TabFormer).

## Our Implementation
We have developed an implementation of the Tabular transformer specifically designed to handle multivariate time series data. The core of our model consists of a tabular row embedder designed to learn representations of individual rows in a tabular dataset. This is then followed by BERT to learn representations of multiple rows in sequence for an effective time series analysis.

### Training Methodology
The model is trained using a masked language modeling technique. The learned representations are subsequently used for tasks involving regression and classification.

## Datasets
Our implementation focuses on two key datasets:
- **Pollution Dataset**: Used for regression analysis.
- **Credit Card Transaction Dataset**: Used for classification tasks.

## Code Organization and Execution
Due to hardware limitations, our code is organized into two Jupyter notebooks intended for execution in Google Colab:

1. **Credit Card Dataset Notebook**: This notebook includes all the code, encompassing both the classes and the executable code.
2. **PRSA Dataset Notebook**: This notebook is structured with all classes organized in a `src` folder and imported into the notebook. This organization provides a clearer view of the main steps performed.

Users are free to choose either notebook depending on their interest or requirements.

## Data Handling
Given the nature of the tasks in terms of memory and processing requirements, we provide the following resources in a Google Drive folder:
- Raw Data
- Preprocessed Data
- Model Checkpoints

### Accessing the Data
Due to GitHub's file size restrictions, only the notebooks and source code are available on GitHub. All additional resources, including data and checkpoints, are stored in the Google Drive folder. Users will need to create a shortcut to the shared folder in their own Drive and follow the instructions provided in each cell of the notebooks for smooth execution.

**Google Drive Link**: [Google Drive Folder](https://drive.google.com/drive/folders/185vO0N18bxc9-Ad6nsoC-7oAwqQJc2Ek?usp=sharing)

## Results Visualization with Wandb
We use [Weights & Biases (wandb)](https://wandb.ai/) for tracking and visualizing our project results. The links to the wandb projects and their reports are as follows:
- **Wandb Project Links**: [Link to Pollution Project](https://wandb.ai/neural-network-tab-bert/PRSATabBert?workspace=user-ferretti-2039579) , [Link to Credit Card Project](https://wandb.ai/neural-network-tab-bert/CreditTabBert?workspace=user-ferretti-2039579)
- **Wandb Reports**: [Link to Pollution Report - MLM](https://wandb.ai/neural-network-tab-bert/PRSATabBert/reports/Tabular-Transformers-for-Modeling-Multivariate-Time-Series-MLM--Vmlldzo2NTA5MjM5?accessToken=e5it5rmkysozeqxfdwjm5a37ftqvh7mxylyajn0bih10lgqkos3qkk95ap65h58a) , [Link to Pollution Report - Regression](https://wandb.ai/neural-network-tab-bert/PRSATabBert/reports/Tabular-Transformers-for-Modeling-Multivariate-Time-Series-Regression--Vmlldzo2NTA5MjE5?accessToken=v9jycb1bkitn01t3jafqyapcjdxovucpxwpb56wz7wxpo10vjzw14vreixnqnvig) , [Link to Credit Card Report - MLM](https://wandb.ai/neural-network-tab-bert/CreditTabBert/reports/Tabular-Transformers-for-Modeling-Multivariate-Time-Series-MLM--Vmlldzo2NTEyNDIy) , [Link to Credit Card Report - Classification](https://wandb.ai/neural-network-tab-bert/CreditTabBert/reports/Tabular-Transformers-for-Modeling-Multivariate-Time-Series-CLASSIFICATION--Vmlldzo2NTEyNzAw)

## Note
Please ensure you follow the step-by-step instructions in the notebooks to avoid issues related to data paths and dependencies. The notebooks guide through setting up the environment, loading data, model training, and evaluating results.

## Authors
 * **Ferretti Paolo** - [pabfr99](https://github.com/pabfr99)  
 * **Domenico Meconi** -  [DomenicoMeconi](https://github.com/DomenicoMeconi)  


