# EE6227-ML-Assignment
NTU EE7226 Genetic Algorithm & Machine Learning - Machine Learning Part 

## Assignment Description
- Assignment 1: Train 3 classifiers, Bayes Decision Rule, Naive Bayes and Linear Discriminant Analysis(LDA), on the given dataset. Then give prediction on the test set.
- Assignment 2: Train a binary classification tree for a given dataset, utlizing some data preprocessing techniques for missing values and outliers. Then give prediction on the test set.

Please Find more details for assignment requirements in the `Assignment-1-classifiers/Ass1.pdf` and `Assignment-2-classification_tree/Ass2.pdf`.

## Repository Overview

- `Assignment-1-classifiers/`: 
  - `bayes_lda.pdf`: Submission report.
  - `poc_classifier.ipynb`: Proof of concept for the classifiers.
  - `poc.py`: Generated py file from the notebook.
  - `results/`: Results of the classifiers, .csv and .mat format.
- `Assignment-2-classification_tree/`: 
  - `clf.pdf`: Submission report.
  - `preprocess.ipynb`: Data preprocessing for the training set.
  - `class_tree.ipynb`: Handmade binary classification tree.
  - `results/`: Results of the classifiers, .xlsx format.
  - `preprocess/`: Preprocessed training set, .xlsx format.
    - `add_head.xlsx`: Orignal training set with added head.
    - `sto_cap_train.xlsx`: Stocasitcally imputed + capped training set.
  - `backup/`: Backup file for pre-puring in classfication tree.
- `README.md`: This file.
- `requirements.txt`: List of packages required to run the code.

## Environment

The packages are listed in `requirements.txt`. To create a conda environment with the packages, run the following command:
```bash
conda create --name your_env_name --file requirements.txt
```
