# Distillgpt_MedLM

## Training gpt2 on Diseases_Symptoms Dataset



1. LanguageDataset Class:
Description:
The LanguageDataset class is designed to handle text dataset ingestion and preprocessing. It extends the PyTorch Dataset to facilitate ingestion from Pandas DataFrames and prepare data for training neural network models. The class is responsible for tasks such as tokenization, padding/truncation to the maximum length, and numericalization of text.

Key Responsibilities:

Ingest text data from Pandas DataFrame.
Tokenize text using a provided tokenizer.
Find the maximum length for padding/truncation.
Convert text to tokenized, numerical tensors.

2. Main Module:
Description:
The main module serves as the entry point for training a Transformer model on a medical dataset. It includes several key functionalities, such as loading the Diseases_Symptoms dataset and preprocessing it for training the Transformer model. This involves converting the dataset to a DataFrame, tokenizing the text, creating a train/valid split, initializing the model, optimizer, loss function, and results DataFrame. Additionally, it sets parameters for the training loop, including batch size, learning rate, and the number of epochs.

Key Responsibilities:

Load Diseases_Symptoms dataset and preprocess for training Transformer model.
Convert dataset to a DataFrame, tokenize text, and create train/valid split.
Initialize the model, optimizer, loss function, and results DataFrame.
Set parameters for the training loop, including batch size, learning rate, and number of epochs.
Training Loop Functionality:

Train a Transformer model on a medical dataset.
Perform training and validation loops over epochs.
Save the trained model.
Additional functions for loading the model and performing inference.
Summary:
The two modules collectively form a comprehensive pipeline for preparing, training, and evaluating a Transformer model on a medical dataset. The LanguageDataset module and class are focused on data preprocessing, while the main module orchestrates the training process, providing a clear structure for loading, preprocessing, training, and saving the model. The separation into modules enhances modularity and readability, making the codebase more maintainable.
