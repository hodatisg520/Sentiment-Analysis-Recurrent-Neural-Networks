# Sentiment Analysis with Recurrent Neural Networks (RNNs)

## Overview
This repository showcases a comprehensive Deep Learning project focused on Natural Language Processing (NLP), specifically **Binary Sentiment Analysis**. The objective is to accurately classify movie reviews as either positive or negative using various advanced Recurrent Neural Network (RNN) architectures.

The project demonstrates the end-to-end Machine Learning lifecycle: from data ingestion and text preprocessing to building, training, and evaluating complex sequential models. By experimenting with different RNN variants, it provides a comparative analysis of their classification accuracy and computational efficiency.

## Dataset
The project utilizes the **IMDB Movie Review Dataset**, a benchmark dataset in the NLP community for sentiment classification.
* **Volume:** 50,000 highly polar movie reviews.
* **Split:** 25,000 reviews for training and 25,000 for testing. (The training set is further divided for validation to monitor early stopping and overfitting).
* **Task:** Binary classification (0 for Negative, 1 for Positive).

## Technical Stack
* **Language:** Python 3.12
* **Deep Learning Framework:** TensorFlow & Keras
* **Data Ingestion:** TensorFlow Datasets (`tfds`)
* **Data Processing:** NumPy

## Model Architectures
To understand the impact of different sequence modeling techniques, this project implements and compares the following architectures:
1. **Gated Recurrent Units (GRU):** A streamlined recurrent cell that effectively captures long-term dependencies while mitigating the vanishing gradient problem, often requiring less computational overhead than standard LSTMs.
2. **Bidirectional LSTM (BiLSTM):** Processes sequences in both forward and backward directions, allowing the network to understand context from both past and future words simultaneously.
3. **Bidirectional GRU (BiGRU):** Combines the computational efficiency of GRUs with the deep contextual understanding of bidirectional processing.

**Common Model Pipeline:**
* **Text Vectorization:** Converts raw text into integer sequences with a constrained vocabulary size (e.g., 10,000 tokens) and maximum sequence length (e.g., 250 tokens).
* **Embedding Layer:** Transforms integer tokens into dense vectors of fixed size (64 dimensions), utilizing masking to correctly handle padded sequence lengths.
* **Recurrent Layer:** GRU / BiLSTM / BiGRU.
* **Dense Layers:** Hidden dense layers with ReLU activation, followed by a final output layer using a Sigmoid activation function to output binary probability.

## Training and Optimization
* **Loss Function:** Binary Crossentropy, ideal for binary classification tasks.
* **Optimizer:** Adam Optimizer (learning rate = 0.001) for robust and adaptive gradient descent.
* **Batching & Epochs:** Trained over 20 epochs using large batch sizes (2048) for stable gradient estimates, evaluated iteratively on a held-out validation set.

## Insights and Evaluation
The models are strictly evaluated on an unseen test set (25,000 samples). 
Key considerations explored in this project include:
* **Bidirectional Context:** How reading text in both directions significantly improves the model's ability to capture sarcasm and complex sentence structures compared to unidirectional models.
* **Vocabulary and Truncation:** The trade-offs between vocabulary size, sequence truncation length, model memory usage, and the retention of rich context.

## Installation and Usage

### Prerequisites
Ensure Python is installed along with the required libraries.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/hodatisg520/Recurrent-Neural-Networks-RNNs-.git
   cd Recurrent-Neural-Networks-RNNs-
   ```

2. Install the necessary dependencies (TensorFlow, TFDS, NumPy, Matplotlib):
   ```bash
   pip install tensorflow tensorflow-datasets numpy matplotlib
   ```

3. Run the Jupyter Notebook to explore the preprocessing, model training, and evaluation steps.

## License
This project is open-source and available under the MIT License.
