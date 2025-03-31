# Hybrid Attention Network (HAN) for Stock Market Sentiment Analysis

## Overview
This project implements a **Hybrid Attention Network (HAN)** for **stock market sentiment analysis** using **Self-Paced Learning (SPL)**. The model leverages **pre-trained GloVe embeddings**, **BiLSTM**, and **attention mechanisms** to classify financial news into sentiment categories.

## Features
- **Hybrid Attention Mechanism**: Captures important words and sentences for sentiment analysis.
- **Self-Paced Learning (SPL)**: Weighs training samples dynamically based on loss values.
- **Pre-trained GloVe Embeddings**: Enhances text representation.
- **Model Training & Prediction**: Includes scripts for training (`train.py`) and inference (`predict.py`).
- **Preprocessing Pipeline**: Text cleaning, tokenization, and padding.

## Project Structure
```
├── data/
│   ├── financial_news.csv       # Dataset
│   ├── glove.6B.100d.txt        # Pre-trained GloVe embeddings
│
├── saved_model/
│   ├── tokenizer.pickle         # Tokenizer object
│   ├── han_stock_model.weights.h5 # Trained model weights
│
├── src/
│   ├── preprocess.py            # Data preprocessing functions
│   ├── model.py                 # Hybrid Attention Network model
│   ├── spl.py                   # Self-Paced Learning implementation
│   ├── train.py                 # Model training script
│   ├── predict.py               # Model inference script
│
└── README.md                    # Project documentation
```

## Installation & Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained GloVe Embeddings** (if not included):
   ```bash
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip -d data/
   ```

## Usage
### Training the Model
Run the training script:
```bash
python src/train.py
```
This will:
- Load financial news data
- Tokenize and preprocess text
- Train the HAN model with SPL
- Save trained model weights & tokenizer

### Making Predictions
Use `predict.py` to classify new financial news headlines:
```bash
python src/predict.py
```
Example prediction:
```python
headline = "Stock market hits record high amid positive earnings reports"
label, prediction_prob = predict_sentiment(headline)
print(f"Predicted Label: {label}, Probabilities: {prediction_prob}")
```

## Model Details
### **Hybrid Attention HAN**
- **Embedding Layer**: Uses pre-trained GloVe embeddings
- **BiLSTM Layer**: Captures sequential dependencies
- **Attention Layer**: Focuses on important words
- **Fully Connected Output Layer**: Predicts sentiment categories

### **Self-Paced Learning (SPL)**
- Adjusts sample importance based on loss values
- Helps model focus on easier samples first

## References
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Self-Paced Learning](https://proceedings.neurips.cc/paper_files/paper/2010/file/edfd32c9b7a2b5e0d04adf8853f8be1d-Paper.pdf)

## License
MIT License

