# Sentiment Analysis Showdown: Traditional SVM vs. Deep Learning (BERT)

## Overview
This project compares the performance of a **traditional machine learning approach** (Support Vector Machine) with a **deep learning transformer-based model** (BERT) for sentiment analysis on textual data. The goal is to highlight the differences in accuracy, training time, and generalization between classical NLP methods and state-of-the-art deep learning models.

## Approach
1. **Data Preprocessing**
   - Text cleaning: lowercasing, punctuation removal, stopword removal
   - Tokenization
   - For SVM: Feature extraction using **TF-IDF vectorization**
   - For BERT: WordPiece tokenization with pre-trained BERT tokenizer

2. **Model Training**
   - **SVM Model**: Linear kernel trained on TF-IDF features
   - **BERT Model**: Fine-tuned pre-trained BERT for sequence classification

3. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Direct performance comparison between SVM and BERT

## Tools & Libraries
- **Python 3**
- **Scikit-learn** – SVM implementation and evaluation metrics
- **NLTK** – Text preprocessing
- **Transformers (Hugging Face)** – BERT model and tokenizer
- **PyTorch** – Deep learning framework for BERT training
- **Pandas / NumPy** – Data manipulation
- **Matplotlib / Seaborn** – Result visualization

## Results
- **SVM**: Faster training, good baseline accuracy
- **BERT**: Higher accuracy and better contextual understanding, but more computationally expensive
- The comparison demonstrates the trade-off between **speed** and **accuracy** in NLP model selection


## Google Colab Notebook
[Click here to open in Colab](https://colab.research.google.com/drive/1MP4cUpXmNpx4DWKM5NYy9lLNlhaGpsb1)

## Dataset
[Click here open Dataset](https://drive.google.com/drive/folders/16xn2rYSayqrugNp138Sq1Hairzxq3pCS)
