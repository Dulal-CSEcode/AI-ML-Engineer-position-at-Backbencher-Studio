# Sentiment Analysis on Movie Reviews

## Overview
This project implements **sentiment analysis** on a dataset of movie reviews, classifying each review as **positive** or **negative**. The main objective is to process raw text data, apply natural language processing (NLP) techniques, and build a machine learning model that accurately predicts sentiment.

## Approach
1. **Data Preprocessing**  
   - Tokenization and stopword removal  
   - Lemmatization for word normalization  
   - Vectorization using **TF-IDF** to convert text into numerical features  
   
2. **Model Training**  
   - Implemented **Logistic Regression** as the classification algorithm  
   - Trained on labeled movie review datasets  

3. **Evaluation**  
   - Measured accuracy, precision, recall, and F1-score  
   - Tested on unseen data to evaluate generalization capability  

## Tools & Libraries
- **Python 3**  
- **NLTK** – Natural Language Toolkit for text preprocessing  
- **Scikit-learn** – Machine learning models and evaluation metrics  
- **Pandas / NumPy** – Data handling and manipulation  
- **Matplotlib / Seaborn** – Data visualization  

## Results
- Achieved **high accuracy** (above 85%) in predicting sentiment on test data  
- Confusion matrix showed a balanced classification for both positive and negative reviews  
- Demonstrated the effectiveness of TF-IDF + Logistic Regression for text classification tasks  


## Google Colab Notebook
[Click here to open in Colab](https://colab.research.google.com/drive/1MP4cUpXmNpx4DWKM5NYy9lLNlhaGpsb1)

## Dataset
[Click here open Dataset](https://drive.google.com/drive/folders/16xn2rYSayqrugNp138Sq1Hairzxq3pCS)
