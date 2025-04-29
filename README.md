# DATA_SCIENCE_PROJECT

### Dataset Source

Paper Reviews Dataset (UCI Machine Learning Repository)  
🔗 https://archive.ics.uci.edu/dataset/410/paper+reviews
Sentiment Analysis on Paper Reviews Dataset Using Machine Learning and Deep Learning
1. Project Overview
This project explores multilingual sentiment analysis using a dataset of academic paper reviews in both English and Spanish. It compares classical machine learning models (Logistic Regression, SVM) and deep learning models (LSTM and Bidirectional LSTM) for sentiment classification into Positive, Neutral, and Negative categories.

2.Technologies and Tools Used
Python 3.11

scikit-learn

TensorFlow / Keras

NLTK

matplotlib & seaborn

pandas & numpy

fastText pretrained embeddings

3. Repository Structure
bash
Copy
Edit
├── notebooks/        # Jupyter Notebooks containing all code


├── LICENSE           # Project license


└── README.md         # Project documentation

All experiments, preprocessing, model training, and evaluation are organized within the notebooks/ folder.

5. Key Methods and Approaches
TF-IDF Vectorization for Logistic Regression and SVM

fastText Embeddings with LSTM

Text Preprocessing: cleaning, tokenization, lemmatization, stopword removal

Dataset Balancing: Upsampling minority classes

Model Training and Evaluation:

Logistic Regression (with and without tuning)

SVM

LSTM and Bidirectional LSTM

Evaluation Metrics:

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

Training Curves (for deep learning models)

5.Results Summary

Model	Accuracy	Weighted F1-Score	Notes
Logistic Regression	0.6625	~0.63	Struggled with Neutral sentiment
Tuned Logistic Regression	0.6500	~0.64	Minor improvement after tuning
SVM	0.6500	~0.56	Poor Neutral detection
Final Bidirectional LSTM	0.7600	0.76	Best overall performance
6. Applications
Academic peer review sentiment analysis

Automated feedback and review classification systems

Multilingual sentiment monitoring for customer platforms

Moderation of user-generated content

7. Future Work
Explore transformer-based models (e.g., XLM-RoBERTa)

Increase dataset size and language coverage

Improve Neutral class classification

