## Sentiment Analysis System (From Scratch)

**Project Type**: Machine Learning | Sentiment Analysis  
**Tools Used**: Python, Pandas, Scikit-Learn, NLTK
**Dataset**: [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

### **Objective**  
The objective of this project is to build a sentiment analysis system that can classify text data as either positive or negative based on the sentiment expressed. The goal is to leverage natural language processing (NLP) techniques and machine learning algorithms to understand the sentiment of text, which can be useful for applications such as customer reviews, social media monitoring, and more.

### **Key Steps**
- **Data Preprocessing & Feature Engineering**:
    - Cleaned and preprocessed the text data by removing stop words, punctuation, and performing tokenization.
    - Applied techniques such as lemmatization and stemming to normalize the text.
    - Extracted features using TF-IDF vectorization to represent the text in a machine-readable format.
- **Model Training & Evaluation**:
    - Implemented machine learning models such as Naïve Bayes, Support Vector Machine (SVM), and Logistic Regression for sentiment classification.
    - Evaluated the models using metrics like accuracy, precision, recall, F1-score, and confusion matrix.
    - Compared models to select the best performer, optimizing for recall and F1-score to improve prediction of both positive and negative sentiments.
- **Model Improvement & Fine-Tuning**:
    - Experimented with different preprocessing strategies and hyperparameters to improve model performance.
    - Applied cross-validation and grid search to fine-tune model parameters for optimal performance.

### **Results**  
The best-performing model was Logistic Regression, achieving an accuracy of 77.49% and an F1-score of .77. The sentiment analysis system successfully classified sentiments of text data, demonstrating its ability to differentiate between positive and negative sentiments effectively.

🔗 **View Interactive Notebook (nbviewer)**: [nbviewer Link](https://nbviewer.org/github/christopherFlynn/Sentiment-Analysis-System/blob/main/Sentiment%20Analysis%20System.ipynb)
