

# TripAdvisor Hotel Reviews Sentiment Analysis

This repository contains a Python script for sentiment analysis on hotel reviews from TripAdvisor. The project focuses on natural language processing (NLP) techniques, utilizing the `nltk` library, to preprocess and analyze the text data. It further employs both traditional machine learning models (such as Decision Trees, Random Forest, Support Vector Machines, Logistic Regression, K-Nearest Neighbors, and Naive Bayes) and a deep learning model (Bidirectional LSTM) to predict sentiment labels.

## Key Features

- **Data Preprocessing:** The script includes extensive data preprocessing steps, such as text cleaning, lemmatization, and removal of stopwords and punctuation. The goal is to prepare the text data for effective machine learning and deep learning model training.

- **Exploratory Data Analysis (EDA):** EDA is performed using Seaborn to visualize the distribution of ratings and the lengths of reviews, providing insights into the dataset.

- **Word Cloud Visualization:** A Word Cloud is generated to visualize the most commonly used words in the cleaned reviews, offering a quick glimpse into the prevalent sentiments.

- **Machine Learning Models:** Traditional machine learning models, including Decision Trees, Random Forest, Support Vector Machines, Logistic Regression, K-Nearest Neighbors, and Naive Bayes, are trained and evaluated using cross-validation. The Logistic Regression model is saved for later use.

- **Deep Learning Model:** A deep learning model is implemented using TensorFlow and Keras. This Bidirectional LSTM model is trained on the tokenized and padded text data, achieving sentiment predictions.

- **Model Deployment:** The Logistic Regression model is saved and loaded for future use. Additionally, a function is provided to make predictions using both the traditional machine learning and deep learning models.

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/tripadvisor-sentiment-analysis.git
   cd tripadvisor-sentiment-analysis
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Script:**
   ```bash
   python sentiment_analysis.py
   ```

4. **Explore the Results:**
   The script will output visualizations and performance metrics for each model. You can also use the provided functions for making predictions on new text data.

## Requirements

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`)

Feel free to explore, modify, and adapt the code for your own sentiment analysis tasks. If you find this project helpful, don't forget to give it a star! 🌟

---

