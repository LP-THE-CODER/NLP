

---

# TripAdvisor Hotel Reviews Sentiment Analysis

This repository contains a Python script designed for sentiment analysis on hotel reviews from TripAdvisor. The project focuses on natural language processing (NLP) techniques, utilizing the `nltk` library, to preprocess and analyze text data. It employs both traditional machine learning models and a deep learning model to predict sentiment labels.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Word Cloud Visualization](#word-cloud-visualization)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Model](#deep-learning-model)
- [Model Deployment](#model-deployment)
- [Functionality](#functionality)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to analyze sentiment in hotel reviews using a combination of traditional machine learning and deep learning models. The script processes the dataset, explores the data through visualizations, and trains multiple models to predict sentiment labels. The models include Decision Trees, Random Forest, Support Vector Machines, Logistic Regression, K-Nearest Neighbors, and a Bidirectional LSTM deep learning model.

## Key Features

- **Data Preprocessing:** Extensive preprocessing steps, including text cleaning, lemmatization, and stopword removal, are performed to prepare the text data for model training.

- **Exploratory Data Analysis (EDA):** Utilizes Seaborn for visualizing the distribution of ratings and review lengths, providing insights into the dataset.

- **Word Cloud Visualization:** Generates a Word Cloud to visualize the most commonly used words in cleaned reviews, offering a quick glimpse into prevalent sentiments.

- **Machine Learning Models:** Trains traditional machine learning models using cross-validation, including Decision Trees, Random Forest, Support Vector Machines, Logistic Regression, K-Nearest Neighbors, and Naive Bayes.

- **Deep Learning Model:** Implements a Bidirectional LSTM model using TensorFlow and Keras, trained on tokenized and padded text data for sentiment predictions.

- **Model Deployment:** Saves the Logistic Regression model for future use and provides a function to make predictions using both traditional machine learning and deep learning models.

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
   The script will output visualizations and performance metrics for each model. Additionally, use the provided functions for making predictions on new text data.

## Dependencies

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`)

## Installation

To set up the project environment, follow these steps:

1. Install Python: [Download Python](https://www.python.org/downloads/)

2. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tripadvisor-sentiment-analysis.git
   cd tripadvisor-sentiment-analysis
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset (`tripadvisor_hotel_reviews.csv`) used in this project is not included in this repository. You can obtain a similar dataset from TripAdvisor or any other source of your choice.

## Code Structure

The main script, `sentiment_analysis.py`, contains the entire workflow, from data loading to model training and evaluation. The code is organized into sections for clarity.

## Exploratory Data Analysis (EDA)

EDA involves visualizing the dataset's characteristics, including rating distributions and review lengths, using Seaborn.

## Data Preprocessing

Text data is preprocessed through cleaning, lemmatization, and stopword removal to enhance the quality of input for model training.

## Word Cloud Visualization

A Word Cloud is generated to visually represent frequently occurring words in the cleaned reviews.

## Machine Learning Models

Traditional machine learning models are trained using cross-validation, and their accuracy is evaluated.

## Deep Learning Model

A Bidirectional LSTM model is implemented using TensorFlow and Keras for deep learning-based sentiment analysis.

## Model Deployment

The Logistic Regression model is saved for future use, and functions are provided for making predictions using both traditional machine learning and deep learning models.

## Functionality

- **Sentiment Prediction:**
  - The script provides functions (`ml_predict` and `dl_predict`) for making predictions on new text data using both Logistic Regression and Bidirectional LSTM models.

## Results and Visualizations

The script outputs visualizations and performance metrics for each model, providing insights into their effectiveness.

## Contributing

Feel free to contribute to this project! Open issues or submit pull requests to enhance the functionality or fix any bugs.



Feel free to explore, modify, and adapt the code for your own sentiment analysis tasks. If you find this project helpful, don't forget to give it a star! 🌟

---

