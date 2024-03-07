# Project Name
**Multiclass Multilabel Prediction for Stack Overflow Questions**

## Table of Contents
* [General Information](#general-information)
  * [Overview](#overview)
  * [Dataset](#dataset)
  * [Objective](#objective)
  * [Methodology](#methodology)
* [Technologies Used](#technologies-used)
  * [Key Libraries and Their Roles](#key-libraries-and-their-roles)
    * [Natural Language Processing (NLP) Libraries](#natural-language-processing-nlp-libraries)
    * [Machine Learning and Deep Learning Libraries](#machine-learning-and-deep-learning-libraries)
    * [Data Visualization Libraries](#data-visualization-libraries)
* [Conclusions](#conclusions)
* [Project Files](#project-files)
* [Future Work](#future-work)
* [Contact](#contact)

## General Information

### Overview
This project focuses on building a model to identify spam questions on Quora, employing NLP techniques and deep learning models with GloVe Embeddings to distinguish between spam and legitimate content.

### Dataset
The dataset for this project can be accessed [here](https://www.dropbox.com/sh/kpf9z73woodfssv/AAAw1_JIzpuVvwteJCma0xMla?dl=0).

### Objective
The primary goal is to develop an effective spam filter for Quora questions, enhancing content quality and user experience on the platform.

### Methodology
1. Data preprocessing included using GloVe embeddings for dimensionality reduction.
2. Exploratory Data Analysis (EDA) involved visualizing data using word clouds and unigram/bigram plots.
3. Developed models: GRU and Bidirectional GRU, with class weights to address the highly imbalanced dataset skewed towards the negative class (94% ham, 6% spam).
4. Performance was evaluated using accuracy, ROC AUC, precision-recall, and other relevant metrics.

## Technologies Used

This project leverages a variety of technologies, libraries, and frameworks:

- **Conda**: 23.5.2
- **Python**: 3.8.18
- **NumPy**: 1.22.3
- **Pandas**: 2.0.3
- **Matplotlib**:
  - Core: 3.7.2
  - Inline: 0.1.6
- **Seaborn**: 0.12.2
- **Scikit-learn**: 1.3.0
- **TensorFlow and Related Packages**:
  - TensorFlow Dependencies: 2.9.0
  - TensorFlow Estimator: 2.13.0 (via PyPI)
  - TensorFlow for macOS: 2.13.0 (via PyPI)
  - TensorFlow Metal: 1.0.1 (for GPU acceleration on macOS, via PyPI)
- **NLTK**: 3.8.1
- **Beautiful Soup**: 4.12.3
- **WordCloud**: 1.9.2

### Key Libraries and Their Roles:

#### Natural Language Processing (NLP) Libraries:
- **NLTK**: Utilized for text processing and analysis tasks such as tokenization, stemming, and lemmatization.
- **Beautiful Soup**: Employed for HTML parsing and cleaning, essential for processing web-sourced textual data.
- **WordCloud**: Used to generate visual word clouds from text data, aiding in the visual analysis of text features.

#### Machine Learning and Deep Learning Libraries:
- **Scikit-learn**: Provides tools for data preprocessing, model selection, and evaluation metrics, supporting a wide range of machine learning tasks.
- **TensorFlow (including Keras)**: The backbone for building and training advanced neural network models, including GRU and Bidirectional GRU architectures, to handle complex text classification challenges.

#### Data Visualization Libraries:
- **Matplotlib** and **Seaborn**: Integral for creating a wide array of data visualizations, from simple plots to complex heatmaps, to analyze model performance and explore data characteristics.


## Conclusions
- GRU Model achieved an accuracy of 94.10%.
- Bidirectional GRU Model showed an accuracy of 90.60%.
  
Both models demonstrated high accuracy, with the GRU model slightly outperforming the Bidirectional GRU model. However, precision for the positive class (spam) needs improvement, indicating potential areas for future work.

## Project Files
- `quora_spam_prediction.ipynb`: Jupyter notebook containing EDA, data preprocessing steps, the model training and evaluation process.

## Future Work
- Investigate additional model architectures and hyperparameter tuning.
- Explore more advanced techniques for handling class imbalance.

## Contact
Created by [@AnirbanG-git] - feel free to contact me!
