# Spam Email and SMS Detection Using Machine Learning

A machine learning-based web application that classifies email and SMS messages as **SPAM** or **NOT SPAM** using Natural Language Processing (NLP), TF-IDF feature extraction, and a Multinomial Naive Bayes classifier.

## Live Demo

**Streamlit Application:** https://spam-email-sms-detection-kfgxn7fbeqyhdahxs4mbf5.streamlit.app/

## Project Overview

Spam messages frequently contain unwanted advertisements, fraudulent offers, suspicious links, and misleading content. This project uses machine learning to analyze message text and predict whether a message is spam.

The application provides a simple web interface where users can enter an email or SMS message and receive a prediction along with a spam score.

## Features

- Classifies messages as SPAM or NOT SPAM.
- Supports email and SMS text input.
- Uses TF-IDF for text feature extraction.
- Implements a Multinomial Naive Bayes classifier.
- Uses a validation dataset to select a classification threshold.
- Displays the spam score with the prediction.
- Provides model evaluation metrics and a confusion matrix.
- Includes an interactive Streamlit web interface.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Dataset handling and preprocessing |
| scikit-learn | Machine learning and evaluation |
| TF-IDF | Text feature extraction |
| Multinomial Naive Bayes | Text classification |
| Streamlit | Web application interface |
| joblib | Saving and loading the trained model |

## Dataset

The project combines data from the following sources:

1. **UCI SMS Spam Collection Dataset** — SMS messages labeled as spam or legitimate.
2. **SpamAssassin Email Corpus** — email messages used for spam detection.

### Dataset Summary

| Class | Number of Messages |
|---|---:|
| NOT SPAM | 6,976 |
| SPAM | 1,918 |
| **Total** | **8,894** |

The dataset is divided using stratified sampling:

- Training set: 60%
- Validation set: 20%
- Test set: 20%

## Machine Learning Methodology

### 1. Data Preprocessing
The SMS and email records are combined, text data is prepared, and class labels are standardized.

### 2. Feature Extraction
TF-IDF converts message text into numerical features.

Configuration:
- Maximum features: 20,000
- N-gram range: Unigrams and bigrams
- English stop-word removal
- Sublinear term-frequency scaling

### 3. Model Training
A Multinomial Naive Bayes classifier is trained using the TF-IDF feature vectors.

### 4. Threshold Selection
The classification threshold is selected using the validation dataset. The final selected threshold is **0.10**, targeting spam recall of at least 0.90.

### 5. Model Evaluation
The final model is evaluated on a separate held-out test dataset.

## Model Performance

The model achieved the following results on the test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 92.36% |
| Spam Precision | 0.77 |
| Spam Recall | 0.93 |
| Spam F1-Score | 0.84 |

### Confusion Matrix

| Actual Class | Predicted NOT SPAM | Predicted SPAM |
|---|---:|---:|
| NOT SPAM | 1,288 | 108 |
| SPAM | 28 | 355 |

The model identifies a high proportion of spam messages, with a spam recall of 0.93. However, some legitimate messages may also be classified as spam.

## Project Structure

```text
Spam_Detection_Project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── spam_model.joblib
│   └── evaluation.json
│
└── data/
    └── dataset files (if included)
```

*Note: Adjust the structure above to match the files and folders actually present in your repository.*

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Git

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Spam_Detection_Project
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Usage

1. Open the Streamlit application.
2. Enter an email or SMS message in the text input area.
3. Click the prediction button.
4. View the predicted class: SPAM or NOT SPAM.
5. Review the displayed spam score.
6. Open the model evaluation section to examine performance metrics, where available.

## Limitations

- Predictions depend on the quality and characteristics of the training data.
- Unseen words, new spam techniques, and misleading messages may affect accuracy.
- Some legitimate messages may be incorrectly flagged as spam.
- A spam score is a model-derived score and should not automatically be interpreted as a perfectly calibrated probability.
- The model should be retrained periodically to adapt to changing spam patterns.

## Future Enhancements

- Compare additional machine learning classifiers.
- Improve preprocessing and feature engineering.
- Add support for additional languages.
- Evaluate the model on newer spam datasets.
- Improve probability calibration and threshold selection.
- Add more detailed evaluation visualizations.

## References

- UCI Machine Learning Repository — SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- SpamAssassin Public Corpus: https://spamassassin.apache.org/old/publiccorpus/
- scikit-learn Documentation: https://scikit-learn.org/
- Streamlit Documentation: https://docs.streamlit.io/
- pandas Documentation: https://pandas.pydata.org/docs/

## Academic Project

**Project Title:** Spam Email and SMS Detection Using Machine Learning  
**Project Type:** Data Science Mini Project  
**Application Framework:** Streamlit

---

If you find this project useful, feel free to explore the code and adapt it for educational purposes.
