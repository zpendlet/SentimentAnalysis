import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import dask.dataframe as dd
import certifi
import os



# Certificate verification so we can use NLTK
os.environ['SSL_CERT_FILE'] = certifi.where()

# Downloading NLTK stuff
nltk.download("punkt")
nltk.download("stopwords")

print("____________________________________________________________________________________")
print("____________________________________________________________________________________")
print("\n"
      "               Sentiment Analysis on Twitter Data     \n"
      "                   Twitter140 data from Kaggle. \n\n"
      "            This process analyzes tweets for patterns in \n"
      "            language to make assumptions about a statement's\n"
      "            sentiment— either positive or negative.  \n")
print("____________________________________________________________________________________")
print("____________________________________________________________________________________")


# Loading my dataset with Dask
trainingData = dd.read_csv('archive/train_data.csv')
testingData = dd.read_csv('archive/test_data.csv')

def isDataClean(data):
    results = "Data is clean and ready to analyze."
    for column in data:
        if data[column].isnull().any():
            results = f"column '{column}' is empty or invalid."
            break  # Exit the loop once an issue is found

    print(results)

isDataClean(trainingData.compute())
isDataClean(testingData.compute())

print("Preprocessing has begun.\n"
      "Please allow serveral minutes for this part, "
      "as the dataset contains \nover a million records.\n" 
      "Preprocessing now...")
def preprocessWithDask(text):
    # Check if the input is a string or bytes-like object
    if isinstance(text, str):
        tokens = word_tokenize(text)  # Tokenizations
        tokens = [word.lower() for word in tokens]  # Convert to lowercase
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove the stopwords and punctuation

        # Removes unnecessary additions to words ("running" -> "run")
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(word) for word in tokens]

        preprocessed_text = ' '.join(tokens)  # Add back into string after processing

        return preprocessed_text
    else:
        # If the sentence is not a string, return an empty string
        return ""


#apply preprocessing
trainingData["preprocessed_text"] = trainingData['sentence'].map(preprocessWithDask)
testingData["preprocessed_text"] = testingData['sentence'].map(preprocessWithDask)

# Compute Dask DFs to pandas for feature extraction
trainingData = trainingData.compute()
testingData = testingData.compute()

print("Data has been preprocessed")

# Method for extracting features
def extract_features(data, max_features=5000):
    print("Now extracting features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    features = tfidf_vectorizer.fit_transform(data['preprocessed_text'])
    return features, tfidf_vectorizer

train_features, tfidf_vectorizer = extract_features(trainingData)
print("Features have been extracted")

# Model training and deployment
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(train_features, trainingData['sentiment'])

test_features = tfidf_vectorizer.transform(testingData['preprocessed_text'])

predictions = naive_bayes_classifier.predict(test_features)
accuracy = accuracy_score(testingData['sentiment'], predictions)
classification_rep = classification_report(testingData['sentiment'], predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)