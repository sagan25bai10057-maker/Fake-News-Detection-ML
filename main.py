import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def clean_text(text):
    text = text.lower()

   
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<.*?>", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\d+", "", text)

    return text

print("\n \n \nHello and Welcome! I am SARTHAK JINDAL (25BCE10189) presenting you my Machine Learning project to detect fake news..")
print("\n \nHere we start !!!")
print("\n \nLoading dataset...")

fake_news = pd.read_csv("Fake.csv")
real_news = pd.read_csv("True.csv")

fake_news["label"] = 0
real_news["label"] = 1

news_dataset = pd.concat([fake_news, real_news])

news_dataset = news_dataset.sample(frac=1)

print("Cleaning text data...")

news_dataset["text"] = news_dataset["text"].apply(clean_text)

news_texts = news_dataset["text"]
news_labels = news_dataset["label"]


print("Converting text to numbers using TF-IDF...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

news_vectors = vectorizer.fit_transform(news_texts)


x_train, x_test, y_train, y_test = train_test_split(
    news_vectors, news_labels, test_size=0.25
)

print("Training model...")

model = SGDClassifier(max_iter=1000)
model.fit(x_train, y_train)


print("Testing model...")

predictions = model.predict(x_test)

acc = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("\nAccuracy:", round(acc * 100, 2), "%")
print("\nConfusion Matrix:")
print(cm)


def check_news(input_news):

    cleaned = clean_text(input_news)
    vector = vectorizer.transform([cleaned])

    result = model.predict(vector)[0]
    score = model.decision_function(vector)[0]

    if result == 0:
        final = "FAKE NEWS"
    else:
        final = "REAL NEWS"

    return final, score


print("\nYou can now test your own news!")

while True:
    user_text = input("\nEnter news (type exit to stop): ")

    if user_text.lower() == "exit":
        print("Program ended.")
        break

    label, confidence = check_news(user_text)

    print("Result:", label)
    print("Confidence:", round(confidence, 2))

    if abs(confidence) < 0.2:
        print("Note: Model is not very sure about this prediction")