import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load the raw dataset
# --------------------------------------------------

df = pd.read_csv("spam.csv", encoding="latin-1")

print("\nFirst few rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nDataset info:")
print(df.info())


# --------------------------------------------------
# Keep only useful columns
# --------------------------------------------------

df = df[["v1", "v2"]]

df.columns = ["label", "message"]


# --------------------------------------------------
# Basic data checks
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# remove duplicate messages
df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)


# --------------------------------------------------
# Convert labels to numbers
# --------------------------------------------------

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nClass distribution:")
print(df["label"].value_counts())


# --------------------------------------------------
# Convert text into numerical features
# --------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["message"])

y = df["label"]


# --------------------------------------------------
# Train / Test split
# --------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# Train the model
# --------------------------------------------------

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)


# --------------------------------------------------
# Evaluate the model
# --------------------------------------------------

from sklearn.metrics import accuracy_score, classification_report

predictions = model.predict(X_test)

print("\nModel Performance")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# --------------------------------------------------
# Test with a custom message
# --------------------------------------------------

test_message = ["Congratulations! You have won a free prize"]

vector = vectorizer.transform(test_message)

prediction = model.predict(vector)

if prediction[0] == 1:
    print("\nTest message prediction: Spam")
else:
    print("\nTest message prediction: Not Spam")


# --------------------------------------------------
# Find most important spam words
# --------------------------------------------------

feature_names = vectorizer.get_feature_names_out()

spam_word_scores = model.feature_log_prob_[1]

word_importance = pd.DataFrame({
    "word": feature_names,
    "score": spam_word_scores
})

word_importance = word_importance.sort_values(
    by="score",
    ascending=False
)

print("\nTop spam words:")
print(word_importance.head(20))


# --------------------------------------------------
# Visualize most important spam words
# --------------------------------------------------

top_words = word_importance.head(10)

plt.figure(figsize=(10,5))

plt.bar(top_words["word"], top_words["score"])

plt.title("Top Words Associated with Spam")

plt.xlabel("Words")

plt.ylabel("Spam Score")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()