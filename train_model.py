from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# Load dataset
data = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv', sep='\t', header=None, names=['label', 'message'])

# Preprocess
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['message']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorizer and model
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Optional accuracy test
X_test_vec = vectorizer.transform(X_test)
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))
