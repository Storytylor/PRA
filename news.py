import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report

data = fetch_20newsgroups(subset='all',remove=('headers','footers','quotes'))

X = data.data
y = data.target

print("Sample document:\n", X[0][:500])

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Feature Extraction (Pattern Recognition)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Prediction
y_pred = model.predict(X_test_tfidf)

# 6. Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 7. Feature importance (Top words per class)
feature_names = np.array(vectorizer.get_feature_names_out())

for i, category in enumerate(data.target_names[:5]):  # show for first 5 classes
    top10 = np.argsort(model.feature_log_prob_[i])[-10:]
    print(f"\nTop words for class '{category}':")
    print(feature_names[top10])
