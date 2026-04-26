import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
print("First 5 rows:\n")
print(df.head())
print("\nDataset Info:\n")
print(df.info())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\n--- Sample Predictions ---")
for i in range(5):
    print("\nActual:", y_test.iloc[i])
    print("Predicted:", y_pred[i])
sns.histplot(df['alcohol'], kde=True)
plt.title("Histogram with KDE Curve (Alcohol Feature)")
plt.xlabel("Alcohol")
plt.ylabel("Density")
plt.show()
