from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
data = fetch_kddcup99(subset='SA')
X = data.data
y = data.target
X = X.astype(str)
for i in range(X.shape[1]):
    le = LabelEncoder()
    X[:, i] = le.fit_transform(X[:, i])
X = X.astype(float)
model = IsolationForest(contamination=0.1)
model.fit(X)
y_pred = model.predict(X)
print("Normal:", np.sum(y_pred == 1))
print("Anomalies:", np.sum(y_pred == -1))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Anomaly Detection on KDD Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
