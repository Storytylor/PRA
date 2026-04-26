import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
train = pd.read_csv("8_mnist_train.csv")
data = train.iloc[:10000]
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X = X / 255.0
gmm = GaussianMixture(n_components=10, covariance_type='diag', random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
plt.figure(figsize=(12,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X.iloc[i].values.reshape(28,28), cmap='gray')
    plt.title(f"Cluster:{labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
