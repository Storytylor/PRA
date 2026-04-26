import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def create_shape(label):
    img = np.zeros((10,10))
    if label == 0:
        for i in range(10):
            for j in range(10):
                if (i-5)**2 + (j-5)**2 < 16:
                    img[i,j] = 1
    elif label == 1:
        img[2:8,2:8] = 1
    elif label == 2:
        for i in range(5):
            img[5+i, 5-i:5+i] = 1
    return img
X = []
y = []
for label in range(3):
    for _ in range(50):
        img = create_shape(label)
        X.append(img.flatten())
        y.append(label)
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(8,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_test[i].reshape(10,10), cmap='gray')
    plt.title(f"A:{y_test[i]} P:{y_pred[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
