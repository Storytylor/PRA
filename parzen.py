from PIL import Image
import numpy as np
import os
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score

data, labels = [], []
shapes = ["circle", "square", "oval", "rectangle", "overlapped", "star"]

for label, shape in enumerate(shapes):
    for file in os.listdir(f"shapes/{shape}"):
        img = Image.open(f"shapes/{shape}/{file}").convert("L")
        img = img.resize((32, 32))
        data.append(np.array(img).flatten())
        labels.append(label)

X = np.array(data) / 255.0
y = np.array(labels)
models = {}
for c in np.unique(y):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(X[y == c])
    models[c] = kde
y_pred = []
for x in X:
    scores = [models[c].score_samples([x])[0] for c in models]
    y_pred.append(np.argmax(scores))

y_pred = np.array(y_pred)
print(accuracy_score(y, y_pred))
plt.figure(figsize=(8,6))
for i in range(min(6, len(X))):
    plt.subplot(2, 3, i+1)
    plt.imshow(X[i].reshape(32,32), cmap='gray')
    plt.title(f"Pred: {shapes[y_pred[i]]}")
    plt.axis('off')
plt.suptitle("Parzen Window (No Split)")
plt.show()
