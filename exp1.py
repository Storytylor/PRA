import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , ConfusionMatrixDisplay , classification_report
from sklearn.datasets import load_iris

iris = load_iris()

X = pd.DataFrame(iris.data,columns=iris.feature_names)
y = iris.target

target_names = iris.target_names

scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)

X_train, X_test , y_train, y_test = train_test_split(X_scaler,y,random_state=42,test_size=0.2)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# 7. Feature Importance (Extract Patterns)
importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Pattern Extraction):")
print(importance)

kmeans = KMeans(n_clusters=3,random_state=42)
clusters = kmeans.fit_predict(X_scaler)

X['Cluster'] = clusters
print("\nCluster-wise Mean Patterns:")
print(X.groupby('Cluster').mean())

def predict_species(sample):
    sample_scaled = scaler.transform([sample])
    pred = model.predict(sample_scaled)
    return target_names[pred[0]]
sample = [5.1, 3.5, 1.4, 0.2]
print("\nSample Prediction:", predict_species(sample))
