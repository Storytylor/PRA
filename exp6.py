import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
output = pd.read_csv("output.csv")
patients = pd.read_csv("patients.csv")
output = output[["subject_id", "value"]]
patients = patients[["subject_id", "gender"]]
output["value"] = pd.to_numeric(output["value"], errors="coerce")
output = output.dropna()
data = pd.merge(output, patients, on="subject_id")
data["gender"] = data["gender"].map({"M": 0, "F": 1})
mean_val = data["value"].mean()
data["target"] = data["value"].apply(lambda x: 1 if x > mean_val else 0)
X = data[["value", "gender"]]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print("\nSAMPLE PREDICTIONS:")
for i in range(5):
    pred = model.predict(X_test.iloc[[i]])[0]
    print(f"Actual: {y_test.iloc[i]} | Predicted: {pred}")
