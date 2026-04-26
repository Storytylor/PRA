import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB

df = pd.read_csv("C://Users//ROHAN//Downloads//Datasets Pra//Datasets Pra//asia_bayesian_dataset_grpA_7.csv")

X = df.drop("LungCancer", axis=1)
y = df["LungCancer"]

model = BernoulliNB()
model.fit(X, y)

sample = [[1,1,0,1,0,1,1]]
prob = model.predict_proba(sample)

print("P(No Lung Cancer):", prob[0][0])
print("P(Yes Lung Cancer):", prob[0][1])
print("Predicted Lung Cancer:", model.predict(sample)[0])

plt.hist(y, edgecolor="black")
plt.title("Lung Cancer Cases")
plt.xlabel("Lung Cancer")
plt.ylabel("Count")
plt.show()
