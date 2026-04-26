
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
df = pd.read_csv("weather.csv")
df = df[['Temperature (C)', 'Humidity']].dropna()

def categorize_temp(temp):
    if temp < 10:
        return 0
    elif temp < 25:
        return 1
    else:
        return 2

def categorize_humidity(h):
    if h < 0.4:
        return 0
    elif h < 0.7:
        return 1
    else:
        return 2

df['Temp_cat'] = df['Temperature (C)'].apply(categorize_temp)
df['Hum_cat'] = df['Humidity'].apply(categorize_humidity)
obs_discrete = df['Temp_cat'] * 3 + df['Hum_cat']
obs_discrete = obs_discrete.values.reshape(-1, 1)
model_discrete = hmm.MultinomialHMM(n_components=3, n_iter=100)
model_discrete.fit(obs_discrete)
hidden_states_discrete = model_discrete.predict(obs_discrete)
X = df[['Temperature (C)', 'Humidity']].values
model_continuous = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
model_continuous.fit(X)
hidden_states_cont = model_continuous.predict(X)
plt.plot(hidden_states_discrete[:100])
plt.plot(hidden_states_cont[:100])
plt.show()
