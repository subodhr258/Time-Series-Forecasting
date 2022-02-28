import joblib
import numpy as np
import matplotlib.pyplot as plt
model = joblib.load('rf_model_200.h5')


import pandas as pd
sequences = pd.read_csv('Weather_data_Sequences_200.csv')

sequences.head()

len(sequences)

n_predict = 200

X = sequences.iloc[:,:-1]
len(X)

for i in range(0,n_predict):
    prediction = round(model.predict([X.iloc[-1,:]])[0],1)
    X = X.append(X.iloc[-1,:].shift(-1).replace(np.NaN,prediction))
    print(X.iloc[-1,:])
X.reset_index(drop=True,inplace=True)
X.tail(n_predict)

plt.plot(X.iloc[-n_predict*2:-n_predict,-1])
plt.plot(X.iloc[-n_predict:,-1])
plt.show()


