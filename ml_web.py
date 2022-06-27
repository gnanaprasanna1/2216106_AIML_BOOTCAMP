import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("SUV_Purchase.csv")

# feature engineering
df = df.drop('User ID', axis=1)
df = df.drop('Gender', axis=1)

# loading the data
x = df.iloc[:, :-1].values
y = df.iloc[:, -1:].values

# spliting the data into training and testing data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# standard scaling
from sklearn.preprocessing import StandardScaler

sst = StandardScaler()
x_train = sst.fit_transform(x_train)

# training
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)

# testing
# predicting
y_pred = model.predict(sst.transform(x_test))
print(y_pred)

pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print('sucess loaded ')
