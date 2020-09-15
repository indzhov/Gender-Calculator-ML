import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

df = pd.read_csv("responses.csv")

df = df.dropna()

gender = df.iloc[:,[144]]

indicator_variable = {"male" : 1, "female" : 0}
gender.replace(indicator_variable, inplace=True)

movies = df.iloc[:,20:31]
scared    = df.iloc[:,63:73]
interests = df.iloc[:,31:63]
spending  = df.iloc[:,134:140]

movies = movies.join([scared, interests, spending])

X = np.array(movies)
y = np.array(gender)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sv = SVC(kernel='linear').fit(X_train,y_train)


pickle.dump(sv, open('gender.pkl', 'wb'))