import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('plays_defense_expert1.csv')

X = df.drop(['action'],axis=1).values

df.action = df.action.astype(int)

y = df['action'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

clf = RandomForestClassifier()
model=clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


filename = '/kaggle_simulations/agent/saved_model/defense_model1.sav'
pickle.dump(model, open(filename, 'wb'))
print("Save model successful")
