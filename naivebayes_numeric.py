# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 08:33:31 2018
Created on Wed Oct 24 19:35:39 2018

@author: Vinay M HARITSA
@ vtricks technologies (gurukulam vidya)
@ for any queries contact : 9620749749

@author: Vinay
"""

import numpy as np

import pandas as pd

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

play_tennis = pd.read_csv("PlayTennis.csv")

play_tennis.head()

#A LabelEncoder converts a categorical data into a number ranging from 0 to n-1, where n is the number of classes in the variable.

#For example, in case of Outlook, there are 3 clasess – Overcast, Rain, Sunny. These are represented as 0,1,2 in alphabetical order.


number = LabelEncoder()

play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])

play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])

play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])

play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])

play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])



#Let’s define the features and the target variables.


features = ["Outlook", "Temperature", "Humidity", "Wind"]

target = "Play Tennis"

#To validate the performance of our model, we create a train, test split. We build the model using the train dataset and we will validate the model on the test dataset.

features_train, features_test, target_train, target_test = train_test_split(play_tennis[features],
play_tennis[target],test_size = 0.33,random_state = 54)


#Let’s create the model now.


model = GaussianNB()

model.fit(features_train, target_train)

#Accuracy score measure the number of right predictions.

pred = model.predict(features_test)

accuracy = accuracy_score(target_test, pred)


#test
print (model.predict([[1,2,0,1]]))


