# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 08:56:14 2018
Created on Wed Oct 24 19:35:39 2018

@author: Vinay M HARITSA
@ vtricks technologies (gurukulam vidya)
@ for any queries contact : 9620749749


"""

#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, Y)

#Predict Output 
predicted= model.predict([[1,2],[3,4]])
print(" output is ")
print(predicted)