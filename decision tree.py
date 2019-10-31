# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 00:45:36 2019

@author: TEJ
"""

                         DECISION TREE------------
                         
- it is a graphical representation of all the possible solutions to a decision based on certain condition.
- root node is base node of the tree, the tree starts with root node(root is entire populaton)                           
- It is a another algorithm and unlike logistic regression which is only use for binarry classification.
- it is used in classification as well as regression, evethough it is more popular in cassification.(he is a boy or girl like that)
- it is used to classify the multiple classification not evev though binary classification.
- Decision tree is a tree were each node represents a feacture(attribute), each link(branch) represents a decision
  and each leaf represents a outcome.(categorical or continous value)

  IMPLEMENTATION OF DECISION TREE 
- load the libraries
- import the dataset, extract independent and dependent variable                           
- visualize the data
- split the data into train data and test data
- train a decision tree
- predict the model
- evaluate the model

import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline   

kyphosis=pd.read_csv('kyphosis.csv')

x=kyphosis.drop('Kyphosis',axis=1) # independent variable
y=kyphosis['Kyphosis'] # dependent variable
print(x)
print(y)

# data analysis
kyphosis.groupby(['Kyphosis']).size()
kyphosis.groupby(['Kyphosis']).size().plot(kind='bar')
sns.barplot(x='Kyphosis',y='Age',data=kyphosis)
sns.pairplot(kyphosis,hue='Kyphosis',palette='Set1') # set1 S is capital

# visualising dataset
plt.figure(figsize=(25,7))
sns.countplot(x='Age',hue='Kyphosis',data=kyphosis,palette='Set1') # red=kyphosis absent and blue=kyphosis present

# split the data into train data and test data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=100)


# training decision tree
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)

# predicting the model
predictions=dtree.predict(x_test)
print(predictions)

# Evaluation of the model
from sklearn.metrics import classification_report,confusion_matrix 
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)












################################################

                    RANDOM FOREST
- it is a ensemble learning algorithm.
- it is work based on vote or mejority. 
  ex- out of 100 trees, 90 trees predict 1 and 10 trees predict 0  mejority should be 1
      output od random forest is 1        
                    
- it buids multiple decision tree and merges them together.
- more accurate and stable prediction.                    
- random forest is very similar to the decision tree, instead of using one tree we used number of trees thats why its called a forest.                    
- here we take the average or voting of the all trees, for each observation.
- it is used for improving the accuracy

                    
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100) # forest consist of 100 trees
rfc.fit(x_train,y_train)                    

rfc_pred=rfc.predict(x_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred)) 

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)   

###################################################################


         IDENTIFY HAND WRITTEN DIGITS RECOGNITION BT RANDOM FOREST
                  (0,1,2,3,4,5,6,7,8,9)

import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()
digits

digits.target
digits.target_names
digits.images

dir(digits) # properties of the datasets

%matplotlib inline
import matplotlib.pyplot as plt
plt.gray()
for i in range(12):
    plt.matshow(digits.images[i]) # visualize my data
# o/p is hand written characters, which is 8 by 8 pixel array and multidimensional array.    
    
digits.data[:5]
# o/p is 1dimensional array, which is 8 by 8 pixel and 64 elements(length of the array) 
 
df=pd.DataFrame(digits.data)
df
pd.set_option('display.max_columns',64)
df.head()

df['target']=pd.DataFrame(digits.target)
df.head()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df.drop(['target'],axis='columns'),df.target,test_size=0.2)

len(X_train)
# 1437

len(X_test)
# 360

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()

model.fit(X_train,Y_train)

model.score(X_test,Y_test)
# 0.9472222222222222


Y_pred=model.predict(X_test)
Y_pred

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test,Y_pred)
score
# 0.9472222222222222s

# tune model ie,

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
# 0.9722222222222222

from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(Y_test,Y_pred)
matrix
# this matrix is visualized by heatmap

import seaborn as sns
sns.heatmap(matrix,annot=True) 
plt.xlabel('predicted')
plt.ylabel('truth')
# from the heatmap,
 37 means- 37 times truth is zero and 37 times model predicted as zero
 2 means - 2 times truth value is 8 and 2 times model predicted as one









#######################################################

                            K-NEAREST NEIGHBOURS(KNN)
                            
- it is a classification algorithm generally used to predict categorical values.  
- it stores available cases and classifies new casses based on similarity measure.   
- K=number of nearest neighbours.                        

it consist following steps:
    -handle data (import and all)
    -similarity(distance between two data)
    -neighbors(locate K most similar data instance)
    -response(generate a response from a set of data instance)
    -accuracy()

TAKE ANOTHER ONE EXAMPLE ie. XUV CSV

import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
%matplotlib inline        

xuv=pd.read_csv('xuv.csv')

x=xuv.iloc[:,[2,3]].values
y=xuv.iloc[:,4].values  # iloc is used for selection by index
print(x)
print(y)   ## here 0 stands for did't purchased and 1 stands for purchased

x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.25,random_state=0) 
  # divided train and test plate size as 75(train data) & 25(test data) ratio 

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test) 

from sklearn.neighbors import KNeighborsClassifier
Classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
Classifier.fit(x_train,y_train)

y_pred=Classifier.predict(x_test)
print(y_pred) # here we get the zero's one"s

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred) 

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

accuracy_score(y_test,y_pred)*100 




# visualising training set results
# train set
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1, x2=np.meshgrid(np.arange(start=x_set[:,0].min() -1, stop=x_set[:,0].max() + 1,step=0.01),
                  np.arange(start=x_set[:,1].min() -1, stop=x_set[:,0].max() + 1,step=0.01))
plt.contourf(x1,x2,Classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
    
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
    
plt.title('K-NN(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()    



# test set
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1, x2=np.meshgrid(np.arange(start=x_set[:,0].min() -1, stop=x_set[:,0].max() + 1,step=0.01),
                  np.arange(start=x_set[:,1].min() -1, stop=x_set[:,0].max() + 1,step=0.01))
plt.contourf(x1,x2,Classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
    
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
    
plt.title('K-NN(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()    

########################################################################

                            KNN USING IRIS DATA SETS
                            
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from pandas.plotting import scatter_matrix

iris = sns.load_dataset("iris")
print(iris.head())
print (iris.shape)
print(iris.describe())  

import csv
import random

def loadCsv(filename):
    lines=csv.reader(open(r'E:\python\csv files\Iris.csv'))
    Dataset=list(lines)
    for row in lines:   
        print (', '.join(row))
        
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'r') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])
            if random.random()<split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
                
trainingSet=[]
testSet=[]
loadDataset(r'Iris.',0.66,trainingSet,testSet)
print('Train:'+repr(len(trainingSet)))
print('Test:'+repr(len(testSet)))                
            
        
        
        


with open(r'E:\python\csv files\Iris.data.csv') as csvfile:    
    lines=csv.reader(csvfile)
    for row in lines:   
        print (', '.join(row))                         




