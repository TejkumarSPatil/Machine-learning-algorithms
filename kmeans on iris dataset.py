# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 22:55:26 2019

@author: TEJ
"""

- K Means clustering is an unsupervised machine learning algorithm. 
  An example of a supervised learning algorithm can be seen when looking at Neural Networks
  where the learning process involved both the inputs (x) and the outputs (y). 
  During the learning process the error between the predicted outcome (predY) and 
  actual outcome (y) is used to train the system. In an unsupervised method such as 
  K Means clustering the outcome (y) variable is not used in the training process.
  
  
  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt ##  for scattered plot
%matplotlib inline
from sklearn import datasets  # imp
from sklearn.cluster import KMeans  ####  run for algorithm
from statistics import mode  
import seaborn as sns      

iris=datasets.load_iris()
print(iris)

x=pd.DataFrame(iris.data)
x

x.head()

x.columns=['sepal_length','sepal_width','petal_length','petal_width']
x

x.columns

pd.set_option('display.max_columns',4)
x

dir(iris)
# ['DESCR', 'columns', 'data', 'feature_names', 'target', 'target_names']
# DESCR means description

x.describe()

x.sum()

x.isnull().sum()  # no NaN values

sns.heatmap(x.isnull(),yticklabels=False,cmap='viridis')

x['sepal_length'].mode()

from scipy import stats
stats.mode(x['sepal_length'])


x['sepal_width'].max()

x['sepal_width'].min()

x['sepal_width'].var()

x['sepal_width'].std()


x[x.sepal_length == 4.6].shape[0]

x.boxplot(column='sepal_length')

x.hist(column='sepal_length')

x.hist(column='sepal_length',bins=30)

x[20:30]

x.sepal_width.apply(lambda n:n*2)

x.loc[4:30,'sepal_width']

x.loc[[2,6,3,8],['sepal_length','sepal_width','petal_length']]

x.loc[[2,6,3,8],['sepal_length','sepal_width']]

x.loc[[2,6,3,8],['sepal_length']]

##############################################################

iris.target

iris.target_names

x1=pd.DataFrame(iris.target) 
x1 

concatn=pd.concat([x,x1],axis=1)
concatn   

concatn.columns=['sepal_length','sepal_width','petal_length','petal_width','target']
concatn.columns
concatn

#########

concatn[concatn.target == 0]

concatn[concatn.target == 0].shape[0]

concatn[concatn.target == 1]

concatn[concatn.target == 1].shape[0]

concatn[concatn.target == 2]

concatn[concatn.target == 2].shape[0]

concatn

concatn.corr()

concatn['flower_names']=concatn.target.apply(lambda g: iris.target_names[g])
concatn  #  g is used instead of x

################################################################
# A scatter plot , is a type of plot or mathematical diagram using Cartesian coordinates to display values for 
#  typically two variables for a set of data.

sns.boxplot(data=x) # starting sns contains only strip swarm and count plot

concatn.head(10)

sns.swarmplot('sepal_length',data=concatn)

sns.stripplot('sepal_length',data=concatn)

sns.stripplot(x='sepal_length',y='sepal_width',data=concatn)

sns.stripplot(x ='flower_names', y = 'petal_length',data=concatn,jitter=True)


sns.swarmplot(x='sepal_length',y='sepal_width',data=concatn)

sns.jointplot(x = 'petal_length',y = 'petal_width',data = concatn)
plt.show()

sns.countplot('sepal_length',data=concatn)

# Hexbin Plot
- Hexagonal binning is used in bivariate data analysis when the data is sparse in density i.e., when the data is 
  very scattered and difficult to analyze through scatterplots.
- An addition parameter called ‘kind’ and value ‘hex’ plots the hexbin plot.

sns.jointplot(x = 'petal_length',y = 'petal_width',data =concatn,kind = 'hex')
plt.show()

# Kernel Density Estimation
sns.jointplot(x = 'petal_length',y = 'petal_width',data =concatn,kind = 'kde')
plt.show()


# Seaborn - Histogram
import seaborn as sns
sns.distplot(concatn['petal_length'],kde = False) # Kernel Density Estimation
plt.show()
# Kernel Density Estimation (KDE) is a way to estimate the probability density function of a continuous random variable.
# It is used for non-parametric analysis.

# Seaborn - Kernel Density Estimates
sns.distplot(concatn['petal_length'],hist=False)
plt.show()


# Fitting Parametric Distribution
sns.distplot(concatn['petal_length'])
plt.show()


sns.set_style('ticks')
sns.pairplot(concatn,hue = 'flower_names',diag_kind = 'kde',kind = 'scatter',palette = 'husl')
plt.show()
#######################  very very imp



###################################################################
                                  VISUALIZATION
                                  
df0=concatn[concatn.target==0] 
df0

df1=concatn[concatn.target==1]
df1
                                  
df2=concatn[concatn.target==2]                                  
df2   

plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.scatter(df0['sepal_length'],df0['sepal_width'],color='red',marker='*')  
plt.scatter(df1['sepal_length'],df1['sepal_width'],color='blue',marker='+') 
# svm perform well  


                       
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.scatter(df0['petal_length'],df0['petal_width'],color='red',marker='*')  
plt.scatter(df1['petal_length'],df1['petal_width'],color='blue',marker='+') 
# Svm perform well

#####################################################################
               creating model

feature_names=['sepal_length','sepal_width','petal_length','petal_width']
X=concatn[feature_names]
Y=concatn.target

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

len(X_train)
len(X_test)

from sklearn.svm import SVC
model=SVC()

model.fit(X_train,Y_train)

model.score(X_test,Y_test)  # accuracy score
# o/p is 1.0

#########################################
# if we increase the C(regularization)
model=SVC(C=10)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
# o/p is 0.90000

############################################
# do with gamma
model=SVC(gamma=10)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
# o/p is 0.9666666

#########################################
# do with kernel 
model=SVC(kernel='linear')
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

# different kernels are 'rbf' 'linear' 'poly' 'sigmoid' 'precomputed' 


##########################################################################
                                  LINEAR  REGRESSION



######################

#  sepal_length
feature_names=['sepal_length']
X=concatn[feature_names]
Y=concatn.target 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)  ## fit method is applies all algorithm.

Y_pred=lr.predict(X_test)
print(Y_pred) # here we not get the zeros and ones

plt.scatter(X_train,Y_train,color='blue',label='Scatter Plot')
plt.plot(X_train,lr.predict(X_train),color='red',label='Regression Line')
plt.title('iris')
plt.xlabel('sepal_length')
plt.ylabel('target')
plt.legend()
plt.show() # like that do other variables

#########################

#  sepal_width
feature_names=['sepal_width']
X=concatn[feature_names]
Y=concatn.target 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)  ## fit method is applies all algorithm.

Y_pred=lr.predict(X_test)
print(Y_pred) # here we not get the zeros and ones

plt.scatter(X_train,Y_train,color='blue',label='Scatter Plot')
plt.plot(X_train,lr.predict(X_train),color='red',label='Regression Line')
plt.title('iris')
plt.xlabel('sepal_width')
plt.ylabel('target')
plt.legend()
plt.show() # like that do other variables

###############################


# petal_length
feature_names=['petal_length']
X=concatn[feature_names]
Y=concatn.target 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)  ## fit method is applies all algorithm.

Y_pred=lr.predict(X_test)
print(Y_pred) # here we not get the zeros and ones

plt.scatter(X_train,Y_train,color='blue',label='Scatter Plot')
plt.plot(X_train,lr.predict(X_train),color='red',label='Regression Line')
plt.title('iris')
plt.xlabel('petal_length')
plt.ylabel('target')
plt.legend()
plt.show() # like that do other variables

################################

# petal_width
feature_names=['petal_width']
X=concatn[feature_names]
Y=concatn.target 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)  ## fit method is applies all algorithm.

Y_pred=lr.predict(X_test)
print(Y_pred) # here we not get the zeros and ones

plt.scatter(X_train,Y_train,color='blue',label='Scatter Plot')
plt.plot(X_train,lr.predict(X_train),color='red',label='Regression Line')
plt.title('iris')
plt.xlabel('petal_width')
plt.ylabel('target')
plt.legend()
plt.show() # like that do other variables

############################################################################
   
a=KMeans(n_clusters=3)
a.fit(concatn)

a.labels_

colormap=np.array(['Red','Blue','Green'])
z=plt.scatter(concatn.sepal_length,concatn.sepal_width,concatn.petal_length,c=colormap[a.labels_])

from sklearn.metrics import accuracy_score
accuracy_score(iris.target,a.labels_)





#######################################################

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
 
# Only needed if you want to display your plots inline if using Notebook
# change inline to auto if you have Spyder installed
%matplotlib inline


iris = datasets.load_iris()


iris.data
iris.feature_names
iris.target
iris.target_names

# Store the inputs as a Pandas Dataframe and set the column names
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
 
y = pd.DataFrame(iris.target)
y.columns = ['Targets']


#  Visualise the data

- We will do this by plotting two scatter plots. 
  One looking at the Sepal values and another looking at Petal.

# Set the size of the plot
plt.figure(figsize=(14,7))
 
# Create a colormap
colormap = np.array(['red', 'lime', 'black'])
 
# Plot Sepal
plt.subplot(1, 2, 1)
plt.scatter(x.Sepal_Length, x.Sepal_Width, c=colormap[y.Targets], s=40)
plt.title('Sepal')
 
plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Petal')


####################################
# Build the K Means Model

model = KMeans(n_clusters=3)
model.fit(x)


#  Next we can view the results. This is the classes that the model decided, 
#  remember this is unsupervised and classified these purely based on the data.
model.labels_



- Visualise the classifier results
   Lets plot the actual classes against the predicted classes from the K Means model.

# View the results
# Set the size of the plot
plt.figure(figsize=(14,7))
 # Create a colormap
colormap = np.array(['red', 'lime', 'black'])
 # Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')
 
# Plot the Models Classifications
plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')

-  Using this code below we using the np.choose() to assign new values, basically we are 
   changing the 1’s in the predicted values to 0’s and the 0’s to 1’s. Class 2 matched so 
   we can leave. 
-  By running the two print functions you can see that all we have done is swap the values.


#################################################


# The fix, we convert all the 1s to 0s and 0s to 1s.
predY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
print (model.labels_)
print (predY)
# View the results
# Set the size of the plot
plt.figure(figsize=(14,7)) 
# Create a colormap
colormap = np.array(['red', 'lime', 'black'])
 # Plot Orginal
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')
 # Plot Predicted with corrected values
plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[predY], s=40)
plt.title('K Mean Classification')




# Performance Metrics
sm.accuracy_score(y, predY)




# Confusion Matrix
sm.confusion_matrix(y, predY)

#############################################################################

                                SVM
                                














