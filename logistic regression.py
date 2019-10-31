# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:00:15 2019

@author: TEJ
"""

                    REGRESSION
- regression analysis is a predictive modelling technique.                    
- which investigates the relation between dependent(target) and independent variable(predictor)
                    
          three types:  linear, logistic and polynomial.
          
                   LINEAR REGRESIION
                   
- range is -infinity to infinity.                   
- Francis Galton is a scientist found linear regression.                   
- It is a one of the algorithm in machine learnig.
- regression analisis is a form of predictive modelling technique.
- which investigates the relation between dependent(we wnt to predict) or target variable and independent variable.
- regression explains the, changes dependant variable in y-axis and changes expliratory variable in x-axis.
- it involves graphing the line over a set of data points. it mostly feds the overall shape of the data.
- the distace between all point and stright line must be minimum.

THREE major usses in regression analysis---
- determine the strength of the predictors.(which explains the strength of the independent variable over the dependant variable.)
  ex-what strength between sales and marketing aspects.

- Forecasting an effect.
  ex- how much dependant variable changes, the change in one or more independat variable.

- Trend Forecasting.
  ex- what is the rate of bitcoin in next six months.   

WHERE IS LINEAR REGRESSION USED?????
- evaulating the trends and sales estimates.

- impact of price changes.

- risk in financial services and insurance domain.

- for better understanding refer the notess.

EX:
    # import wine excel
import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
%matplotlib inline           

# collecting x and y

import pandas as pd
df = pd.read_csv("D:\DS\ds\python\csv files\\wine.csv")
df.head()
x=df['Alcohol'].values
y=df['Mg'].values



# taking the maen of x and y       y=mx+c
x_mean=df['Alcohol'].mean()
y_mean=df['Mg'].mean()

# eq of regression line is, y=mx+c

n=len(x)

numr=0
dnom=0

for i in range(n):
    numr += (x[i]-x_mean) * (y[i]-y_mean) # + is the summation
    dnom += (x[i]-x_mean)**2
m=numr/dnom    
c=y_mean-(m*x_mean)
print(m,c)       # y=mx+c    here we get m and c

# after calculating m and c, we predict the value of Y

Y_pred=(m*x) + c  # here Y predicted value

plt.plot(x,Y_pred,color='red',label='Regression Line') # execute separatly

plt.scatter(x,y,color='blue',label='Scatter Plot')

plt.xlabel('Alcohol')
plt.ylabel('Mg')
plt.legend()
plt.show()

# lets calculate the R_sqare value
n1 = 0
d1 = 0
for i in range(n):
    Y_pred=(m*x[i]) + c
    n1 += (y[i] - y_mean)**2   # numarator 1
    d1 += (y[i] - Y_pred)**2   # denominator 1

R_square =  (d1 / n1)
print(R_square) 
########################################################


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.5,random_state=0) 

# Ordinary Least Squares
def ordinary_least_squares(x, y):
    x_mean=df['Alcohol'].mean()
    y_mean=df['Mg'].mean()
    
    # eq of regression line is, y=mx+c
    
    n=len(x)
    
    numr=0
    dnom=0
    
    for i in range(n):
        numr += (x[i]-x_mean) * (y[i]-y_mean) # + is the summation
        dnom += (x[i]-x_mean)**2
    m=numr/dnom    
    c=y_mean-(m*x_mean)
    b = c[0]
    w = m[0]
    
    return lambda x: w*x + b


hypothesis = ordinary_least_squares(x_train, y_train)
hypothesis


























  APPLYING LINEAR REGRESSION TO THE XUV CAR
  
###########################################################  
os.chdir('E:\\python\\csv files')  

xuv=pd.read_csv('xuv.csv')

ge=pd.get_dummies(xuv['Gender'])
ge

ge=pd.get_dummies(xuv['Gender'],drop_first=True)
ge

xuv1=pd.concat([ge,xuv],axis=1)
xuv1

xuv1.drop(['Gender'],axis=1,inplace=True)
xuv1

xuv1.drop(['User_id'],axis=1,inplace=True)
xuv1

feacture_names=['Age','Estimates Salary','M']
x=xuv1[feacture_names] 
y=xuv1.Purchased


###############################################################
 
x=xuv.iloc[:,-3:-2].values
y=xuv.iloc[:,3].values  # iloc is used for selection by index
print(x)
print(y)   ## here 0 stands for did't purchased and 1 stands for purchased

 # We build the model using the train dataset and we will validate the model on the test dataset.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.3,random_state=0) 
  # divided train and test plate size as 75 & 25 ratio



(Training Set: Here, you have the complete training dataset. You can extract features 
      and train to fit a model and so on.

Validation Set: This is crucial to choose the right parameters for your estimator. 
                We can divide the training set into train set and validation set. 
                Based on the validation test results, the model can be trained(for instance, changing parameters, classifiers). 
                This will help us get the most optimized model.

Testing Set: Here, once the model is obtained, you can predict using the model obtained on the training set.)
   
 (  It is not essential that 70% of the data has to be for training and rest for testing.)
  
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)  ## fit method is applies all algorithm.

y_pred=lr.predict(x_test)
print(y_pred) # here we not get the zeros and ones

plt.scatter(x_train,y_train,color='blue',label='Scatter Plot')
plt.plot(x_train,lr.predict(x_train),color='red',label='Regression Line')
plt.title('xuv')
plt.xlabel('Age')
plt.ylabel('Estimates Salary')
plt.legend()
plt.show()





####################################################################


# below is the calculation of xuv
x1=xuv['Age'].values
y1=xuv['Estimates Salary'].values

# taking the maen of x and y       y=mx+c
x1_mean=xuv['Age'].mean()
y1_mean=xuv['Estimates Salary'].mean()
print(x1_mean)
print(y1_mean)

# eq of regression line is, y=mx+c

n=len(x1)

numr=0
dnom=0

for i in range(n):
    numr += (x1[i]-x1_mean) * (y1[i]-y1_mean) # + is the summation
    dnom += (x1[i]-x1_mean)**2
m=numr/dnom    
c=y1_mean-(m*x1_mean)
print(m,c)       # y=mx+c    here we get m and c

# after calculating m and c, we predict the value of Y

Y_pred=(m*x) + c  # here Y predicted value

plt.plot(x1,Y_pred,color='red',label='Regression Line') # execute separatly
plt.scatter(x1,y1,color='blue',label='Scatter Plot')
plt.title('xuv')
plt.xlabel('Age')
plt.ylabel('Estimates Salary')
plt.legend()
plt.show()  # it is a negaive linear regression

# lets calculate the R_sqare value
n1 = 0
d1 = 0
for i in range(n):
    Y_pred=(m*x1[i]) + c
    n1 += (y1[i] - y1_mean)**2   # numarator 1
    d1 += (y1[i] - Y_pred)**2   # denominator 1

R_square =  (d1 / n1)
print(R_square)   

 #########################################################
 
 
 
 
 
 

                      LOGISTIC REGRESSION
                      
- range of sigmoid curve is  0 to 1 or viceversa.                      
- it produces result in a binary fomate.
- it is used to predict a outcome categorical dependant variable.
- it is used in classification problems.ex: (it's bird or not) (the patient ill or not)
  ex: 0 or 1, yes or no, true or false, high and low (otcome always descrete or categorical in nature)
        ie, 0 or 1

applications: weather prediction(raining or not,it is sunny or not)  

IMPLEMENTING LINEAR AND LOGISTIC REGRESSION: 5 STEPS, 
- collecting data(import libraries)
- analysing the data
- data wrangling.(cleaning the data ie,null values,repeating the data)
- train and test.(here we build our model)
- accuracy check. 
###################################################              
      
                     COLLECTING THE DATA                 
import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
%matplotlib inline     

import titavic csv file   


import pandas as pd
df2 = pd.read_csv(r"C:\Users\tejkumar.p\Desktop\python\csv files\titanic csv\train.csv")
df2.head() 
 
# df=pd.read_csv('titanic.csv')    
# df.head(10)    we get first 10 rows
          
# print('total passengers : '+str(len(df.index)))    finding total passengers
df['Name'].count() 


##################################################################################################

                      ANALYSING THE DATA 
# who survived and who are not survived
df.info()                      
df.groupby(['Survived']).size()
df.groupby(['Survived']).size().plot(kind='bar')   # 0 stands for didt survive and 1 stands for survive

df.groupby(['Sex']).size()


sns.countplot(x='Survived',hue='Sex',data=df) # barplot# it shows the graphof how many male are survived and females are survived
df.groupby(['Sex','Survived']).size()


sns.countplot(x='Survived',hue='Pclass',data=df)

sns.countplot(x='Pclass',hue='Sex',data=df)
df.groupby(['Pclass','Sex']).size()

df['Age'].plot.hist() # distribution of age
df['Fare'].plot.hist()
df['Fare'].plot.hist(bins=20)  # we get the clear picture   
df['Fare'].plot.hist(bins=20,figsize=(10,5))

df.groupby(['Siblings/Spouses Aboard']).size().plot(kind='bar')   

sns.boxplot(x='Pclass',y='Age',data=df)

sns.boxplot(x='Siblings/Spouses Aboard',y='Age',data=df) 
   
#######################################################################################

                    DATA WRANGLING(DATA CLEANING )
                    
df.isnull()   # False means no null values and True means null values     
df.isnull().sum()  # here no missing(NaN) columns here data set is alredy clean
sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis') # viridis is a color name 

  

           extract csv file of titanic with null values
df2.isnull()
df2.isnull().sum()  # in Fare column we find the null values so we must washout the null values.
sns.heatmap(df2.isnull(),yticklabels=False,cmap='viridis') # yellow color shows the null values
df2.head(10)

df2.drop('Fare',axis=1,inplace=True) # we remove the Fare column
df2.head(10) # check for column is removed or not
sns.heatmap(df2.isnull(),yticklabels=False,cmap='viridis') # check for column is removed or not
df2.isnull().sum()       #check for column is removed or not 
           
pd.get_dummies(df2['Sex'])  # we get hu and all male and hu and all female                    
   ## only male column is enough to represent hu and all male and female 
   
sex=pd.get_dummies(df2['Sex'],drop_first=True)

# we can also check the Pclass
pd.get_dummies(df2['Pclass'])   
pclass=pd.get_dummies(df2['Pclass'],drop_first=True) ## 2nd class 3rd get zero then the person is definitly belongs to the 1st class

concatn=pd.concat([df2,sex,pclass],axis=1)    ##  we ew get side by side

concatn.drop(['Pclass','Sex','Name'],axis=1,inplace=True)  ## not required unwanted columns

concatn.head(10) ## it is the final data set all values converted into zero and one
#############################################################################################

                        TRAIN AND TEST DATA
- here we split data into train subset and test subset.
- next is build model on train data and predict the data on test subset.

    TRAIN DATA----
# here we need to define independent and dependent variable.                      
X=concatn.drop('Survived',axis=1)  # except survive column all other columns are independent variable 
y=concatn['Survived'] # predict this column as passengers survive or not

# next is split the data into triny and test subset.
    
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1) # split size=30%

from sklearn.linear_model import LogisticRegression # logistic regression is a graph on linaer regression
logmodel=LogisticRegression() # creating the instance
logmodel.fit(X_train,y_train)  # fit my model    

predictions = logmodel.predict(X_test) ## i need to make thr predictions and pass the log model

from sklearn.metrics import classification_report # here is the main predictions and classification report
classification_report(y_test,predictions)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)     # we get four outcomes
    
#############################################################
                       ACCURACY CHECK 

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
 
######################################################################

  TAKE ANOTHER ONE EXAMPLE ie. XUV CSV

import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
%matplotlib inline        

xuv=pd.read_csv(r'C:\Users\tejkumar.p\Desktop\python\csv files\xuv.csv')

x=xuv.iloc[:,[2,3]].values
y=xuv.iloc[:,4].values  # iloc is used for selection by index
print(x)
print(y)   ## here 0 stands for did't purchased and 1 stands for purchased

x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.25,random_state=0) 
  # divided train and test plate size as 75(train data) & 25(test data) ratio 
  


  

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)  #  scaling our data when our data contains more than 1000 rows





from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train) # it tells the all logistic regression
  
y_pred=classifier.predict(x_test)
print(y_pred) # here we get the zero's one"s

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred) 

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

accuracy_score(y_test,y_pred)*100  # 80% people are ready to bye a car

















                   
            
             