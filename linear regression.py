# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:22:51 2019

@author: TEJ
"""
# imports
import numpy as np
import matplotlib.pyplot as plt

# generate random data-set
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# plot
plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
############################################################################

                           LR_single_variable (very imp)

-- QUE: PREDICT THE PRICE OF AREA 33OO


import os
os.chdir('D:\\DS\\ds\\python\\csv files\\LR_single_variable very imp')
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('LR_single_variable 1st excel.csv')
df

from sklearn import linear_model
%matplotlib inline
plt.scatter(df.area,df.price,color='red',marker='*')
plt.xlabel('area')
plt.ylabel('price')

reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

######################################################################

# for multiple features then,
reg.fit(df[['area','bedrooms','home_age']],df.price)
# predict reg.pred([[4000,5,23]])

####################################################################

reg.predict([[3300]])
# o/p is array([628715.75342466])

reg.coef_
# o/p is,  array([135.78767123])

reg.intercept_
# o/p is, 180616.43835616432

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='*')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')


manually we calculated as,

Y=(m*x) + C

Y=(135.78767123*3300) + 180616.43835616432 
Y
# O/P IS, 628715.7534151643 by comparing, both are same

################################################################################
- If we store the python objects to the file, we use 2 process 

1. PICKLE method
2. JOBLIB method   (# if model consist of large numpy array, then use JOBLIB method
                    # which gives efficiency to the model)


PICKLE METHOD--

import pickle

with open('reg_pickle','wb') as f:  # study sklearn model persistance
    pickle.dump(reg,f)
# file is created on folder see in csv folder the folder    
    
with open('reg_pickle','rb') as f:
    mp=pickle.load(f)   
    
mp.predict([[3300]])
# array([628715.75342466])
    


JOBLIB---

from sklearn.externals import joblib
joblib.dump(reg,'reg_joblib')  # model is successfully saved

mj=joblib.load('reg_joblib')   # loading from the folder

mj.predict([[3300]])
# array([628715.75342466])

mj.coef_
# array([135.78767123])

mj.intercept_
# 180616.43835616432




################################################################################
import os
os.chdir('D:\\DS\\ds\\python\\csv files')
import pandas as pd
import matplotlib.pyplot as plt
df1=pd.read_csv('LR_single_variable(areas).csv')
df1

new_area_pred=reg.predict(df1)
new_area_pred

df1['prices']=new_area_pred
df1

df1.to_csv('output.csv')







######################################################################


                     MULTIPLE LINEAR REGRESION

- below datasets contains multiple independent variable
- we have world happiness data
find, 
       how to describe the data?
       can we make predictive model to calculate the happiness score?

import pandas as pd
import numpy as np


df1=pd.read_csv('2015.csv')
df1
pd.set_option('display.max_columns',12)

df2=pd.read_csv('2016.csv')
df2

df3=pd.read_csv('2017.csv')
df3

df1.columns.size # 12 columns

df2.columns.size  # 13 columns

df3.columns.size  # 12  columns

##########################################
               EXPLORATORY DATA ANALYSIS

# next we drop the un-necessary columns

df1.drop(['Region','Standard Error'],axis=1,inplace=True)
df1

df2.drop(['Region','Lower Confidence Interval','Upper Confidence Interval'],axis=1,inplace=True)
df2

df3.drop(['Whisker.high','Whisker.low'],axis=1,inplace=True)
df3

#################################################

# in above df1,df2 and df3 the columns names are not same, so make it same

df1.columns=['Country','Happiness_Rank','Happiness_Score','Economy','Family','Health','Freedom',
             'Trust','Generosity','Dystopia_Residual']

df2.columns=['Country','Happiness_Rank','Happiness_Score','Economy','Family','Health','Freedom',
             'Trust','Generosity','Dystopia_Residual']

df3.columns=['Country','Happiness_Rank','Happiness_Score','Economy','Family','Health','Freedom',
             'Trust','Generosity','Dystopia_Residual']

####################################################

df4=df1.append(df2,ignore_index=True)
df4

happy=df4.append(df3,ignore_index=True)
happy

###############################################

happy.isnull().sum()
sns.heatmap(happy.isnull(),yticklabels=False,cmap='viridis') # no NaN values

##########################################################################################
                       VISUALIZATION OF DATA BY PLOTS (data mining)
                       
 # correlation by heatmap                      
import matplotlib.pyplot as plt

plt.matshow(happy.corr())
plt.show()                       
             

import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = happy.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)   
# Happiness_Rank and Happiness_Rank highly correlated each other 
# correlation exist between 0 and 1        
# drop the 'Country' and 'Happiness_Rank' column which is not neccesary

happy.drop(['Country','Happiness_Rank'],axis=1,inplace=True)  
happy 

################################################################################

                         PREPARE THE MODELS
                         
                         
feature_names = ['Economy','Family','Health','Freedom',
             'Trust','Generosity','Dystopia_Residual']
x = happy[feature_names]
y =happy.Happiness_Score                         
                         


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
models=[]
models.append(('LR',LinearRegression()))

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)  # y predicted values and y_test is the actual value

# Visualising the Training set results and best fit line
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train,model.predict(x_train), color = 'blue')


# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, model.predict(x_train), color = 'blue')
plt.show()






###############################
result_model=pd.DataFrame({'actual':y_test,'predict':y_pred})
result_model['diff']=y_test - y_pred  # y_test is actual value
result_model # o/p is some cases is positive and some cases negative
             # difference is very small so our model reasonably well 

from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
%matplotlib inline
            
mean_absolute_error(y_test,y_pred) #0.000257722727904718
mean_squared_error(y_test,y_pred)  #8.877314043713253e-08

np.sqrt(mean_squared_error(y_test,y_pred))  # 0.00029794821771095145
# here root mean square error is very very low so model is good performed
 
model.score(x_test,y_test) # variance
# if variance is 1 perfect prediction
           
             
################################

              finding intercept and co-efficients

model.intercept_  # o/p is 0.00019507504511295792
# above is the intercept value of Happiness_Score

model.coef_ 

# o/p is ([1.00003992, 0.99994824, 0.99992688, 0.99996666, 0.99991521,
#       1.00006031, 0.99994221]) 
# these are co-efficients of  the independant variable

t=pd.DataFrame(feature_names)
t.columns=['feature_names']
t


s=pd.DataFrame(model.coef_)
s.columns=['model.coef_']
s
    
con=pd.concat([t,s],axis=1)
con
################################################################
import seaborn as sns
sns.regplot(x='actual',y='predict',data=result_model)
# actual and predicted values are matching and no differece between actual and predicted values
# the model is 100% accurate

eq is,

Happiness_Score = 0.00019507504511295792 + (1.000040 * Economy) + (0.999948 * Family) +
                  ...... + (0.999942 * Dystopia_Residual)    




































              



                                               









                                              
                         
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       









































