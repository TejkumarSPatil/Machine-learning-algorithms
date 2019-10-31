# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:40:34 2019

@author: TEJ
"""

survival - Survival (0 = No; 1 = Yes)
class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
name - Name
sex - Sex
age - Age
pclass- passener class
sibsp - Number of Siblings/Spouses Aboard
parch - Number of Parents/Children Aboard
ticket - Ticket Number
fare - Passenger Fare
cabin - Cabin
embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat - Lifeboat (if survived)
body - Body number (if did not survive and body was recovered)







import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd
%matplotlib inline

os.chdir('E:\\python\\csv files\\titanic csv')



train_df=pd.read_csv(r'D:\DS\ds\python\csv files\titanic csv\train.csv')
train_df


test_df=pd.read_csv('test.csv')
test_df

pd.set_option('display.max_columns',12)

train_df.columns
train_df.columns.size

test_df.columns
test_df.columns.size



from the train dataset, 
                        continous: Age, Fare                       
                        descrete: SibSp, Parch,  Pclass
                        categorical: Survived, Sex, Embarked
                        
tell similar to test dataset.                        
                        
- Ticket is a mix of numeric(numbers) and alphanumeric data(lowercase and uppercase letters)types. 
- Cabin is alphanumeric.  


train_df.head() 
train_df[0:6]
  
train_df.tail()                   


train_df.isnull().sum()  # Cabin > Age > Embarked
test_df.isnull().sum()   # Cabin > Age

train_df.info()
test_df.info()

- Seven features are integer or floats of train data set.
- Six in case of test dataset.


train_df.describe()
test_df.describe()


train_df.describe(include='all') 
# both are same
train_df.describe(include=['object'])

# or, for individual column checking
train_df.groupby(['Embarked']).size() # it gives frequency(highest count)


# We want to know how well does each feature correlate with Survival.
import matplotlib.pyplot as plt
plt.matshow(train_df.corr())
plt.show()                       
             

import seaborn as sns

train_df.corr()


f, ax = plt.subplots(figsize=(10, 8))
corr = train_df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)  


# finding total family members of each person

sum=train_df['SibSp'] + train_df['Parch']
sum

train_df['Name']
concatn=pd.concat([train_df['Name'],sum],axis=1)
concatn

##################################################
# Pclass

train_df.groupby(['Pclass']).size()
train_df.groupby(['Pclass']).size().plot(kind='bar')
train_df[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', 
        ascending=False)

train_df.groupby(['Pclass','Survived']).size().reset_index(name='counts').plot(kind='bar')
train_df.groupby(['Pclass','Survived']).size().reset_index(name='counts')
train_df.groupby(['Pclass','Sex','Survived']).size().reset_index(name='counts')

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

Observations.
- Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
- Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
- Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
- Pclass varies in terms of Age distribution of passengers.

###############

#Sex
train_df.groupby(['Sex']).size()
train_df.groupby(['Sex']).size().plot(kind='bar')
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by=['Survived'],
        ascending=False)

# Embarked
train_df.groupby(['Embarked']).size()
train_df.groupby(['Embarked']).size().plot(kind='bar')
train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',
          ascending=False)

train_df.groupby(['Embarked','Survived']).size().reset_index(name='counts')
train_df.groupby(['Embarked','Survived']).size().reset_index(name='counts').plot(kind='bar')
train_df.groupby(['Embarked','Sex','Survived']).size().reset_index(name='counts')



# SibSp
train_df.groupby(['SibSp']).size()
train_df.groupby(['SibSp','Survived']).size().reset_index(name='counts')
train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by=['Survived'],
        ascending=False)


# Parch
train_df.groupby(['Parch']).size()
train_df.groupby(['Parch','Survived']).size().reset_index(name='counts')
train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by=['Survived'],
        ascending=False)

####################################################################

#AGE
train_df.boxplot(column='Age')

train_df[['Age','Survived']].groupby(['Age'],as_index=False).mean().sort_values(by='Survived',
        ascending=False)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

train_df[train_df.Age <= 50].shape[0]

train_df[train_df.Age <=50].groupby('Survived').size()

train_df[train_df.Age <=50].groupby('Survived').count()


Observations.
- Infants (Age <=4) had high survival rate.
- Oldest passengers (Age = 80) survived.
- Large number of 15-25 year olds did not survive.
- Most passengers are in 15-35 age range.

############################################################################################

# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


##########################################################################

# Fare
train_df[train_df.Fare <= 50].shape[0]

train_df[train_df.Fare <= 50].groupby('Survived').count()

train_df[train_df.Fare <= 50].groupby('Survived').size()

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

Observations.
- Higher fare paying passengers had better survival. Confirms our assumption for creating fare ranges.

###########################################################################
# before


train_df.shape  # (891, 12)

test_df.shape   #  (418, 11)


##################################################

                              Wrangle data
                              
- Correcting by dropping features                              
                              
train_df.drop(['Cabin','Ticket'],axis=1,inplace=True)
train_df

test_df.drop(['Cabin','Ticket'],axis=1,inplace=True)
test_df



## after

train_df.shape   # (891, 10)

test_df.shape   # (418, 9)


combine=[train_df,test_df]
combine

###########################################################################33

# Name

for dataset in combine:   # classification on names
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex'])
train_df
test_df

Observations.

When we plot Title, Age, and Survived, we note the following observations.

- Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
- Survival among Title Age bands varies slightly.
- Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).

############


We can replace many titles with a more common name or classify them as Rare.

# Rare='Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

##########

- We can convert the categorical titles to ordinal


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
train_df
test_df 

##########

- Now we can safely drop the Name feature from training and testing datasets.
  We also do not need the PassengerId feature in the training dataset.

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape # ((891, 9), (418, 9))

######################################################################################

# Sex

- Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df    
test_df  

###########################################

- So instead of guessing age values based on median, use random numbers between mean and standard deviation,
   based on sets of Pclass and Gender combinations.  

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


train_df.isnull().sum()  # Age 177 NaN values
test_df.isnull().sum()   # Age 86 NaN values 

############################

# Age columns contains NaN values in train and test datasets.
# below procedure is replacing NaN values

- Let us start by preparing an empty array to contain guessed Age values based on
    Pclass x Sex combinations.


guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = datase0t[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.isnull().sum()
test_df.isnull().sum()

########

# Age coluumn create a Age band both train and test dataset

- Let us create Age bands and determine correlations with Survived.

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


Let us replace Age with ordinals based on these bands.....

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df   # ageband is applicable both to train dataset and test data set
           
          
  # next we remove the ageband column          
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

train_df
test_df

#############################################################

# combining of SibSp and Parch


- We can create a new feature for FamilySize which combines Parch and SibSp. 
    This will enable us to drop Parch and SibSp from our datasets.


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)        
    

- We can create another feature called IsAlone.

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


- Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df
test_df

#########################################################################

#make it single of pclass and age  

- We can also create an artificial feature combining Pclass and Age.

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']]

############################################################################

# EMBARKED

- Embarked feature takes S, Q, C values based on port of embarkation. 
- Our training dataset has two missing values. We simply fill these with the most common occurance.

freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

######

Converting categorical feature to numeric--
- We can now convert the EmbarkedFill feature by creating a new numeric Port feature.

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df
test_df

##############################################################

#  Fare

- We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently 
  for this feature. We do this in a single line of code.

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df
test_df.isnull().sum()

##
- We can not create FareBand train data set

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

- Convert the Fare feature to ordinal values based on the FareBand.


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']  = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df
test_df

train_df.shape  # (891, 9)
test_df.shape # (418, 9)

#################################################################

                            PREDICTING MODELS

# LOGISTIC REGRESSION

feacture_names=['Pclass','Sex','Age','Fare','Embarked','Title','IsAlone','Age*Class']
x_train=train_df[feacture_names]
y_train=train_df.Survived
X_test=test_df.drop('PassengerId',axis=1).copy()  


from sklearn.linear_model import LogisticRegression  
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(X_test)

acc_log = round(model.score(x_train, y_train) * 100, 2)
acc_log

o/p is, 80.36

######

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)

o/p is,

     Feature  Correlation
1        Sex     2.201527
5      Title     0.398234
2        Age     0.287166
4   Embarked     0.261762
6    IsAlone     0.129141
3       Fare    -0.085150
7  Age*Class    -0.311198
0     Pclass    -0.749006

- Sex is highest positivie coefficient, implying as the Sex value increases the
      (male: 0 to female: 1) probability of Survived=1 increases the most.
- Inversely as Pclass increases, probability of Survived=1 decreases the most.
- This way Age*Class is a good artificial feature to model as it has second highest 
   negative correlation with Survived. So is Title as second highest positive correlation.

#############################################################################

# Support Vector Machines
from sklearn.svm import SVC

svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(X_test)

acc_svc=round(svc.score(x_train,y_train)*100,2)
acc_svc

o/p is,
83.84

###################################################################

# KNN

from sklearn.neighbors import KNeighborsClassifier

KNN=KNeighborsClassifier()
KNN.fit(x_train,y_train)

y_pred=KNN.predict(X_test)

acc_KNN=round(KNN.score(x_train,y_train)*100,2)
acc_KNN

o/p is,
83.95

#####################################################################

# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

GNN=GaussianNB()
GNN.fit(x_train,y_train)

y_pred=GNN.predict(X_test)

acc_GNN=round(GNN.score(x_train,y_train)*100,2)
acc_GNN

o/p is,
72.28

########################################################################

# DT

from sklearn.tree import DecisionTreeClassifier

DTC=DecisionTreeClassifier()
DTC.fit(x_train,y_train)

y_pred=DTC.predict(X_test)

acc_DTC=round(DTC.score(x_train,y_train)*100,2)
acc_DTC

o/p is,
86.76

########################################################

# RF

from sklearn.ensemble import RandomForestClassifier

RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)

y_pred=RFC.predict(X_test)

acc_RFC=round(RFC.score(x_train,y_train)*100,2)
acc_RFC

o/p is,
86.31

########################################################

# GB

from sklearn.ensemble import GradientBoostingClassifier

GBC=GradientBoostingClassifier()
GBC.fit(x_train,y_train)

y_pred=GBC.predict(X_test)

acc_GBC=round(GBC.score(x_train,y_train)*100,2)
acc_GBC

o/p is,
84.74

######################################################

models=pd.DataFrame({'Model':['KNN','SVC','LOGREG','DT','GNN','RF','GB'],
                    'score':[acc_KNN,acc_svc,acc_log,acc_DTC,acc_GNN,acc_RFC,acc_GBC]})
print(models)

o/p is,

    Model  score
0     KNN  83.95
1     SVC  83.84
2  LOGREG  80.36
3      DT  86.76
4     GNN  72.28
5      RF  86.53
6      GB  84.74


################################################################

# final output

submission=pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": y_pred})
submission    
    
##################################################################













