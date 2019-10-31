# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:54:35 2019

@author: TEJ
"""

                        DATA  EXLORATION
                        
-  first we should analyse and “get to know” the data set.
-  to gain some understanding about the potential features and to see if data cleaning is needed OR NOT.                        
  
- import the necessary libraries 
                      
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mode
from pylab import *
%matplotlib inline

os.chdir('D:\DS\ds\python\csv files')

dia=pd.read_csv('diabetes.csv')

dia

pd.set_option('display.max_columns',9)

dia.isnull().sum()
dia.isna().sum()

dia.columns  # it contains a limited feactures

dia.head() # automatically print first 5 rows

dia.shape

We can observe that the data set contain 768 rows and 9 columns. 
‘Outcome’ is the column which we are going to predict , which says if the patient is diabetic or not. 
1 means the person is diabetic and 0 means person is not. 
We can identify that out of the 768 persons, 500 are labeled as 0 (non diabetic) and 268 as 1 (diabetic)


dia.groupby(['Outcome']).size()
dia.groupby(['Outcome']).size().plot(kind='bar')
############################################################################################
                                     DATA visualization

- Visualization of data is an imperative aspect of data science. 
- It helps to understand data and also to explain the data to another person. 
- Python has several interesting visualization libraries such as Matplotlib, Seaborn etc.
- here, we use pandas visualization

plt.hist('Pregnancies',bins=15,data=dia)
plt.hist('Glucose',bins=10,data=dia)
plt.hist('BloodPressure',bins=10,data=dia)
plt.hist('SkinThickness',bins=10,data=dia)
plt.hist('Insulin',bins=10,data=dia)
plt.hist('BMI',bins=10,data=dia)
plt.hist('DiabetesPedigreeFunction',bins=10,data=dia)
plt.hist('Age',bins=10,data=dia)
plt.hist('Outcome',bins=10,data=dia)


dia.hist(figsize=(9,9)) # it gives whole

dia.groupby('Outcome').hist(figsize=(9,9)) ## it gives 0 and 1 of Outcome

dia.groupby(['Pregnancies','Outcome']).size().reset_index(name='counts') 

dia.boxplot(column='Pregnancies')   # check how data is distributed

sns.boxplot(data=dia)

sns.stripplot(x='Outcome',y='Glucose',data=dia)

sns.swarmplot(x='Outcome',y='Glucose',data=dia)
#############################################################################
                         DATA CLEANING
   Duplicate or irrelevant observations.
   Bad labeling of data, same category occurring multiple times.
   Missing or null data points.
   Unexpected outliers.                      
                         
                         
                         
Blood pressure : By observing the data we can see that there are 0 values for blood pressure.
                   And it is evident that the readings of the data set seems wrong because a living person cannot have diastolic blood pressure of zero.
                   By observing the data we can see 35 counts where the value is 0.
                   
                   
dia[dia.BloodPressure == 0] # it gives peoples who have blood pressure as 0

dia[dia.BloodPressure == 30]  # it gives peoples who have blood pressure as 30

dia[dia.BloodPressure == 0].shape[0]  #  it gives total count

dia[dia.BloodPressure == 30].shape[0]

dia[dia.BloodPressure >= 100]  #  it gives peoples who have blood pressure greater than 100

dia[dia.BloodPressure >= 100].shape[0] # it gives count

dia[dia.BloodPressure == 0].groupby('Outcome').size()

dia[dia.BloodPressure == 0].groupby('Outcome').count()
 
###################################

glucose levels : Even after fasting glucose level would not be as low as zero.
                  Therefor zero is an invalid reading. By observing the data we can see 5 counts where the value is 0.

dia[dia.Glucose == 0]

dia[dia.Glucose == 0].shape[0]

dia[dia.Glucose == 0].groupby('Outcome').size()

dia[dia.Glucose == 0].groupby('Outcome').count()

########################################

Skin Fold Thickness : For normal people skin fold thickness can’t be less than 10 mm better yet zero. 
                      Total count where value is 0 : 227.

dia[dia.SkinThickness == 0]

dia[dia.SkinThickness == 0].shape[0]

dia[dia.SkinThickness == 0].groupby('Outcome').size()

dia[dia.SkinThickness == 0].groupby('Outcome').count()

#############################################

BMI : Should not be 0 or close to zero unless the person is really underweight which could be life threatening.

dia[dia.BMI == 0]

dia[dia.BMI == 0].shape[0]

dia[dia.BMI == 0].groupby('Outcome').size()

dia[dia.BMI == 0].groupby('Outcome').count()

###############################################

Insulin : In a rare situation a person can have zero insulin but by observing the data, 
             we can find that there is a total of 374 counts.
             
dia[dia.Insulin == 0]  

dia[dia.Insulin == 0].shape[0] 

dia[dia.Insulin == 0].groupby('Outcome').size()

dia[dia.Insulin == 0].groupby('Outcome').count()

######################################################

dia[dia.SkinThickness == 0].shape[0] # 227 zeros 
dia[dia.Insulin == 0].shape[0] # 374 zeros
dia[dia.BloodPressure == 0].shape[0] # 35 zeros
dia[dia.Glucose == 0].shape[0] # 5 zeros
dia[dia.BMI == 0].shape[0]  # 11 zeros

- From the above, remove the zeros rows of BloodPressure,Glucose and BMI because these zeros not 
  effecting the model.

dia_mod = dia[(dia.BloodPressure != 0) & (dia.BMI != 0) & (dia.Glucose != 0)] 
dia_mod 
dia_mod.shape         
##############################################################   

                   FEACTURE ENGINEERING
                   
- Feature engineering is the process of transforming the gathered data into features that better
  represent the problem that we are trying to solve to the model, to improve its performance and accuracy. 

- Feature engineering create more input features from the existing features and also combine several 
  features to produce more intuitive features to feed to the model.                  

dia_mod.columns

(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')

- By a crude observation we can say that the ‘Skin Thickness’ is not an indicator of diabetes. 
  But we can’t deny the fact that it is unusable at this point.
  
- We separate the data set into features and the response that we are going to predict.
   We will assign the features to the X variable and the response to the y variable.
   
   
   
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = dia_mod[feature_names]
y =dia_mod.Outcome


 

- Model selection or algorithm selection phase is the most exciting and the heart of machine learning. 
  It is the phase where we select the model which performs best for the data set at hand. 
  
- First we will be calculating the “Classification Accuracy (Testing Accuracy)” of a given set of
  classification models with their default parameters to determine which model performs better with the
  diabetes data set.  

- We will import the necessary libraries to the notebook. We import 7 classifiers namely K-Nearest Neighbors, 
  Support Vector Classifier, Logistic Regression, Gaussian Naive Bayes
  Random Forest and Gradient Boost to be contenders for the best classifier.
  
                              Model Selection
                              
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC   # support vector classifier(support vector machine)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))    # support vector classifier
models.append(('LR', LogisticRegression()))
models.append(('DT', Decision`TreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier())) 

some times SVM can used as classification as well as regression see the notes

######################################################################################################

                            Evaluation Methods

- It is a general practice to avoid training and testing on the same data. 
  The reasons are that, the goal of the model is to predict out-of-sample data, and the model could be overly complex leading to overfitting.
   To avoid the aforementioned problems, there are two precautions.

           1. Train/Test Split
           2. K-Fold Cross Validation
           
- We will import “train_test_split” for train/test split and “cross_val_score” for k-fold cross validation.
   “accuracy_score” is to evaluate the accuracy of the model in the train/test split method.          


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

###########

                       MODEL EVALUATION


                             Train/Test Split(finding accuracy of each algorithm)
                             

- This method split the data set into two portions : a training set and a testing set.
  The training set is used to train the model. And the testing set is used to test the model, and evaluate the accuracy.

- Next we can split the features and responses into train and test portions.

-  train/test split is still useful because of its flexibility and speed

 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)


- Then we fit each model in a loop and calculate the accuracy of the respective model 
  using the “accuracy_score”.
  
 ################################################ 
model=KNeighborsClassifier()
model.fit(x_train,y_train) 

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

Out[184]: 0.7247706422018348
 
 ##################################################3 
model=SVC()
model.fit(x_train,y_train) 

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

Out[185]: 0.6605504587155964 
 ####################################################3
model=LogisticRegression()
model.fit(x_train,y_train) 

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred) 

Out[186]: 0.7614678899082569 
#################################################### 
 
model=DecisionTreeClassifier()
model.fit(x_train,y_train) 

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred) 

Out[187]: 0.6513761467889908 
################################################ 
 
model=GaussianNB()
model.fit(x_train,y_train) 

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

Out[188]: 0.7385321100917431  
####################################################
model=RandomForestClassifier()
model.fit(x_train,y_train) 

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

Out[189]: 0.7385321100917431
########################################################
model=GradientBoostingClassifier()
model.fit(x_train,y_train) 

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

Out[190]: 0.7614678899082569
 ######################################################
  
  
names = []
scores = []
for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)  

o/p is,

 Name     Score
0  KNN  0.724771
1  SVC  0.660550
2   LR  0.761468
3   DT  0.637615
4  GNB  0.738532
5   RF  0.715596
6   GB  0.756881

FROM THE ABOVE,              DT, RF and GB acuuracy are different
#########################################################################

axis = sns.barplot(x = 'Name', y = 'Score', data = tr_split)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()

#######################
  ######################                        
                         K-Fold Cross Validation
- tune the model                         
- This method splits the data set into K equal partitions (“folds”), 
   then use 1 fold as the testing set and the union of the other folds as the training set. 
- Then the model is tested for accuracy. 
- The process will follow the above steps K times, using different fold as the testing set each time. 
- The average testing accuracy of the process is the testing accuracy.   


names = []
scores = []

for name, model in models:
    score = cross_val_score(model,x, y, cv=10, scoring='accuracy').mean()
    
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

o/p is,
  Name     Score
0  KNN  0.711521
1  SVC  0.656075
2   LR  0.776440
3   DT  0.685532
4  GNB  0.755681
5   RF  0.748813
6   GB  0.765442


axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
plt.show()


 #################           

- We will move forward with K-Fold cross validation as it is more accurate and use the data efficiently. 
  We will train the models using 10 fold cross validation and calculate the mean accuracy of the models. 
  “cross_val_score” provides its own training and accuracy calculation interface.
  
- We can plot the accuracy scores using seaborn
- We can see the Logistic Regression, Gaussian Naive Bayes, Random Forest and Gradient Boosting have performed better than the rest. 
  From the base level we can observe that the Logistic Regression performs better than the other algorithms.

- At the baseline Logistic Regression managed to achieve a classification accuracy of 77.64 %. 
  This will be selected as the prime candidate for the next phases.
  
-  we were able to identify that Logistic Regression performed better than the other selected classification models. 
   In this article we will be discussing the next stages of the machine learning workflow, advanced feature engineering, and hyper parameter tuning.  

################################################################################################################3                                
   
                      advanced Feature Engineering (Revisited)
                                 
-  there are features that don’t improve the model. Such cases can be found by further analysis 
   of the features related to the model. “One highly predictive feature makes up for 10 duds”.  

- the selected model which is Logistic Regression, and how feature importance affects it.

- Scikit Learn provided useful methods by which we can do feature selection 
   and find out the importance of features which affect the model.                               

1. Univariate Feature Selection : Statistical tests can be used to select those
     features that have the strongest relationship with the output variable.

2. Recursive Feature Elimination : The Recursive Feature Elimination (or RFE) works by 
   recursively removing attributes and building a model on those attributes that remain. 
   It uses the model accuracy to identify which attributes (and combination of attributes) 
   contribute the most to predicting the target attribute.
   
3. Principal Component Analysis : Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form. 
    Generally this is called a data reduction technique.
    A property of PCA is that you can choose the number of dimensions or principal component in the transformed result.   

4. Feature Importance : Bagged decision trees like Random Forest and Extra 
                         Trees can be used to estimate the importance of features.
                         
 In this tutorial we will be using Recursive Feature Elimination as the feature selection method. 


- First we import RFECV, which comes with inbuilt cross validation feature. 
   Same as the classifier model, RFECV has the fit() method which accepts features and the response/target. 
                    
#################################################################################################

                     Logistic Regression — Feature Selection by RFE
                                (Recursive Feature Elimination)
# recursive feature elimmination with cross validation                     
from sklearn.feature_selection import RFECV
logreg_model = LogisticRegression()
rfecv = RFECV(estimator=logreg_model, step=1, cv=10, scoring='accuracy')
rfecv.fit(x, y)  

- After fitting, it exposes an attribute grid_scores_ which returns a list of accuracy scores for each of the features selected. 
  We can use that to plot a graph to see the no of features which gives the maximum accuracy for the given model.                   
                     

plt.figure()
plt.title('Logistic Regression CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

- By looking at the plot we can see that inputting 4 features to the model gives the best accuracy score.
  RFECV exposes support_ which is another attribute to find out the features which contribute the most to predicting. 
  In order to find out which features are selected we can use the following code.


feature_importance = list(zip(feature_names, rfecv.support_))
new_features = []
for key,value in enumerate(feature_importance):
    if(value[1]) == True:
        new_features.append(value[0])
print(new_features)

o/p is,

['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction']

- We can see that the given features are the most suitable for predicting the response class.
  We can do a comparison of the model with original features and the RFECV selected features to 
  see if there is an improvement in the accuracy scores.



# Calculate accuracy scores 
x_new = dia_mod[new_features]
initial_score = cross_val_score(logreg_model, x, y, cv=10, scoring='accuracy').mean()
print("Initial accuracy : {} ".format(initial_score))
fe_score = cross_val_score(logreg_model, x_new, y, cv=10, scoring='accuracy').mean()
print("Accuracy after Feature Selection : {} ".format(fe_score))


Initial accuracy : 0.7764400711728514 
Accuracy after Feature Selection : 0.7805877119643279 

- By observing the accuracy there is a slight increase in the accuracy after feeding the selected 
  features to the model.
  
  ########################################################################################
  
                         Gradient Boosting — Feature Selection (another algorithm)
                         
- it cosist of 2 they are: bagging
                           boosting
                           
-                            
                         
- boosting is stage wise process, every stage boost the cllasifier from the previous stage such
   that error is reduced                     
- We can also analyze the second best model we had which is Gradient Boost classifier, to see if features 
  selection process increases the model accuracy and if it would be better than Logistic Regression after the process.
- We follow the same procedure which we did for Logistic Regression.
- GB is a example of ensemble learning.(many models work together)                     
- it is special type of boosting technique where it reduces the errors sequentially.

- for single thread we can break, but combination thread called rope, we cant break. like that GB is  
  consist of week models to make it strong one.
  
- Decision tree is the good example of ensemble learning.  

gb_model = GradientBoostingClassifier()
gb_rfecv = RFECV(estimator=gb_model, step=1, cv=10, scoring='accuracy')
gb_rfecv.fit(x, y)
plt.figure()
plt.title('Gradient Boost CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(gb_rfecv.grid_scores_) + 1), gb_rfecv.grid_scores_)
plt.show()


- We can see that having 4 input features generates the maximum accuracy.



feature_importance = list(zip(feature_names, gb_rfecv.support_))
new_features = []
for key,value in enumerate(feature_importance):
    if(value[1]) == True:
        new_features.append(value[0])
print(new_features)


- ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
- The above 4 features are most suitable for the model. We can compare the accuracy before and after feature selection.


x_new_gb = dia_mod[new_features]
initial_score = cross_val_score(gb_model, x, y, cv=10, scoring='accuracy').mean()
print("Initial accuracy : {} ".format(initial_score))
fe_score = cross_val_score(gb_model, x_new_gb, y, cv=10, scoring='accuracy').mean()
print("Accuracy after Feature Selection : {} ".format(fe_score))


- Initial accuracy : 0.764091206294081 
- Accuracy after Feature Selection : 0.7723495080069458


We can see that there is an increase in accuracy after the feature selection.

- However Logistic Regression is more accurate than Gradient Boosting. 
  So we will use Logistic Regression for the parameter tuning phase.
  
  #######################################################################
  
                         Model Parameter Tuning
                         
-  which gives decent accuracy scores.
-  It also provides the user with the option to tweak the parameters to further increase the accuracy.
- Out of the classifiers we will select Logistic Regression for fine tuning, 
  where we change the model parameters in order to possibly increase the accuracy of 
  the model for the particular data set. 

- Instead of having to manually search for optimum parameters, we can easily perform an exhaustive 
   search using the GridSearchCV, which does an “exhaustive search over specified parameter values for an estimator”.                        

-It is clearly a very handy tool, but comes with the price of computational cost when parameters to be searched are high.

- Important : When using the GridSearchCV, there are some models which have parameters that don’t work with each 
  other. Since GridSearchCV uses combinations of all the parameters given, if two parameters don’t work with 
  each other we will not be able to run the GridSearchCV.

- If that happens, a list of parameter grids can be provided to overcome the given issue.

- First we import GridSearchCV.



from sklearn.model_selection import GridSearchCV
# Specify parameters
c_values = list(np.arange(1, 10))
param_grid = [
    {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
    {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']}]

    
- Then we fit the data to the GridSearchCV, which performs a K-fold cross validation on the data for the 
  given combinations of the parameters. This might take a little while to finish. 

   
             
grid = GridSearchCV(LogisticRegression(), param_grid, cv=10, scoring='accuracy')
grid.fit(x_new, y)

- After the training and scoring is done, GridSearchCV provides some useful attributes to find the best parameters and the best estimator.


print(grid.best_params_)
print(grid.best_estimator_)

o/p is,
{'C': 1, 'multi_class': 'ovr', 'penalty': 'l2', 'solver': 'liblinear'}




                  HYPER PARAMETER TUNING
- We can observe that the best hyper parameters are as follows.

{'C': 1, 'multi_class': 'ovr', 'penalty': 'l2', 'solver': 'liblinear'}


- We can feed the best parameters to the Logistic Regression model and observe whether it’s accuracy has increased.

logreg_new = LogisticRegression(C=1, multi_class='ovr', penalty='l2', solver='liblinear')
initial_score = cross_val_score(logreg_new, x_new, y, cv=10, scoring='accuracy').mean()
print("Final accuracy : {} ".format(initial_score))


We manage to achieve a classification accuracy of 78.0  9% which we can say is quite good.

#########################################################################################################









                     What is Model Evaluation?
                     
- Model evaluation is the process of choosing between models, different model types, tuning parameters, and features. 
   Better evaluation processes lead to better, more accurate models in your applications.                     

- In this article, we’ll be discussing model evaluation for supervised classification models.
    We’ll cover evaluation procedures, evaluation metrics, and where to apply them.
    
# Features/Response
feature_names = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction']
x1 = dia_mod[feature_names]
y1 = dia_mod.Outcome    
    


                   Train/Test Split
                   
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, stratify = y1, random_state = 0)

logreg = LogisticRegression(C=1, multi_class='ovr', penalty='l2', solver='liblinear')

logreg.fit(x1_train, y1_train)
y1_pred = logreg.predict(x1_test) 

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy {}'.format(accuracy)) 


                   K-Fold Cross Validation 
                   
accuracy = cross_val_score(logreg, x1, y1, cv = 10, scoring='accuracy').mean()
print('Accuracy {}'.format(accuracy)) 

#######################################################################################
                       Model Evaluation Metrics
                       
- A module evaluation metric is a criterium by which the performance or the accuracy of a model is measured.                  

1. Classification Accuracy--
  Classification accuracy is by far the most common model evaluation metric used for classification problems. 
  Classification accuracy is the percentage of correct predictions.
   
2. Confusion Matrix--

   A confusion matrix can be defined loosely as a table that describes the performance of a classification
   model on a set of test data for which the true values are known. 
   
   Scikit-learn provides a method to perform the confusion matrix on the testing data set. 
   The confusion_matrix method requires the actual response class values and the predicted values to determine the matrix.

   it can be categorized as a binary classification problem. Therefore the confusion matrix is a 2 X 2 grid. 
   

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y1_test, y1_pred)
print(confusion)

           Metrics computed from the confusion matrix--
           
# True Positives
TP = confusion[1, 1]
TP      

# True Negatives
TN = confusion[0, 0]
TN

# False Positives
FP = confusion[0, 1]
FP

# False Negatives
FN = confusion[1, 0]
FN 

            Classification accuracy  
            
print((TP + TN) / float(TP + TN + FP + FN)) # manual method

0.7955801104972375
print(accuracy_score(y1_test, y1_pred)) 


-- Sensitivity/Recall/True positive rate        

Sensitivity or recall is the ratio of correct positive predictions to the total no. of positive predictions.
 Or more simply, how sensitive the classifier is for detecting positive instances. 
 This is also called the True Positive Rate.
 
print(TP / float(TP + FN))

0.5806451612903226


--   Specificity

Specificity is the ratio of correct negative predictions to the total no. of negative predictions.
 This determines how specific the classifier is in predicting positive instances.
 
           specificity=  TN / (TN + FP)

print(TN / float(TN + FP))
0.907563025210084


--   False Positive Rate

 The false positive rate is the ratio of negative predictions that were determined to be positive to the total 
 number of negative predictions.
 Or, when the actual value is negative, how often is the prediction incorrect.

print(FP / float(TN + FP))
0.09243697478991597

--  Precision

 Precision is the ratio of correct predictions to the total no. of predicted correct predictions.
 This measures how precise the classifier is when predicting positive instances.
 
print(TP / float(TP + FP))
0.7659574468085106
  


-- Confusion matrix advantages:
1. Variety of metrics can be derived.
2. Useful for multi-class problems as well.


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_pred)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 — Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)





###################################################################################################################
          Adjusting Classification Threshold
          
- It’s possible to adjust the logistic regression model’s classification threshold to increase the model’s sensitivity          
 
- After training, the model exposes an attribute called predict_proba, which returns the probability of the test data being in a particular response class.
   From this, we’ll get the probabilities of predicting a diabetic result. 
 
# store the predicted probabilities for class 1 (diabetic)
y_pred_prob = logreg.predict_proba(x1_test)[:, 1] 
plt.hist(y_pred_prob, bins=8, linewidth=1.2)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')


- Since it’s a binary classification problem, the classification probability threshold is 0.5, 
  which means if the probability is less than 0.5, it’s classified as “0 (non-diabetic)”.
  If the probability is more than 0.5, it’s classified as “1 (diabetic)”.
  
- We can use the Scikit-learn’s binarize method to set the threshold to 0.3,
   which will classify as ‘0 (non-diabetic)’ if the probability is less than 0.3,
   and if it’s greater it will be classified as ‘1 (diabetic)’.  
  
# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize
y_pred_class = binarize([y_pred_prob], 0.3)[0]


- Next we’ll print the confusion matrix for the new threshold predictions, and compare with the original.
 
# new confusion matrix (threshold of 0.3)
confusion_new = confusion_matrix(y1_test, y_pred_class)
print(confusion_new) 

TP1 = confusion_new[1, 1]
TP1

TN1 = confusion_new[0, 0]
TN1

FP1 = confusion_new[0, 1]
FP1

FN1 = confusion_new[1, 0]
FN1


Next we’ll calculate sensitivity and specificity to observe the changes from the previous 
confusion matrix calculations.

Previously the sensitivity calculated was 0.58. We can observe that the sensitivity has increased, 
which means it’s more sensitive to predict “positive (diabetic)” instances. 
 
# sensitivity has increased
print(TP1 / float(TP1 + FN1)) 

Using the same process, we can calculate the specificity for the new confusion matrix. 
Previously it was 0.90. We observe that it has decreased.

# specificity has decreased
print(TN1 / float(TN1 + FP1))


We adjust the threshold of a classifier in order to suit the problem we’re trying to solve.

                                 ROC curve
                                 
- An ROC curve is a commonly used way to visualize the performance of a binary classifier,
  meaning a classifier with two possible output classes. 
  The curve plots the True Positive Rate (Recall) against the False Positive Rate (also interpreted as 1-Specificity).

-  Scikit-learn provides a method called roc_curve to find the false positive and true positive rates across various thresholds, which we can use to draw the ROC curve. We can plot the curve as follows.                                 

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y1_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 — Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


We’re unable to find the threshold used to generate the ROC curve on the curve itself. 
But we can use the following method to find the specificity and sensitivity across various thresholds.


def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
    
The following is an example to show how the sensitivity and specificity behave with several thresholds.


evaluate_threshold(0.3)

evaluate_threshold(0.5)


ROC curve is a reliable indicator in measuring the performance of a classifier. 
It can also be extended to classification problems with three or more classes using the “one versus all” approach.

    
AUC (Area Under the Curve)

- AUC or Area Under the Curve is the percentage of the ROC plot that is underneath the curve.
  AUC is useful as a single number summary of classifier performance.

- In Scikit-learn, we can find the AUC score using the method roc_auc_score.

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y1_test, y_pred_prob))



- Also, the cross_val_score method, which is used to perform the K-fold cross validation method, 
  comes with the option to pass roc_auc as the scoring parameter.
- Therefore, we can measure the AUC score using the cross validation procedure as well.


cross_val_score(logreg, x1, y1, cv=10, scoring='roc_auc').mean()


--ROC/AUC advantages:
   Setting a classification threshold is not required.
   Useful even when there is a high class imbalance.






















 
 
 
 
 
 
 
 
 
 






