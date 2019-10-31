
                                   NAIVEBAYES ALGORITHM
                                   
- classification is based on bayes theorm.
- feactures of a perticular class is unrelated to other feactures.
- naivebayes is a simple but surprisingly powerful algorithm for predictive modelling.
- it is a classification technique based on bayes theorm, with an assumption of independent among predictors.
- it comprices two parts: they are naive and bayes.
- it explains presence of the perticular feacture in a class, is unrelated to the presence of any other feacture.                                    
      
   BAYES THEORM:
       given a hypothesis(H) and evidence(E), Bayes theorm states that the relationship between the probability
       of the hypothesis before getting the evidence P(H) and probability of the hypothesis after getting the
       evidence p(H|E) is,
                            P(H|E)=P(E|H)*P(H) / P(E)
                            
           P(H) is the prior probability
           P(H|E) is the posterior probability
           
application:
- news categorization. ex:sports,national,international,news channels
- spam filtering in emails.
- object and face detection.
- medical diagnosis.
- weather prediction.

TYPES OF NAIVEBAYES:
    gaussian,multinomial and bernoulli

GAUSSIAN: it is used in classification and assumes that the feacture follow in normal distribution.
MULTINOMIAL: it is used for descrete counts.

STEPS USED IN NAIVEBAYES:
- handling data: load the data from CSV file and make train,test datasets
- summarizing the data: calculate the pobabilities and make predictions. 
- making the predictions: we calculate probability from train,test data.(single prediction)
- evaluate the accuracy.
- typing all together  


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
accuracy = accuracy_score(target_test, pred)*100




###############################################################################3




           
                               
                               
                        NEURAL NETWORK ALGORITHM

- it is used in bank notes are real or fake?
- machine learns by experience.

                                                          
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                                                          