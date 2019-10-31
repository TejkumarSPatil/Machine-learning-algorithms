# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:24:13 2018

@author: Vinay
"""

import matplotlib.pyplot as plt  
#matplotlib inline
import numpy as np  
from sklearn.cluster import KMeans 

X = np.array([[5,3],  
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91],])

plt.scatter(X[:,0],X[:,1], label='True Position')  

kmeans = KMeans(n_clusters=2)  
kmeans.fit(X)  

##  x=np.array([40,60],[100,200])
##  kmeans.fit(x)

print(kmeans.cluster_centers_)  

print(kmeans.labels_)  

plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow') 

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')  
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')  


 #ELBOW METHOD
 
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

 
 
 ###############################################################################
              
                      CLUSTERING
                      
- clustering is the process of dividing the datasets into groups, consisting of similar data points.
- clustering is the unsupervised technique.
   ex:items arranged in a mall.(fruits and vegetables)
- it is used in banking, insurance office, retail stores etc.

types of clusering: exclusive clustering
                    overlapping clustering
                    hierarchial clustering
                    
                    
                    


--                EXCLUSIVE CLUSTRING

-- hard clustering
-- all data points belongs to the one cluster.
-- ex: K-Means clustering

                  

                  OVERLAPPING CLUSTERING
                  
-- soft cluster
-- data point belongs to multiple cluster.
-- ex: Fuzzy/C-mean clustering

                  HIERARCHIAL CLUSTERING
- REFER THE NOTES

################################################################

                      K-Means clustering                  
                           
 -- it is used to group similar elements or data points into a cluster.
 -- K refers the number of clusters.
 
# INITIALIZATION
import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
%matplotlib inline      

df=pd.DataFrame({'x':[12,20,28,18,29,33,24,45,45,52,51,52,55,53,55,23,56,54,67],
                 'y':[34,54,23,67,89,76,54,32,12,34,54,56,78,90,65,43,23,45,67]}) 
    
np.random.seed(200)
k=3  # number of clusters

# centroids [i]=[x,y]
centroids={    
    i+1:[np.random.randint(0,100),np.random.randint(0,100)]
    for i in range(k)
}
    
fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color='k') # 3 different colors
colmap={1:'r',2:'g',3:'b'} # colmap=color map

for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()  # three random datapoints(r,g,b) are initially selected as centroids  


 
# assignmet stage  (i need to assign each point to one of the cluster)
def assignment(df,centroids):
    for i in centroids.keys():
        # sqrt((x1-x2)^2 - (y1-y2)^2)
        df['distance_from_{}'.format(i)]=(
        np.sqrt(
        (df['x']-centroids[i][0]) ** 2
        + (df['y']-centroids[i][1]) ** 2))
        
    centroid_distance_cols=['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest']=df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest']=df['closest'].map(lambda x: int (x.lstrip('distance_from_')))
    df['color']=df['closest'].map(lambda x:colmap[x])
    return df

df=assignment(df,centroids)
print(df.head())

fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')

for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
    
plt.xlim(0,100)
plt.ylim(0,100)
plt.show() # co-ordinate point of (12,34), distance from 1st cluster is 22.803
# here centroids are not updated

    

# update stage    (centroids will updated)
import copy
old_centroids=copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0]=np.mean(df[df['closest']==i]['x'])
        centroids[i][1]=np.mean(df[df['closest']==i]['y'])
    return k

centroids=update(centroids)

fig=plt.figure(figsize=(5,5))
ax=plt.axes()
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
    
plt.xlim(0,100)
plt.ylim(0,100)

for i in old_centroids.keys():
    old_x=old_centroids[i][0]
    old_y=old_centroids[i][1]
    dx=(centroids[i][0]-old_centroids[i][0])*0.75
    dy=(centroids[i][1]-old_centroids[i][1])*0.75

    ax.arrow(old_x,old_y,dx,dy,head_width=2,head_length=2,fc=colmap[i],ec=colmap[i])
plt.show()  # (centroids will updated) centroids are shifted in the middle of cluster




## repeat assignment stage
df=assignment(df,centroids)
print(df.head())
# plt results
fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')

for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
    
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()   # soe points colors changed 

                   
                        
  ## continue untill all assigned categories don't change any more

while True:
    closest_centroids=df['closest'].copy(deep=True)
    centroids=update(centroids)
    df=assignment(df,centroids)
    if closest_centroids.equals(df['closest']):
        break
    
fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')

for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
    
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()   ##  compare old and this result here some dota are color changed and it is the final output
    # it is the output of the three clusters how arranged
    
    
    
# repeat this using sckit learn
df=pd.DataFrame({'x':[12,20,28,18,29,33,24,45,45,52,51,52,55,53,55,23,56,54,67],
                 'y':[34,54,23,67,89,76,54,32,12,34,54,56,78,90,65,43,23,45,67]})

from sklearn.cluster import KMeans
Kmeans= KMeans(n_clusters=3)
Kmeans.fit(df)                
###########################################

labels=Kmeans.predict(df)
centroids=Kmeans.cluster_centers_

fig=plt.figure(figsize=(5,5))
colors=map(lambda x:colmap[x+1],labels)
colors1=list(colors)
plt.scatter(df['x'],df['y'],color=colors1,alpha=0.5,edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid,color=colmap[idx+1])
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()    ##  colors are in diferent order





                           ELBOW METHOD


from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([12,20,28,18,29,33,24,45,45,52,51,52,55,53,55,23,56,54,67])
x2 = np.array([34,54,23,67,89,76,54,32,12,34,54,56,78,90,65,43,23,45,67])

plt.plot()
plt.xlim([0, 100])
plt.ylim([0, 100])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

















 
 
 
 
 
 
 
 
 
 