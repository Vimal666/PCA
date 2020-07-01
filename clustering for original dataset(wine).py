# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:57:25 2020

@author: Vimal PM
"""

######Performing cluster for original dataset########
#importing necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#First I would like to do the KMEANS clustering
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 
#imporing the dataset using pd.read_csv()
wine=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\ASSIGNMENT9\wine.csv")
wine.columns
Index(['Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
       'Dilution', 'Proline']
wine.head()
 Type  Alcohol  Malic   Ash  ...  Color   Hue  Dilution  Proline
0     1    14.23   1.71  2.43  ...   5.64  1.04      3.92     1065
1     1    13.20   1.78  2.14  ...   4.38  1.05      3.40     1050
2     1    13.16   2.36  2.67  ...   5.68  1.03      3.17     1185
3     1    14.37   1.95  2.50  ...   7.80  0.86      3.45     1480
4     1    13.24   2.59  2.87  ...   4.32  1.04      2.93      735

#Normalizing data's for making unitless and scale free
def norm_func(i):
    x=(i-i.min()) / (i.max()-i.min())
    return (x)


dfnorm=norm_func(wine.iloc[:,:])
dfnorm.describe()

             Type     Alcohol       Malic  ...         Hue    Dilution      Proline
count  178.000000  178.000000  178.000000  ...  178.000000  178.000000   178.000000
mean     0.604869    1.226855    1.468762  ...    0.196748    1.024185   468.727782
std      0.775035    0.811827    1.117146  ...    0.228572    0.709990   314.907474
min     -0.333333   -0.743763   -0.127586  ...   -0.280702   -0.317500    -0.165476
25%     -0.333333    0.588737    0.734914  ...    0.021798    0.350000   222.334524
50%      0.666667    1.276237    0.997414  ...    0.204298    1.192500   395.334524
75%      1.666667    1.903737    2.214914  ...    0.359298    1.582500   706.834524
max      1.666667    3.056237    4.932414  ...    0.949298    2.412500  1401.834524
dfnorm.head()#first 5 rows of my normalized dataset
 Type   Alcohol     Malic  ...       Hue  Dilution      Proline
0 -0.333333  2.456237  0.842414  ...  0.279298    2.3325   786.834524
1 -0.333333  1.426237  0.912414  ...  0.289298    1.8125   771.834524
2 -0.333333  1.386237  1.492414  ...  0.269298    1.5825   906.834524
3 -0.333333  2.596237  1.082414  ...  0.099298    1.8625  1201.834524
4 -0.333333  1.466237  1.722414  ...  0.279298    1.3425   456.834524
k=list(range(2,15))#here i'm defining my clusters range randomly from 2 to 15

#Next I need to identify the total sum of square using TWSS[] 
TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(dfnorm)
    WSS=[]#With in sum of squares
    for j in range(i):
         WSS.append(sum(cdist(dfnorm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,dfnorm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
#Here I have to deal with 4000 observation so I can't sellect my clusters number
#Therefor I'm going for screeplot to identify the elbo point
plt.plot(k,TWSS,"ro-");plt.xlabel("No_of_clusters");plt.ylabel("total_within_ss");plt.xticks(k)    
#From the screeplot I can see the Elbo point lying on 12th datapoint.
model=KMeans(n_clusters=12)
model.fit(dfnorm)
#KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
     #  n_clusters=12, n_init=10, n_jobs=None, precompute_distances='auto',
     #  random_state=None, tol=0.0001, verbose=0)
#getting the labels of cluster
model.labels_
array([ 2,  2,  9,  1, 11,  1,  4,  4,  2,  2,  1,  4,  4,  9,  1,  4,  4,
        9, 10,  5, 11, 11,  2,  2,  5,  5,  9,  4,  5,  2,  4,  1,  2,  4,
        2,  5,  5,  9,  2, 11, 11,  2,  2,  0,  5,  2,  2,  2,  2,  4,  9,
        4,  9,  4,  2,  9,  2,  4,  4,  8,  0,  3,  0,  3,  7,  0,  8,  8,
       11, 11,  5,  3,  8,  2,  5,  3,  3,  8, 11,  8,  7, 11,  0,  8,  8,
        3,  8,  6,  0,  0,  8,  3,  8,  7,  7,  5,  0,  3,  0,  3, 11,  6,
        3,  3,  0,  7,  8,  8,  7,  0,  6,  7,  6,  3,  3,  3,  8,  7,  7,
        6,  0,  8,  7,  3,  3,  3,  7,  8,  7,  6,  0,  8,  6,  6,  0,  0,
       11,  8,  6,  6,  6, 11,  8,  6,  5,  5,  3,  0,  0,  6,  8,  8,  3,
        0,  0, 11,  8,  5,  0,  0,  8,  0,  6,  0,  6,  8,  0,  0, 11,  0,
        8,  8,  0, 11, 11,  5,  5,  6])
MD=pd.Series(model.labels_)
MD.head
1       2
2       9
3       1
4      11
       ..
173    11
174    11
175     5
176     5
177     6


########performing Heirarchical cluster######
from scipy.cluster.hierarchy import linkage#here I'm importing the linkage function from hierarchy of cluster from scipy module
#for seeing dendrogram i'm going to hierarchy as sch
import scipy.cluster.hierarchy as sch
C=linkage(dfnorm,method="complete",metric="Euclidean")
plt.figure(figsize=(15,5));plt.title("hierarchical clustering dendogram");plt.xlabel("index");plt.ylabel("distance")
sch.dendrogram(
        C,
        leaf_rotation=0.,
        leaf_font_size=8.,
)
plt.show()
#Before I was calculated cluster number as 12 from the screeplot,Here also I'm going to use this same number of clusters as 10
#Impoting the agglomerative clustering for how many clusters that we need to see or cut
from sklearn.cluster import AgglomerativeClustering
C_linkage=AgglomerativeClustering(n_clusters=12,linkage="complete",affinity="euclidean").fit(dfnorm)
#labels of clusters
C_linkage.labels_

        5, 10,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 10, 10,  5,  5,
        5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 10, 10, 10,
        5, 10,  5,  5, 10,  5,  5, 10,  9,  3,  3, 11,  0,  4,  4,  0, 11,
        3,  0,  3,  1, 11,  1,  0,  3, 11,  3,  0,  8, 11, 11,  4,  3,  0,
       11,  4,  4,  4,  4,  3,  3,  3, 11, 11,  0,  2, 11,  0,  4, 11, 11,
       11, 11, 11,  4, 11,  3, 11,  0,  0, 11,  4,  4, 11,  4, 11, 11,  3,
        8,  0,  1,  8,  8,  8, 11, 11,  4, 11,  8,  2,  2,  2,  2,  3,  3,
        7,  7,  7,  3,  3,  3,  3,  7,  7,  3,  7,  7,  7,  7,  2,  2,  2,
        7,  3,  7,  7,  7,  6,  6,  7,  3,  3,  3,  7,  7,  7,  7,  7,  2,
        3,  7,  7,  7,  7,  7,  7,  7]
cluster_labels=pd.Series(C_linkage.labels_)#here transforming the series of labels to a new a dataset called "cluster_labels"
#So in the conclusion I can say from the screeplot that number of clusters for original dataset and principle component analysis both are same
#number clusters for original dataset=12
#number of clusters for principle components=12
