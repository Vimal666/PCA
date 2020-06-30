# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:25:16 2020

@author: Vimal PM
"""

#Importing necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 

#loading dataset using pd.read_csv()
wine=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\ASSIGNMENT9\wine.csv")
wine.shape
#(178, 14)
#columns names
wine.columns
#Index(['Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       #'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
      # 'Dilution', 'Proline']
wine.head
<bound method NDFrame.head of      Type  Alcohol  Malic   Ash  ...  Color   Hue  Dilution  Proline
0       1    14.23   1.71  2.43  ...   5.64  1.04      3.92     1065
1       1    13.20   1.78  2.14  ...   4.38  1.05      3.40     1050
2       1    13.16   2.36  2.67  ...   5.68  1.03      3.17     1185
3       1    14.37   1.95  2.50  ...   7.80  0.86      3.45     1480
4       1    13.24   2.59  2.87  ...   4.32  1.04      2.93      735
..    ...      ...    ...   ...  ...    ...   ...       ...      ...
173     3    13.71   5.65  2.45  ...   7.70  0.64      1.74      740
174     3    13.40   3.91  2.48  ...   7.30  0.70      1.56      750
175     3    13.27   4.28  2.26  ...  10.20  0.59      1.56      835
176     3    13.17   2.59  2.37  ...   9.30  0.60      1.62      840
177     3    14.13   4.10  2.74  ...   9.20  0.61      1.60      560
#getting mean,median,mode and the variance and  standard deviations using describe()
wine.describe()
             Type     Alcohol       Malic  ...         Hue    Dilution      Proline
count  178.000000  178.000000  178.000000  ...  178.000000  178.000000   178.000000
mean     1.938202   13.000618    2.336348  ...    0.957449    2.611685   746.893258
std      0.775035    0.811827    1.117146  ...    0.228572    0.709990   314.907474
min      1.000000   11.030000    0.740000  ...    0.480000    1.270000   278.000000
25%      1.000000   12.362500    1.602500  ...    0.782500    1.937500   500.500000
50%      2.000000   13.050000    1.865000  ...    0.965000    2.780000   673.500000
75%      3.000000   13.677500    3.082500  ...    1.120000    3.170000   985.000000
max      3.000000   14.830000    5.800000  ...    1.710000    4.000000  1680.000000
#Standardizing the dataset to make our data unitless and scale free using scale()
wine_norm=scale(wine)
#Next I'm going to apply PCA to reduce dimension of my dataset
pca=PCA(n_components=14)
pca_values=pca.fit_transform(wine_norm)
#Above I have build the 14 principal components for each and every variables
#Next I would like to see how many amount of variance contains in each and every principle components
var=pca.explained_variance_ratio_
var
([0.39542486, 0.17836259, 0.10329102, 0.06627984, 0.06267875,
       0.0480556 , 0.03955707, 0.02500244, 0.02103871, 0.01873615,
       0.01613203, 0.01205691, 0.00925458, 0.00412945])
#Weight of the principle components
pca.components_[0]
#From above analysis see,First PC1 conatains 39% of data's and PC2 contains 17% of data's, PC3 contains 10% of data's etc... 
#Next I'm going for cumilative variance
variance=np.cumsum(np.round(var,decimals=4)*100)
variance
#([ 39.54,  57.38,  67.71,  74.34,  80.61,  85.42,  89.38,  91.88,
     #   93.98,  95.85,  97.46,  98.67,  99.6 , 100.01])
#Variance plot for PCA components
plt.plot(variance,color="blue")
#Next I would like to plot b/w PC1 and PC2
x=pca_values[:,0]
y=pca_values[:,1]
plt.scatter(x,y,color=["red"])
#Next i'm going for Kmeans clustering
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 
#In question they are asking to take first 3 principle composition(PC1,PC2,PC3)
#for that creating a dataframe for my pca_values dataset
newdf=pd.DataFrame(pca_values[:,0:3])#here I'm adding only the first 3 principle composition
#Above I have taken 67.17% of my dataset from first 3 Principle composition
k=list(range(1,15))#here I'm defining my cluster range from 1 to 14
#Next I have to calcuate the sum of squares using twss[]
TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(newdf)
    WSS=[]#With in sum of squares
    for j in range(i):
         WSS.append(sum(cdist(newdf.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,newdf.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
#Therefor I'm going for screeplot to identify the elbo point
plt.plot(k,TWSS,"ro-");plt.xlabel("No_of_clusters");plt.ylabel("total_within_ss");plt.xticks(k)    
#from the screeplot I can see number 11 is my Elbo point
#There for I'm going to choose 11 clusters for my first 3 principle components
model=KMeans(n_clusters=11)
model.fit(newdf)
#KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
      # n_clusters=11, n_init=10, n_jobs=None, precompute_distances='auto',
      # random_state=None, tol=0.0001, verbose=0)
#getting the labels of cluster
model.labels_
array([ 9,  4,  1,  9,  7,  9,  9,  1,  9,  9,  9,  4,  4,  9,  9,  1,  1,
        1,  9,  1,  9,  1,  4,  4,  1,  7,  1,  4,  1,  4,  1,  9,  4,  1,
        1,  1,  1,  4,  4,  9,  9,  4,  9,  4,  4,  1,  9,  9,  1,  9,  9,
        9,  9,  1,  9,  1,  9,  1,  9,  6,  8,  8,  6,  3,  2,  3,  6,  6,
        8,  4,  8,  3,  8,  7,  3,  6,  6,  8,  4,  3,  6,  3,  2, 10,  3,
        3,  2,  2,  2,  2,  8,  2,  8,  3,  3,  1,  2,  6,  4,  3,  6,  6,
        2,  8,  8,  2,  8,  8,  8,  3,  3,  8,  2,  2,  2,  2,  8,  2,  0,
        8,  3,  7,  2,  8,  3,  8,  3,  2,  2,  2,  0,  0,  0,  0,  0,  0,
       10, 10,  0, 10, 10,  0, 10,  0,  0,  0,  0, 10,  5,  5,  5,  5, 10,
        5,  0,  5,  5, 10,  5,  5, 10,  5, 10,  0,  0,  0,  5,  0,  5,  5,
        0,  0,  5,  5,  5,  5,  5,  5])
MD=pd.Series(model.labels_)
MD.head
<bound method NDFrame.head of 0      9
1      4
2      1
3      9
4      7
      ..
173    5
174    5
175    5
176    5
177    5

#Hierarchical clustering
from scipy.cluster.hierarchy import linkage#here I'm importing the linkage function from hierarchy of cluster from scipy module
#for seeing dendrogram i'm going to hierarchy as sch
import scipy.cluster.hierarchy as sch
C=linkage(newdf,method="complete",metric="Euclidean")
plt.figure(figsize=(15,5));plt.title("hierarchical clustering dendogram");plt.xlabel("index");plt.ylabel("distance")
sch.dendrogram(
        C,
        leaf_rotation=0.,
        leaf_font_size=8.,
)
plt.show()
#Before I was calculated cluster number as 10 from the screeplot,Here also I'm going to use this same number of clusters as 10
#Impoting the agglomerative clustering for how many clusters that we need to see or cut
from sklearn.cluster import AgglomerativeClustering
C_linkage=AgglomerativeClustering(n_clusters=10,linkage="complete",affinity="euclidean").fit(newdf)
#labels of clusters
C_linkage.labels_
array([1, 3, 1, 1, 4, 1, 3, 1, 3, 3, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3,
       3, 3, 3, 4, 1, 3, 1, 3, 1, 1, 3, 1, 3, 3, 3, 3, 8, 1, 1, 3, 1, 3,
       3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 7, 9, 9, 8, 5, 0, 0,
       8, 8, 9, 8, 9, 5, 0, 4, 5, 8, 8, 0, 3, 5, 8, 5, 0, 2, 0, 5, 0, 0,
       0, 0, 0, 0, 0, 5, 5, 3, 0, 8, 5, 5, 8, 8, 0, 8, 8, 0, 8, 0, 8, 5,
       5, 8, 0, 0, 0, 5, 8, 0, 9, 8, 5, 4, 0, 0, 5, 0, 0, 0, 0, 0, 9, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 6, 2,
       2, 2, 2, 2, 6, 6, 2, 2, 2, 2, 2, 2, 6, 2, 6, 6, 2, 2, 2, 2, 2, 2,
       2, 6], dtype=int64)
cluster_labels=pd.Series(C_linkage.labels_)#here transforming the series of labels to a new a dataset called "cluster_labels"
