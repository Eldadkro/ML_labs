"""
Created on Tue Oct 13 19:36:13 2020
@author: ravros
#IRIS DATA
"""
#import libraries
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt 
#import pandas as pd 
from pandas import DataFrame
from sklearn.metrics import silhouette_samples, silhouette_score
#import datasets 
from sklearn import datasets

def clust(df,n_cl):
    kmeans = KMeans(n_clusters = n_cl).fit(df)
    res = kmeans.labels_+1      
    centroids = kmeans.cluster_centers_

    plt.scatter(df['x1'], df['x2'], c = res, s=50, alpha = 1)

    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(range(len(res)),res)
    plt.show()
#show the silhouette values for k=3 
    silhouette_avg_ = silhouette_score(df, res)
    sample_silhouette_values_ = silhouette_samples(df, res)  
    plt.plot(sample_silhouette_values_) 
    plt.plot(silhouette_avg_, 'r--')
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    y=silhouette_avg_
    xmin=0
    xmax=len(res)
#The vertical line for average silhouette score of all the values
    plt.hlines(y, xmin, xmax, colors='red', linestyles="--") 
    plt.show()

    print("For n_clusters =", n_cl,
      "The average silhouette_score is:", silhouette_avg_)
    return res

#import datasets 
#from sklearn import datasets
Iris = datasets.load_iris()

y = Iris.data
lb = Iris.target # true labeling
y0 = np.array(y[:,0:4]) #useless as it returns nparray 

x1 = y0[:,0] 
x2 = y0[:,1]
x3 = y0[:,2] 
x4 = y0[:,3]

df = DataFrame(y0, columns=['x1','x2','x3','x4']) 

plt.plot()
plt.subplot()
plt.title('Dataset')
plt.scatter(x1,x2)
plt.show()

res3 = clust(df, 3)

#y2 = np.array(y[50:150,0:4])
#lb2=