# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ##  IDs: 313429607, 317225993
# 

# %%
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:49:38 2020

@author: ravros
"""
import numpy as np
import re
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_samples, silhouette_score


# %%
#functions defonotion
def readFile(fileName):
    file = open(fileName,'r',encoding="cp437")
    fileStr = ""
    for line in file:
        fileStr += line
    return fileStr
        
# Remove extra spaces
# Remove non-letter chars    
# Change to lower 
def preProcess(fileStr):
    fileStr = re.sub(" +"," ", fileStr)
    fileStr = re.sub("[^a-zA-Z ]","", fileStr)
    fileStr = fileStr.lower()
    return fileStr

#Divide the file in chuncks of the same size wind
def partition_str(fileStr, wind):
    n = wind
    chunks = [fileStr[i:i+n] for i in range(0, (len(fileStr)//n)*n, n)]
    #print(chunks)
    count = len(chunks)
    return chunks, count;

# %% [markdown]
# # result 1 + 2

# %%
fileContant = preProcess(readFile("text1.txt"))
#wind - chunks size 
wind = 5000
#Divide the each file into chunks of the size wind 
chunks, count = partition_str(fileContant, wind)
wordsSet =  set(fileContant.split())
stopWordsSet = set(readFile('stopwords_en.txt').split())
dictionary = wordsSet.difference(stopWordsSet)


# %%
# Count the number of dictionary words in files - Frequency Matrix
wordFrequency = np.empty((count,len(dictionary)),dtype=np.int64)
for i in range(count):
    print(i)
    for j,word in enumerate(dictionary):
        wordFrequency[i,j] = len(re.findall(word,chunks[i]))


# %%
# find the distance matrix between the text files - Distance Matrix
dist = np.empty((count,count))
for i in range(count): 
    for j in range(count):
        # calculate the distance between the frequency vectors
        dist[i,j] = np.linalg.norm(wordFrequency[i,:]-wordFrequency[j,:])
# find the sum of the frequency colomns and select colomns having sum > 100
minSum = 100
sumArray =  wordFrequency.sum(axis=0)
indexArray = np.where(sumArray > minSum)

indexArraySize = len(indexArray[0])
wordFrequency1 = np.empty((count,indexArraySize),dtype=np.int64)

# generate a frequencey file with the selected coloumns 
for j in range(indexArraySize):
    wordFrequency1[:,j] = wordFrequency[:,indexArray[0][j]]

 # find the another distance matrix between the text files 
dist1 = np.empty((count,count))
for i in range(count): 
    for j in range(count):
        dist1[i,j] = np.linalg.norm(wordFrequency1[i,:]-wordFrequency1[j,:])
   
np.save('dist2',dist1,allow_pickle = True)  

# %% [markdown]
# # result 4

# %%
#finction clust
def clust(dist,n_cl):
 
#cluster the data into k clusters, specify the k  
    kmeans = KMeans(n_clusters = n_cl)
    kmeans.fit(dist)
    #labels_ = best_label // its the symbol for each point (vector) to which center 
    #from couple of seeds and its detail the number cluster
    labels = kmeans.labels_ +1
    # its will be shaped like [1,46(data vectors)] something like this yes 
#show the clustering results  
    fig = plt.figure()
    # defines the size of the plot in squares where [0,0,1,1] will be a regular plot
    ax = fig.add_axes([0,0,1,1])
    ax.bar(range(len(labels)),labels)
    plt.show()

# calculate the silhouette values  
    silhouette_avg_ = silhouette_score(dist, labels)
    sample_silhouette_values_ = silhouette_samples(dist, labels)
    print(silhouette_avg_)
# show the silhouette values 
    plt.plot(sample_silhouette_values_) 
    plt.plot([silhouette_avg_]*46, 'r--') #useless line
    plt.title("The silhouette plot for the various vectors.")
    plt.xlabel("data number ")
    plt.ylabel("silhouette value for each value")
    y=silhouette_avg_
    xmin=0
    xmax=len(labels)
# The vertical line for average silhouette score of all the values
    plt.hlines(y, xmin, xmax, colors='red', linestyles="--") 
    plt.show()

    print("For n_clusters =", n_cl,
      "The average silhouette_score is:", silhouette_avg_)
    return labels

# %% [markdown]
# # result 5 + 6 

# %%
dist = np.load('dist2.npy')
labels = clust(dist1, 2)
lab = labels


# %%
labels = clust(dist1, 3)


# %%
labels = clust(dist1, 4)

# %% [markdown]
# # result 7
# after looking at the results we noticed that k=2 has an avg_silhoute = ~0.47 which is the highest among the Ks that we tested.$\\$
# that's why there are probably 2 books.
# %% [markdown]
# # result 8 
# in chunk 83 was the first time we noticed the begining of the second book and after 415000 charecters

# %%
strin = ""
for j,item in enumerate(lab):
   print("("+str(j)+","+str(item)+")")


# %%
83*5000

# %% [markdown]
# # result 9

# %%
def bookDetection(wind):
    prev = lab[0]
    counter = 0
    l = []
    prev_index = 0
    for j,item in enumerate(lab):
        if item != prev:
            counter += 1
        else:
            counter = 0
        if counter == 5:
            l.append((prev_index*wind,(j - 4)*wind))
            prev_index = j-4
            counter = 0
            prev = item
    l.append((prev_index*wind,j*wind))
    return l


# %%
bookDetection(wind)

# %% [markdown]
# the function returns list of tuples where each tuple has two elements , the first one is the begining of the book and the second one is the ending of the book 
# 
# and we can see that the function doesn't even need the k we found and can automaticly detect each book section and return a list of books.

