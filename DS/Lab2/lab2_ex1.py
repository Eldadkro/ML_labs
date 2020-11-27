# import regular expressins packge
# import numbers package
import numpy as np
import re

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

         
rows = 5
fileContent = [""]*rows
#read  and preprocess files 
fileContent[0] = preProcess(readFile('DB.txt'))
fileContent[1] = preProcess(readFile('HP_small.txt'))
fileContent2 = preProcess(readFile('Tolkien.txt'))
numParts = 3
# split the third file to parts
partLength = int(len(fileContent2)/numParts) 
fileContent[2]  = fileContent2[0:partLength]
fileContent[3]  = fileContent2[partLength:partLength*2]
fileContent[4]  = fileContent2[partLength*2:partLength*3]
#___________________________________ 
# construct DICTIONARY concat files contents
numFiles = rows
allFilesStr = ""
for i in range(numFiles):
    allFilesStr += fileContent[i]
# generate a set of all words in files 
wordsSet =  set(allFilesStr.split())
# Read stop words file - words that can be removed
stopWordsSet = set(readFile('stopwords_en.txt').split())
# Remove the stop words from the word list
dictionary = wordsSet.difference(stopWordsSet)
#_______________________________________
# count the number of dictionary words in files
wordFrequency = np.empty((rows,len(dictionary)),dtype=np.int64)
for i in range(rows):
    for j,word in enumerate(dictionary):
        wordFrequency[i,j] = len(re.findall(word,fileContent[i]))
# find the distance matrix between the text files
dist = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        # calculate the distance between the frequency vectors
        dist[i,j] = np.linalg.norm(wordFrequency[i,:]-wordFrequency[j,:])
print("dist=\n",dist)
# find the sum of the frequency colomns and select colomns having sum > 20
minSum = 20
sumArray =  wordFrequency.sum(axis=0)
indexArray = np.where(sumArray > minSum)
indexes = indexArray[0]
wordFrequency2 = np.zeros((rows,len(indexes)),dtype=np.int64)
for j,index in enumerate(indexArray[0]):
    wordFrequency2[:,j] = wordFrequency[:,index]

dist2 = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        # calculate the distance between the frequency vectors
        dist2[i,j] = np.linalg.norm(wordFrequency2[i,:]-wordFrequency2[j,:])


fileContent_eliot = preProcess(readFile("Eliot.txt"))
fileContent_Tolkin = preProcess(readFile("Tolkien.txt"))

readSize = len(fileContent_eliot)//2
fileContent.append(fileContent_eliot[:readSize])
fileContent.append(fileContent_eliot[readSize:])

readSize = len(fileContent_Tolkin)//4
fileContent.append(fileContent_Tolkin[:readSize])
fileContent.append(fileContent_Tolkin[readSize:2*readSize])
fileContent.append(fileContent_Tolkin[2*readSize:3*readSize])
fileContent.append(fileContent_Tolkin[3*readSize:])

fileContent = fileContent[6:]

rows = len(fileContent)

wordFrequency = np.empty((rows,len(dictionary)),dtype=np.int64)
for i in range(rows):
    for j,word in enumerate(dictionary):
        wordFrequency[i,j] = len(re.findall(word,fileContent[i]))
# find the distance matrix between the text files
dist = np.empty((rows,rows))
for i in range(rows): 
    for j in range(rows):
        # calculate the distance between the frequency vectors
        dist[i,j] = np.linalg.norm(wordFrequency[i,:]-wordFrequency[j,:])

print("dist=\n",dist)








