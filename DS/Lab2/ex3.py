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


fileContent_eliot = preProcess(readFile("Eliot.txt"))
fileContent_Tolkin = preProcess(readFile("Tolkien.txt"))

readSize = len(fileContent_eliot)//2
fileContent = []
fileContent.append(fileContent_eliot[:readSize])
fileContent.append(fileContent_eliot[readSize:])

readSize = len(fileContent_Tolkin)//4
fileContent.append(fileContent_Tolkin[:readSize])
fileContent.append(fileContent_Tolkin[readSize:2*readSize])
fileContent.append(fileContent_Tolkin[2*readSize:3*readSize])
fileContent.append(fileContent_Tolkin[3*readSize:])

rows = len(fileContent)

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