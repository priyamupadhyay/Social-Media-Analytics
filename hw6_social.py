"""
Social Media Analytics Project
Name:
Roll Number:
"""
from ast import operator


import operator
from collections import Counter
import re
import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df = pd.read_csv(filename)
    return df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    ret = fromString.replace("From: ","",1)
    f1 = ret.find("(")
    f2 = ret.find(")")
    if len(ret)>f2:
        ret = ret[0:f1-1:] + ret[f2+1::]
    return ret


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    f1 = fromString.find("From:")
    f2 = fromString.find("(")
    if len(fromString)>f2:
        fromString = fromString[0:f1:] + fromString[f2+1::]
        f3 = fromString.find(" from")
        f4 = fromString.find(")")
        fromString = fromString[0:f3:] + fromString[f4+1::]
    return fromString


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    f1 = fromString.find("From:")
    f2 = fromString.find("from ")
    if len(fromString)>f2:
        fromString = fromString[0:f1:] + fromString[f2+5::]
    return fromString.replace(")","",1)


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    line = message.split("#")
    s = []
    v = ""
    for i in range(1,len(line)):
        for j in line[i]:
            if j in endChars:
                break
            else:
                v += j
        v = "#" + v
        s.append(v)
        v = ""
    return s


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    #d2=stateDf.query('state == state')['region']
    d2=stateDf.loc[stateDf['state'] == state, 'region'].iloc[0]
    #print(d2)
    return d2


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names = []
    positions = []
    states = []
    region = []
    hashtags = []
    for row_index,row in data.iterrows():
        col = row["label"]
        names.append(parseName(col))
        positions.append(parsePosition(col))
        states.append(parseState(col))
        region.append(getRegionFromState(stateDf, parseState(col)))
        val1 = row["text"]
        hashtags.append(findHashtags(val1))
    data["name"] = names
    data["position"] = positions
    data["state"] = states
    data["region"] = region
    data["hashtags"] = hashtags
    #print(names)
    return


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1:
        return "negative"
    elif score > 0.1:
        return "positive"
    else:
        return "neutral"
    return


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments = []
    for row_index,row in data.iterrows():
        col = row["text"]
        sentiments.append(findSentiment(classifier, col))
    data["sentiment"] = sentiments
    #print(data)
    return


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):  
    #count = data[colName].value_counts()[dataToCount]
    #count = data.groupby([colName]).size()
    dict1 ={}
    for index, row in data.iterrows():
        if len(colName) !=0 and len(dataToCount) !=0:
            if (row[colName] == dataToCount):
                if (row["state"] in dict1):
                    dict1[row["state"]] += 1
                else:
                    dict1[row["state"]] = 1
        else:
            if row["state"] in dict1:
                dict1[row["state"]] += 1
            else:
                dict1[row["state"]] = 1  
    return dict1


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    odict = {}
    for index, row in data.iterrows():
        if row['region'] not in odict:
            odict[row['region']] = {}
        if row[colName] not in odict[row['region']]:
            odict[row['region']][row[colName]] = 1
        else:
             odict[row['region']][row[colName]] += 1
    return odict


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    dicttag = {}
    #print(len(data["hashtags"]))
    for index, row in data.iterrows():
        tag = row["hashtags"]
        for i in range(len(tag)):
            if tag[i] not in dicttag:
                dicttag[tag[i]] = 1
            else:
                dicttag[tag[i]] += 1
    #print(dicttag)
    return dicttag


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count): 
    most_common = {} 
    s = Counter(hashtags) 
    n = count 
    sort = list(sorted(s.items(), key=operator.itemgetter(1),reverse=True))[:n] 
    for key,value in sort: 
        most_common[key] = value 
    return most_common


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    return


### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    '''print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()'''
    """test.testMakeDataFrame()
    test.testParseName()
    test.testParsePosition()
    test.testParseState()
    test.testFindHashtags()
    test.testGetRegionFromState()
    test.testAddColumns()"""
    ## Uncomment these for Week 2 ##
    '''print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()'''
    test.testFindSentiment()
    test.testAddSentimentColumn()
    df = pd.read_csv("data/politicaldata.csv")
    df1 = pd.read_csv("data/statemappings.csv")
    addColumns(df, df1)
    addSentimentColumn(df)
    test.testGetDataCountByState(df)
    test.testGetDataForRegion(df)
    test.testGetHashtagRates(df)
    test.testMostCommonHashtags(df)
    ## Uncomment these for Week 3 ##
    """print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()"""
