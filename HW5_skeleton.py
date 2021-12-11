# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 00:42:10 2019
HW5 code skeleton
@author: Steven Paredes
@abc123: obr635
@Date: 
"""
# Dont share this one you did awful on it

#Note: This assignment is tricky to say the least :(
#%% import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import statistics

def median(sortedList): 
    return statistics.median(sortedList)
    
# works
def prctile(sortedList, perc): 
    """    
    Returns the number that is ranked perc% in the non-decreasingly sorted list. Therefore,
    prctile(sortedList, 50) should be roughly equivalent to median(sortedList), and 
    prctile (sortedList, 100) should return the max value
    """ 
    loc = (int) (len(sortedList) // (100 / perc))
    return sortedList[loc]
    

def maxValue(listOfNums):
    return reduce(lambda x, y: x if x > y else y, listOfNums)

def minValue(listOfNums):
    return reduce(lambda x, y: x if x < y else y, listOfNums)
    
def sum(numList):
    return reduce(lambda x, y: x + y, numList)

def mean(numList):
    return sum(numList) / len(numList) 

def var(listOfNums):
    m = mean(listOfNums)
    dev = [(x - m)**2 for x in listOfNums]
    return mean(dev)


def std(listOfNums, correction=False):
    if (not correction):
        return var(listOfNums)**0.5
    else:
        n = len(listOfNums)
        return (var(listOfNums) * n / float(n-1))**0.5

    
def summaryStatistics(listOfNums):
    """Givne a list of numbers in random order, return the summary statistics 
    that includes the max, min, mean, population standard deviation, median,
    75 percentile, and 25 percentile.
    """    
    # Return the following statistics in a key-value pair (i.e., dictionary)

    sortedArray = listOfNums[:]
    sortedArray.sort()

    return {'max': sortedArray[-1], 
            'min': sortedArray[0], 
            'median': median(sortedArray),
            'mean': mean(listOfNums), 
            'std': std(listOfNums, True),
            'mean+std': mean(listOfNums) + std(listOfNums, True),
            'mean-std': mean(listOfNums) - std(listOfNums, True),
            '25%': prctile(sortedArray, 25),
            '75%': prctile(sortedArray, 75)
            }

#%% Q1 setup
import math
import pandas as pd
import scipy.stats as stats
import numpy.random as rn

rn.seed(0)

#%% Q1a

m = 10**5;  # number of coins
N = 16; # number of tosses each coin

tosses = np.array(rn.rand(m,N))
headCounts = np.array(np.sum(tosses < .5, axis = 1))

# generate headCounts and plot histogram
pmfHeadCounts = plt.hist(headCounts, bins = range(18))
plt.title('Q1a: ')
plt.ylabel("Number of Coins")
plt.xlabel("HeadCount")
plt.show()


#%% Q1b
bin_center = (pmfHeadCounts[1][1:] + pmfHeadCounts[1][:-1]) / 2
plt.bar(bin_center, pmfHeadCounts[0]/sum(pmfHeadCounts[0]), width=1.5)

plt.title('Q1b: ')
plt.ylabel("P (value <= k)")
plt.xlabel("k")
plt.show()
#%% Q1c
plt.hist(np.sort(headCounts), cumulative=True, bins=len(headCounts)+1, density=True, histtype='step')

plt.title('Q1c: ')
plt.ylabel("P (value <= k)")
plt.xlabel("k")
plt.show()
#%% Q1d
result = []
for a in range(17): 
    result.append(stats.binom.cdf(a,16,.5))
    
plt.plot(result)
plt.hist(np.sort(headCounts), cumulative=True, bins=len(headCounts)+1, density=True, histtype='step')
plt.title('Q1d: ')
plt.ylabel("P (value <= k)")
plt.xlabel("k")
plt.legend(["Empirical CDF", " Binomial Distribution CDF"])
plt.show()
#%% Q1e
# you messed this up completely
counts, lowlim, binsz, _ = stats.cumfreq(headCounts, numbins=len(headCounts)+1 )
graph = np.arange(counts.size) * binsz + lowlim

plt.title('Q1e: ')
plt.plot(graph, counts, 'ro')
plt.xlabel('Value')
plt.ylabel('Cumulative Frequency')
plt.show()
#%% Q1f

plt.figure()
plt.title('Q1f: ')
plt.show()
#%% Q2a - Q2d
#Q2a, simulate fair coin and collect headCountsFair
headCountsFair = np.array(rn.rand(10,1000))
headCountsFair = np.sum(headCountsFair < .5, axis = 0)

#Q2b, simulate loadd coin and collect headCountsLoaded
headCountsLoaded = np.array(rn.rand(10,1000))
headCountsLoaded = np.sum(headCountsLoaded < .6, axis = 0)

#Q2c, plot both headCounts on the same figure
pmfHeadCountsFair = plt.hist(headCountsFair, bins = 10)
bin_center_fair = (pmfHeadCountsFair[1][1:] + pmfHeadCountsFair[1][:-1]) / 2
plt.close()

pmfHeadCountsLoaded = plt.hist(headCountsLoaded, bins = 10)
bin_center_loaded = (pmfHeadCountsLoaded[1][1:] + pmfHeadCountsLoaded[1][:-1]) / 2
plt.close()

plt.plot(bin_center_fair, pmfHeadCountsFair[0]/sum(pmfHeadCountsFair[0]))
plt.plot(bin_center_loaded, pmfHeadCountsLoaded[0]/sum(pmfHeadCountsLoaded[0]))
plt.legend(["Head Counts Fair", "Head Counts Loaded"])
plt.title('Q2c: ')
plt.ylabel("P (value <= k)")
plt.xlabel("k")
plt.show()

#Q2d, perform ttest and print out p-value
resultStat, resultT = stats.ttest_ind(headCountsFair, headCountsLoaded)
print("Q2d :")
print("T-test fair and loaded results = ",resultT)
print("This indicates that the two boxes of coins are different")
#%% Q2e Repeat Q2a, Q2b and Q2d (with only 10 coins) 10 times and count number of p-values <= 0.05

#%% Q2e Repeat Q2a, Q2b and Q2d 1000 times and count number of p-values <= 0.05



#%% Q3 setup
import pandas as pd

data = pd.read_csv('brfss.csv', index_col=0)

# data is a numpy array and the columns are age, current weight (kg), 
# last year's weight (kg), height (cm), and gender (1: male; 2: female).
data = data.drop('wtkg2',axis=1).dropna(axis=0, how='any').values
#dataStats = np.arange(24).reshape(3,8)

curWeight = data[:,1]
curWeightStats = summaryStatistics(curWeight)

prevWeight = data[:,2]
prevWeightStats = summaryStatistics(prevWeight)

height = data[:,3]
heightStats = summaryStatistics(height)

dataStats = pd.DataFrame([curWeightStats, prevWeightStats, heightStats])
dataStats.index = ["Current Weight", "Previous Weight", "Height"]

dataSet = ["Current Weight", "Previous Weight", "Height"]

plt.title('Q3a: ')

plt.scatter(dataSet, dataStats['max'], c="red", marker="v")
plt.scatter(dataSet, dataStats['min'], c="red", marker="^")
plt.scatter(dataSet, dataStats['median'], c="black", marker="x")
plt.scatter(dataSet, dataStats['mean'], c="black", marker="+")
plt.scatter(dataSet, dataStats['25%'], c="blue", marker="<")
plt.scatter(dataSet, dataStats['75%'], c="blue", marker=">")
plt.scatter(dataSet, dataStats['mean+std'], c="green",marker="v")
plt.scatter(dataSet, dataStats['mean-std'], c="green",marker="^")

plt.legend(['max', 'min', 'median', 'mean', '25%', '75%', 'mean+std', 'mean-std'], bbox_to_anchor=(1.05, 1))
plt.show()
#%% Q3b
# you messed this one up as well
age = data[:,0]
weightChange = curWeight[:] - prevWeight[:]
pccCurWeight = stats.pearsonr(weightChange, curWeight)
pccPrevWeight = stats.pearsonr(weightChange, prevWeight)
pccAge = stats.pearsonr(weightChange, age)

#plt.title('Q3b: Pearson Correlation')
#plt.scatter(curWeight, weightChange)
#plt.scatter(prevWeight, weightChange)
#plt.scatter(age, weightChange)

plt.title('Q3b: Weight Change vs Current Weight (pcc = ' + str(pccCurWeight[0]) + ')')
plt.scatter(weightChange, curWeight)
plt.ylabel("Current Weight")
plt.xlabel("Weight Change")
plt.show()

plt.title('Q3b: Weight Change vs Previous Weight (pcc = ' + str(pccPrevWeight[0]) + ') ')
plt.scatter(weightChange, prevWeight)
plt.ylabel("Previous Weight")
plt.xlabel("Weight Change")
plt.show()

plt.title('Q3b: Weight Change vs Age (pcc = ' + str(pccAge[0]) + ') ')
plt.scatter(weightChange, age)
plt.ylabel("Age")
plt.xlabel("Weight Change")
plt.show()

print("\nQ3b :")
print("From the data that I had generated and processed I believe that weight change does not strongly correlate with any of the data. That being said weight change has the strongest correlation with age at a whopping pcc = " + str(pccAge[0]))
#%% Q3c
dataMale = data[data[:,4] == 1]
maleWeightChange = dataMale[:,1] - dataMale[:,2]
meanMaleWeightChange = np.mean(maleWeightChange)
semMWC = np.std(maleWeightChange)/math.sqrt(maleWeightChange.shape[0])

print("\nQ3c :")
print("Mean for male weight change: ", meanMaleWeightChange)
print("SEM for male weight change: ",semMWC)

dataFemale = data[data[:,4] == 2]
femaleWeightChange = dataFemale[:,1] - dataFemale[:,2]
meanFemaleWeightChange = np.mean(femaleWeightChange)
semWWC = np.std(femaleWeightChange)/math.sqrt(femaleWeightChange.shape[0])

print("Mean for female weight change: ", meanFemaleWeightChange)
print("SEM for female weight change: ",semWWC)

resultStat, resultP = stats.ttest_ind(femaleWeightChange, maleWeightChange)

print("T-test male and female weight change pvalue = ", resultP)
#%% Q3d
# messed this one up too
maleWeightHeightRatio = dataMale[:,1] / dataMale[:,3]
femaleWeightHeightRatio = dataFemale[:,1] / dataFemale[:,3]

semMWHR = np.std(maleWeightHeightRatio) / math.sqrt(maleWeightHeightRatio.shape[0])
semWWHR = np.std(femaleWeightHeightRatio) / math.sqrt(femaleWeightHeightRatio.shape[0])

print("\nQ3d : ")
print("SEM male weight/height ratio: ", semMWHR)
print("SEM female weight/height ratio: ", semWWHR)

resultStat, resultP = stats.ttest_ind(maleWeightHeightRatio, femaleWeightHeightRatio)
print("T-test of male and female weight/height ratio pvalue = ", resultP)
#%% Q3e
