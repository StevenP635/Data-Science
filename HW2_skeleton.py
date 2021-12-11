
# -*- coding: utf-8 -*-
"""
HW1 Assignment
@author: Steven Paredes
@abc123: obr635
@date: 9/17/2020
"""
import math
import matplotlib.pyplot as plt
from functools import reduce
from IPython import get_ipython
from collections import OrderedDict 
import statistics

# barely works
def merge(sortListA, sortListB):
    """Given two non-decreasingly sorted list of numbers, 
       return a single merged list in non-decreasing order
    """
    mergedList = []
    x = y = 0
    while x < len(sortListA) and y < len(sortListB):
        if sortListA[x] < sortListB[y]:
            mergedList.append(sortListA[x])
            x += 1
        else:
            mergedList.append(sortListB[y])
            y += 1
    mergedList += sortListA[x:]
    mergedList += sortListB[y:]
 
    return mergedList

# barely works
def mergeSort(numList):
    """
    Given a list of numbers in random order, 
    return a new list sorted in non-decreasing order, 
    and leave the original list unchanged.
    """
    if len(numList) < 2:
        return numList
    middle = len(numList)//2
    
    leftList =  mergeSort(numList[:middle])
    rightList = mergeSort(numList[middle:])
  
    return merge(leftList, rightList)

# works
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

    sortedArray = mergeSort(listOfNums)

    return {'max': sortedArray[-1], 
            'min': sortedArray[0], 
            'median': median(sortedArray),
            'mean': mean(listOfNums), 
            'std': std(listOfNums, True),
            '25%': prctile(sortedArray, 25),
            '75%': prctile(sortedArray, 75)
            }

def rescaleToInt(listOfNums, maxInt): 
    """Given a list of real numbers in any range, scale them to be integers
    between 0 and maxInt (inclusive). For each number x in the list, the new number
    is computed with the formula round(maxInt * (x-min) / (max-min)), where min and 
    max represent the minimum and maximum value of the list, respectively.
    """    
    temp = summaryStatistics(listOfNums)
    max = temp['max']
    min = temp['min']
    z = 0
    for x in listOfNums:
        listOfNums[z] = round(maxInt * (x-min) / (max-min))
        z+=1
    return listOfNums
    
def myHistWithRescale(listOfNums, maxInt):
    """Givne a list of real numbers in any range, first scale the numbers to
    inters between 0 and maxInt (inclusive), then return the number of occurrences
    of each integer in a list
    """
    
    rescaledData = rescaleToInt(listOfNums, maxInt)
    
    # complete the following part to count the number of occurrences of 
    # each digit in scaledData
    counts = dict()
    for x in rescaledData:
        # If element exists in dict then increment its value else add it in dict
        if x in counts.keys():
            counts[x] += 1
        else:
            counts[x] = 1 

    orderedCounts = OrderedDict(sorted(counts.items()))   

    return orderedCounts


## This if statement makes sure that the following code will be executed only 
## if you are running it as a script (rather than loading it as a module).
if __name__ == '__main__':
   
    import random
    
    # Test merge sort
    
    random.seed(0)
    
    a = [random.randint(0, 20) for _ in range(10)]
    
    b = mergeSort(a)
    
    print('random array is: ', a)
    print('sorted array is: ', b)
    print('mergeSort is', 'correct' if b == sorted(a) else 'incorrect')
    print('profiling running time of python sorted function on a')

    get_ipython().run_line_magic('timeit', 'sorted(a)')

    print('profiling running time of my merge sort function on a')
    get_ipython().run_line_magic('timeit', 'mergeSort(a)')
    

    # Generate three sets of random data
    
    listA = [10*random.random() for _ in range(10000)]
    listB = [random.gauss(4, 2) for _ in range(10000)]
    listC = [math.exp(random.gauss(1.5, 0.2)) for _ in range(10000)]


    print('profiling running time of python sorted function on listA')
    get_ipython().run_line_magic('timeit', 'sorted(listA)')

    print('profiling running time of my merge sort function on listA')
    get_ipython().run_line_magic('timeit', 'mergeSort(listA)')
    
    
    # calculates the summary statistics 
    ssA = summaryStatistics(listA)   
    ssB = summaryStatistics(listB)
    ssC = summaryStatistics(listC)    
            
    print("\nSummary statistics for data set A: ")
    for pair in ssA.items():
        print(pair[0] + ': %.3f ' % (pair[1]))

    print("\nSummary statistics for data set B: ")
    for pair in ssB.items():
        print(pair[0] + ': %.3f ' % (pair[1]))
            
    print("\nSummary statistics for data set C: ")
    for pair in ssC.items():
        print(pair[0] + ': %.3f ' % (pair[1]))
    
    print('\n')
        
    ## get the dictionary ready to plot

    ssA['mean+std'] = ssA['mean']+ssA['std']
    ssB['mean+std'] = ssB['mean']+ssB['std']
    ssC['mean+std'] = ssC['mean']+ssC['std']

    ssA['mean-std'] = ssA['mean']-ssA['std']
    ssB['mean-std'] = ssB['mean']-ssB['std']
    ssC['mean-std'] = ssC['mean']-ssC['std']
    
    del ssA['std']
    del ssB['std']
    del ssC['std']

    
### complete code below to plot the summary statistics using Fig 1 in HW1 as a template
    plt.close('all')
    plt.figure(1)
    
    dataSet = ['Data A', 'Data B', 'Data C']
    
    allMax = [ssA['max'], ssB['max'], ssC['max']]
    plt.scatter(dataSet, allMax, c="red", marker="v")
    
    allMin = [ssA['min'], ssB['min'], ssC['min']]
    plt.scatter(dataSet, allMin, c="red", marker="^")
    
    allMed = [ssA['median'], ssB['median'], ssC['median']]
    plt.scatter(dataSet, allMed, c="black", marker="x")
    
    allMeans = [ssA['mean'], ssB['mean'], ssC['mean']]
    plt.scatter(dataSet, allMeans, c="black", marker="+")
    
    allTFP = [ssA['25%'], ssB['25%'], ssC['25%']]
    plt.scatter(dataSet, allTFP, c="blue", marker="<")
    
    allSFP = [ssA['75%'], ssB['75%'], ssC['75%']]
    plt.scatter(dataSet, allSFP, c="blue", marker=">")
    
    allMPS = [ssA['mean+std'], ssB['mean+std'], ssC['mean+std']]
    plt.scatter(dataSet, allMPS, c="green",marker="v")
    
    allMMS = [ssA['mean-std'], ssB['mean-std'], ssC['mean-std']]
    plt.scatter(dataSet, allMMS, c="green",marker="^")

    # add a title
    plt.title("Fig1: Summary Statistics")

    # add a label to the y-axis
    plt.ylabel("Value")
    plt.legend(ssA.keys(), bbox_to_anchor=(1.05, 1))

    plt.show()  
    
    
    # count the number of occurrences of each 
    # intger in resaled data
    countA = myHistWithRescale(listA, 20)
    countB = myHistWithRescale(listB, 20)
    countC = myHistWithRescale(listC, 20)

    
    print (countA)
    print (countB)
    print (countC)
    

    # complete the following code to plot the counts using Fig 2 in HW1 as template

    plt.figure(2)

    dataSet = ['Rescaled Data A','Rescaled Data B','Rescaled Data D']
    
    plt.plot(list(countA.keys()), list(countA.values()), c="red", marker="+", linestyle='dotted')
    plt.plot(list(countB.keys()), list(countB.values()), c="green", linestyle="dashdot")
    plt.plot(list(countC.keys()), list(countC.values()), c="blue", marker="x", linestyle='dashed')
    
    plt.title("Fig2: Distribution of Rescaled Data")
    plt.ylabel("Frequencey")
    plt.xlabel("Value")
    plt.legend(dataSet, bbox_to_anchor=(1.05, 1))
    

    plt.show()  
