import numpy as np
from decimal import *
getcontext().prec = 10

def levenshtein_ratio_and_distance(s,t,ratio_calc=False):
    """ levenshtein _ratio_and_distance:
        calculates levenshtein distance between two strings.
        if ratio_calc=true the function computes the levenshtein distance ratio 
        of similarity between two strings 
        For all i and j, distance [i,j] will contain the levenshtein 
        distance between the first i charachter of s the first
        characters j of t
    """
    # initialize atrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance= np.zeros((rows, cols), dtype = int)

    # populate matrix of zeroes with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k
    
    # Iterate over the matrix to compute the cost of deletions , insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # if the character are the same in the two strings in a given position [i,j] then the cost is 0
            else :
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of substitution is 2. if we calculate just the distance, the cost of substitution is 1. 
                if ratio_calc == True :
                    cost = 2
                else :
                    cost = 1
            distance[row][col] = min(distance[row-1][col]+1,
                                distance[row][col-1]+1,
                                distance[row-1][col-1]+ cost)
    if ratio_calc == True :
        # Computation of the Levenshtein Distance Ratio
        print((len(s)+ len(t))- distance[row][col])
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+ len(t))
        return Ratio
    else :
        return "The strings are {} edits away".format(distance[row][col])


# Use fuzzy string
str1 = "Apple Inc."
str2 = "Apple Inc"

Distance = levenshtein_ratio_and_distance(str1, str2)
print(Distance)

Ratio = levenshtein_ratio_and_distance(str1, str2, ratio_calc = True)
print(Ratio)



# source : https://www.datacamp.com/community/tutorials/fuzzy-string-python