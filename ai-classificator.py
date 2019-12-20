# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Classification Method
# 
# * Gradient Boosting
# * Multidimensional Logistic Regression
# * Super Vector Machine (SVM)
# * Multidimensional Naive Bayes
# * K Nearest Neighbors (KNN)
# * Random Forest
# * Fuzzy
# * Jaringan Syaraf Tiruan (JST) / Artificial Neural Network (ANN)
# 
# %% [markdown]
# # Fuzzy
# - The Levenshtein Distance
# 
#     The Levenshtein distance is a metric to measure how apart are two sequences of words. In other words, it measures the minimum number of edits that you need to do to change a one-word sequence into the other. These edits can be insertions, deletions or substitutions. This metric was named after Vladimir Levenshtein, who originally considered it in 1965.
# 
#     The formal definition of the Levenshtein distance between two strings a and b can be seen as follows:
# 
#     ![formula](img/Levenshtein.png)
# 
# 
#     Where 1(ai≠bj) denotes 0 when a=b and 1 otherwise. It is important to note that the rows on the minimum above correspond to a deletion, an insertion, and a substitution in that order.

# %%
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
        Ratio = 0.001
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

# %% [markdown]
# # K-Nearest Neighbor
# 
# Jarak K terdekat terhadap ciri pada suatu klasifikasi. KNN merupakan methode klasifikasi paling sederhana. KNN bekerja dengan cara menghitung jarak dari suatu ciri terhadap data ciri yang ada pada data pelatihan.
# 
# metode pengukuran jarak a dan b menggunakan persamaan :
# jarak = (a**2+b**2)**0.5
# 
# ```python
#     def euclidean_distance(self):
#         dif = data_latih - data_uji
#         for i in dif :
#             n = n + (i**2)
#         return (n**(1/2))
#         
#     def manhattan_distance(self):
#         dif = data_latih - data_uji
#         for i in dif:
#             n = n + abs(i)
#         return n
# ```
# 
# ## 1. pre-processing
# pengondisian data agar sesuai dengan bentuk yang diharapkan. data dengan ciri lebih dari 1 membentuk matriks 2 dimensi. susun data berjajar, dengan x merepresentasikan indeks data ke-1 sampai dengan ke-n dan y merepresentasikan ciri dari setiap indeks data. 
# 
# ## 2. pengukuran jarak 
# Pengukuran data dilakukan pada ciri data uji ke seluruh ciri data latih. k data terdekat dengan data latih diurutkan secara ascending dan dihitung label dari setiap kelas yang dominan dengan syarat k harus bilangan ganjil.
# 
# ## Data Set
# source dataset [link](http://mlg.ucd.ie/datasets/bbc.html)
# 

# %%
# KNN
import pandas as pd
import numpy as np
# import matplotlib as plt

class Knn :
    n = 0

    def __init__(self, data_latih, data_uji):
        self.data_latih = data_latih
        self.data_uji = data_uji

    # cek nilai absolute positif
    def abs(nilai_abs):
        if nilai_abs < 0 :
            return nilai_abs*(-1)
        else :
            return nilai_abs

    # p = 1 --> manhattan, p = 2 --> euclidean
    def distance(self, p):
        dif = data_latih - data_uji
        print(dif)
        for i in dif :
            if p == 1 :
                n = n + abs(i**p)
            else:
                n = n + (i**p)
        return (n**(1/p))
    
    def show_data(self):
        print(self.data_latih)
        print(self.data_uji)


# pre-processing
def data_latih(data):
    label = []
    for i in range(1,4):
        n = "kelas_%d" % i
        label.append(n)
    
    labels = []
    for i in range(0, len(data)):
        if i < len(data)/3 :
            labels.append(label[0])
        if i > len(data)/3 and i < len(data)/2:
            labels.append(label[1])
        else :
            labels.append(label[2])
    return data, labels

# test dummy data
data = np.array([[2,3,2,3,2,2,2,2,3,3],[4,4,4,4,5,5,5,5,4,5],[7,7,7,7,8,8,8,7,8,8]])
label = ("kelas_1","kelas_2","kelas_3")
data_uji = np.array([6])

df_latih = pd.DataFrame(data)
df = []
for i in range(0, len(df_latih)):
    print(df_latih[i:])
print(df_latih)

# get-started
knn = Knn(data, data_uji)


# %%
