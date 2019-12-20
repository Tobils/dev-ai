# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Feature Extractions Method
# 1. Local Binary Pattern 
#    - Local Binary Pattern Histogram (LBPH)
#    - Local Binary Pattern Histogram Uniform (LBPH-Uniform)
#     
# 2. Mel Frequency Ceptrum (MFCC)
# 3. Principal Component Analysis (PCA)
# %% [markdown]
# ## 1. Local Binary Pattern
# 1. Local Binary Pattern Histogram (LBPH)
# 2. Local Binary Pattern Histogram Uniform (LBPH-Uniform)

## LBP 1 citra
# img = cv2.imread('hand_shape.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ciri_lbp  = LBPH(img_gray)
# df = pd.DataFrame(ciri_lbp)

# %%
# Local Binary Pattern Histogram LBPH 
# reference : https://github.com/lionaneesh/LBP-opencv-python/blob/master/Basic-3x3-LBP.py
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import pandas as pd

def set_backgroud_0(gray, thresh):
    x,y = thresh.shape
    for i in range(x):
        for j in range(y):
            if thresh[i,j] == 0:
                gray.itemset((i,j),0)
            else:
                pass
    return gray

def thresholded(center, pixels):
    threshold = []
    for a in pixels:
        if a >= center:
            threshold.append(1)
        else :
            threshold.append(0)
    return threshold

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx, idy]
    except:
        return default

## x,y --> posisi pixel citra
def LBPH(raw_data):
    # ciri uniform
    pattern = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 127, 128, 129, 131, 135, 143, 159, 192,
               193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]
  
    patt = np.zeros((59,1), dtype=np.int64)

    for x in range(0, len(raw_data)):
        for y in range(0, len(raw_data[0])):
            center          = raw_data[x,y]
            top_left        = get_pixel_else_0(raw_data, x-1, y-1)
            top_up          = get_pixel_else_0(raw_data, x, y-1)
            top_right       = get_pixel_else_0(raw_data, x+1, y-1)
            right           = get_pixel_else_0(raw_data, x+1, y)
            left            = get_pixel_else_0(raw_data, x-1, y)
            bottom_left     = get_pixel_else_0(raw_data, x-1, y+1)
            bottom_right    = get_pixel_else_0(raw_data, x+1, y+1)
            bottom_down     = get_pixel_else_0(raw_data, x, y+1)

            lbp_threshold = thresholded(center, [top_left, top_up, top_right, right, bottom_right,                                                 
                                            bottom_right, bottom_left, left])
            weights = [1, 2, 4, 8, 16, 32, 64, 128]

            lbp_value = 0
            for idx in range(0, len(lbp_threshold)):
                lbp_value += weights[idx] * lbp_threshold[idx]
            
            # set apttern
            for i in range(len(pattern)):
                if lbp_value == pattern[i]:
                    patt[i] += 1
                else :
                    patt[58] += 1
    return patt


## LBP Banyak Citra
## 1. baca path
## 2. lbp citra pada setiap path
## 3. simpan ciri
data_lbp = []
path = []
for i in range(10):
    kelas = "/Users/dev-tobil/Documents/dev-ai/dev-ai/dataset/dataset_img/data_latih/kelas_%d" %i
    path.append(kelas)


for path_file in path:
    print(path_file)
    for im in glob.glob(path_file+"/*jpg"):
        print(im)
        img = cv2.imread(im, 0)

        # watershed --> background removal
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # img_no_bg = set_backgroud_0(img,thresh)

        # histogram equalization
        img_hq      = cv2.equalizeHist(img)
        hasil_lbp   = LBPH(img_hq)
        data_lbp.append(hasil_lbp)
    del hasil_lbp

# concate data dan jadikan dalam bentuk DataFrame len(data_lbp) = 10
df = pd.DataFrame(data_lbp[0])
for i in range(1, len(data_lbp)):
    tmp = pd.DataFrame(data_lbp[i])
    df  = pd.concat([df,tmp], axis=1)
    del tmp

# simpan ciri dalam bentuk csv
df.to_csv('data_latih_citra_tangan.csv', mode='a')
print(df)

# %%
