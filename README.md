# dev-ai
machine learning, jst, chatbot, image-processing, audio signal processing, classification method, feature extraction are included
# Daftar Isi
1. [setup](##\ /1.\ /setup)
2. [fuzzy string python](##\ /1.\ /fuzzy-string-python)
3. [k nearest neighbor](##\ /2.\ /k-nearest-neighbor)
4. [basic opencv capture from camera](##\ /4.\ /basic\ /opencv\ /capture\ /from\ /camera)

## 1. setup
* install pip on macos : sudo easy_install pip
* install jupyter : sudo python3 -m pip install jupyter
* running jupyter : jupyter notebook
* apabila gagal, install matplotlib : sudo pip install matplotlib
* install opencv packages : sudo python -m pip install opencv-python

## 2. fuzzy-string-python
- The Levenshtein Distance

    The Levenshtein distance is a metric to measure how apart are two sequences of words. In other words, it measures the minimum number of edits that you need to do to change a one-word sequence into the other. These edits can be insertions, deletions or substitutions. This metric was named after Vladimir Levenshtein, who originally considered it in 1965.

    The formal definition of the Levenshtein distance between two strings a and b can be seen as follows:

    ![formula](./img/Levenshtein.png?raw=true)


    Where 1(ai≠bj) denotes 0 when a=b and 1 otherwise. It is important to note that the rows on the minimum above correspond to a deletion, an insertion, and a substitution in that order.

## 3. k-nearest-neighbor

## 4. basic opencv capture from camera
- Capture from camera
    ```python
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```