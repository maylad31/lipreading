import cv2
import numpy as np
data = np.load('crop.npz')['data']
for i in data:
    
    cv2.imshow('crop',i)
    cv2.waitKey(31)
