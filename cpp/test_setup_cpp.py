import numpy as np
import dsi
import time
import cv2
import sys
sys.path.append("../src")
from dvs_sensor import *
import matplotlib.pyplot as plt

dsi.initSimu(200, 200)
dsi.initLatency(100.0, 30.0, 100.0, 1000.0)
dsi.initContrast(0.3, 0.6, 0.035)
init_bgn_hist_cpp("../data/noise_pos_3klux.npy", "../data/noise_pos_3klux.npy")
img = np.zeros((200, 200), dtype=np.uint8)
img[:, :] = 125
img[50:150, 50:150] = 150
dsi.initImg(img)
img[:, :] = 125
img[55:155, 55:155] = 170
s = dsi.updateImg(img, 46000)
print(s)
s = dsi.getShape()
print(s)
s = dsi.getCurv()
print(s)
print("Test completed")
