import numpy as np
import dsi
import time
import cv2
import sys
sys.path.append("../src")
from dvs_sensor import *
import matplotlib.pyplot as plt

dsi.initSimu(260, 346)
dsi.initLatency(100.0, 30.0, 100.0, 1000.0)
dsi.initContrast(0.3, 0.6, 0.035)
init_bgn_hist_cpp("../data/noise_pos_3klux.npy", "../data/noise_pos_3klux.npy")
path_img = "/home/joubertd/Documents/Data/Simulations/Comparison_ESIM/frames_book/"
img = cv2.imread(path_img + 'frame_{}.png'.format(1606209988861033))
img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 0]
dsi.initImg(img)
img = cv2.imread(path_img + 'frame_{}.png'.format(1606209988909912))
img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 0]
s = dsi.updateImg(img, 46000)
print(s)
s = dsi.getShape()
print(s)
s = dsi.getCurv()
print(s)
print("Test completed")