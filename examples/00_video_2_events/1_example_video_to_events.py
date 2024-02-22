# Joubert Damien, 03-02-2020
"""
    Script converting a video into events. 
    The framerate of the video might not be the real framerate of the original video. 
    The user specifies this parameter at the beginning.
    Please run get_video_youtube.py before executing this script.
"""
import cv2
import sys
sys.path.append("./src")
from event_buffer import EventBuffer
from dvs_sensor import DvsSensor
from event_display import EventDisplay
# import numpy as np
from tqdm import tqdm
filename = "./data/video/See Hummingbirds Fly Shake Drink in Amazing Slow Motion  National Geographic.mp4"
lat = 100
jit = 10
ref = 100
tau = 300
th = 0.3
th_noise = 0.01
cap = cv2.VideoCapture(filename)
dvs = DvsSensor("MySensor")
dvs.set_shape(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
dvs.set_dvs_sensor(th_pos=0.15, th_neg=0.15, th_n=0.05, lat=100, tau=300, jit=30, bgn=0.1)
dvs.init_thresholds()
dvs.init_bgn_hist("./data/noise_pos_0.1lux.npy", "./data/noise_neg_0.1lux.npy")
isInit = False
dt = 1000  # FPS must be 1 kHz
ev_full = EventBuffer(1)
ed = EventDisplay("Events", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 4 * dt, 1)
time = 0
ret, im = cap.read()
im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
dvs.init_image(im)

if cap.isOpened():
    for frame in tqdm(range(100), desc="Converting video to events"):
        ret, im = cap.read()
        if im is None:
            break
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0]
        ev = dvs.update(im, dt)
        ed.update(ev, dt)
        ev_full.increase_ev(ev)

cap.release()
ev_full.write('ev_{}_{}_{}_{}_{}_{}.dat'.format(lat, jit, ref, tau, th, th_noise))