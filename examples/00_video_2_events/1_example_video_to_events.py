# Joubert Damien, 03-02-2020 - updated by AvS 22-02-2024
"""
    Script converting a video into events. 
    The framerate of the video might not be the real framerate of the original video. 
    The user specifies this parameter at the beginning.
    Please run get_video_youtube.py before executing this script.
"""
import cv2
import sys
sys.path.append("../../src")
from event_buffer import EventBuffer
from dvs_sensor import DvsSensor
from event_display import EventDisplay
from arbiter import SynchronousArbiter, BottleNeckArbiter, RowArbiter
from tqdm import tqdm
import os

if not os.path.exists("./outputs"):
    os.mkdir("./outputs")

filename = "../../data/hummingbird_video.mp4"
th_pos = 0.4        # ON threshold = 50% (ln(1.5) = 0.4)
th_neg = 0.4        # OFF threshold = 50%
th_noise= 0.01      # standard deviation of threshold noise
lat = 100           # latency in us
tau = 40            # front-end time constant at 1 klux in us
jit = 10            # temporal jitter standard deviation in us
bgnp = 0.1          # ON event noise rate in events / pixel / s
bgnn = 0.01         # OFF event noise rate in events / pixel / s
ref = 100           # refractory period in us
dt = 1000           # time between frames in us
time = 0

cap = cv2.VideoCapture(filename)

# Initialise the DVS sensor
dvs = DvsSensor("MySensor")
dvs.initCamera(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   lat=lat, jit = jit, ref = ref, tau = tau, th_pos = th_pos, th_neg = th_neg, th_noise = th_noise,
                   bgnp=bgnp, bgnn=bgnn)
# To use the measured noise distributions, uncomment the following line
dvs.init_bgn_hist("../../data/noise_pos_161lux.npy", "../../data/noise_neg_161lux.npy")

# Skip the first 50 frames of the video to remove video artifacts
for i in range(50): 
    ret, im = cap.read()

# Convert the image from uint8, such that 255 = 1e4, representing 10 klux
im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0] / 255.0 * 1e4

# Set as the initial condition of the sensor
dvs.init_image(im)

# Create the event buffer
ev_full = EventBuffer(1)

# Create the arbiter - optional, pick from one below
# ea = BottleNeckArbiter(0.01, time)                # This is a mock arbiter
# ea = RowArbiter(0.01, time)                       # Old arbiter that handles rows in random order
ea = SynchronousArbiter(0.1, time, im.shape[0])  # DVS346-like arbiter

# Create the display
render_timesurface = 1
ed = EventDisplay("Events", 
                  cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 
                  dt, 
                  render_timesurface)

if cap.isOpened():
    # Loop over num_frames frames
    num_frames = 50
    for frame in tqdm(range(num_frames), desc="Converting video to events"):
        # Get frame from the video
        ret, im = cap.read()
        if im is None:
            break
        # Convert the image from uint8, such that 255 = 1e4, representing 10 klux
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LUV)[:, :, 0] / 255.0 * 1e4
        # Calculate the events
        ev = dvs.update(im, dt)
        # Simulate the arbiter
        # num_produced = ev.i
        # ev = ea.process(ev, dt)
        # num_released = ev.i
        # statistics for the arbiter
        # print("{} produced, {} released".format(num_produced, num_released))
        # Display the events
        ed.update(ev, dt)
        # Add the events to the buffer for the full video
        ev_full.increase_ev(ev)

cap.release()
# Save the events to a .dat file
ev_full.write('outputs/ev_{}_{}_{}_{}_{}_{}.dat'.format(lat, jit, ref, tau, th_pos, th_noise))