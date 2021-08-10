import numpy as np
import os
import sys
import dsi
cur_dir = os.getcwd()
sys.path.append(cur_dir + "/../../src")
from dvs_sensor import init_bgn_hist_cpp, EventBuffer
from event_display import EventDisplay
from tqdm import tqdm
import h5py

dir_res = "track_dataset_v3"
if not os.path.exists(dir_res):
    os.mkdir(dir_res)

# Parameters
res = [346, 260]
sensi = 0.15
sensi_noise = 0.03
lat_comp = 300
jit = 100
tau_log = 1000
ref_per = 100
radius = np.linalg.norm(res, 2)
background = 25                   # Smallest brightness seen by the event sensor
speeds = [100, 1000, 10000]       # px / s
sizes = [1]                       # the size of the PSF is 3 x sizes
sat_mag = [12, 9, 6]              # Exponential brightness of the object
nb_runs = 30
noise_distribs = [
    ["data/noise_pos_0.1lux.npy", "data/noise_neg_0.1lux.npy"],
]
noise_names = ["3klux", "161lux", "0.1lux"]
x = np.tile(np.arange(0, res[0], 1), (res[1], 1))
y = np.tile(np.arange(0, res[1], 1), (res[0], 1)).transpose()
ed = EventDisplay("Events", res[1], res[0], 20000)
# Save params
f = open('{}/params_v2.csv'.format(dir_res), "w")
id = 0
for noise in enumerate(noise_distribs):
    for speed in speeds:
        for size in sizes:
            for m in sat_mag:
                for r in range(0, nb_runs, 1):
                    id += 1
                    f.write('{};{};{};{};{};{}\n'.format(id, speed, size, m, r, noise))
f.close()
# Generate sequences
for i, noise in enumerate(noise_distribs):
    filename = 'Noise_{}'.format(noise_names[i])
    dsi.initSimu(res[1], res[0])
    dsi.initLatency(lat_comp, jit, ref_per, tau_log)
    dsi.initContrast(sensi, sensi, sensi_noise)
    init_bgn_hist_cpp("../../" + noise[0], "../../" + noise[1])
    time_simu = 0
    dsi.initImg(np.full((res[1], res[0]), 1))
    id = 0
    for speed in speeds:
        dur_simu = radius / speed * 1e6
        dt = int(1e6 / (2 * speed))
        for size in sizes:
            for m in sat_mag:
                for r in range(0, nb_runs, 1):
                    ev_cpp = EventBuffer(1)
                    # Initial position/speeds of the object
                    angle = np.random.uniform(0, 2 * np.pi)
                    x0 = radius // 2 * np.cos(angle)
                    y0 = radius // 2 * np.sin(angle)
                    dx = - speed * np.cos(angle) / 1e6 * dt
                    dy = - speed * np.sin(angle) / 1e6 * dt
                    x0 += res[0] // 2
                    y0 += res[1] // 2
                    id += 1
                    t0 = time_simu
                    for t in tqdm(range(0, int(dur_simu)+1, dt)):
                        time_simu += dt
                        x0 += dx
                        y0 += dy
                        if 0 <= x0 < res[0] and 0 <= y0 < res[1]:
                            ct = np.exp(-np.sqrt(np.power(x0 - x, 2) + np.power(y0 - y, 2)) / (size))
                        else:
                            ct = np.zeros((res[1], res[0]))
                        img = 1 + ct * np.exp((background-m)/2.5)
                        buf = dsi.updateImg(img, dt)
                        ev_cpp.add_array(np.array(buf["ts"], dtype=np.uint64),
                                         np.array(buf["x"], dtype=np.uint16),
                                         np.array(buf["y"], dtype=np.uint16),
                                         np.array(buf["p"], dtype=np.uint64),
                                         100000000)
                    ev_cpp.sort()
                    events = np.zeros((4, ev_cpp.i), dtype=np.uint32)
                    events[0, :] = ev_cpp.get_x()
                    events[1, :] = ev_cpp.get_y()
                    events[2, :] = ev_cpp.get_p()
                    events[3, :] = ev_cpp.get_ts()
                    with h5py.File('{}/{}_event.h5'.format(dir_res, id),
                                   'w') as h5f:
                        g = h5f.create_group("events")
                        g.create_dataset("recording", compression="gzip",
                                         data=events)
                    # Compute and save GT
                    x0 = radius // 2 * np.cos(angle) + res[0] // 2
                    y0 = radius // 2 * np.sin(angle) + res[1] // 2
                    x_gt = x0 + dx / dt * (ev_cpp.get_ts() - t0)
                    y_gt = y0 + dy / dt * (ev_cpp.get_ts() - t0)
                    events_gt = np.empty(ev_cpp.i, dtype=[('x', np.float32), ('y', np.float32), ('id', np.uint16), ('ts', np.uint32)])
                    events_gt['x'] = x_gt
                    events_gt['y'] = y_gt
                    events_gt['id'][:] = id
                    events_gt['ts'] = ev_cpp.get_ts()
                    with h5py.File('{}/{}_gt.h5'.format(dir_res, id),
                                   'w') as h5f:
                        g = h5f.create_group("events_GT")
                        g.create_dataset("recording", compression="gzip",
                                         dtype=[('x', np.float32), ('y', np.float32), ('id', np.uint16), ('ts', np.uint32)],
                                         data=events_gt)




