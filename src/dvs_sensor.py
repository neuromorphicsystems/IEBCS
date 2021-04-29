# Damien JOUBERT 17-01-2020
import numpy as np
import cv2
from event_buffer import EventBuffer
from tqdm import tqdm
import dsi

# Global variables
# Log bin for the noise distributions
bins = []
for dec in range(-3, 5, 1):
    bins.append(np.arange(10 ** dec, 10 ** (dec + 1), 10 ** dec))
bins = np.array(bins)
FREQ = bins.reshape(bins.shape[0] * bins.shape[1])
# Noise generation methods
NOISE_FREQ = 1     # Pixels have the same +/- noise frequency but with different phases
NOISE_MEASURE = 2  # Pixels have a noise distribution measured in one lighting conditions


class DvsSensor:
    """ This structure add the parameters needed to simulate the DVS sensor """
    shape = (50, 50)  # (y, x)
    m_th_pos = 0.2  # Mean positive sensitivity (%)
    m_th_neg = -0.2  # Mean negative sensitivity (%)
    m_th_noise = 0.02  # Mean reset noise standard deviation of the transistor (%)
    m_latency = 100  # Mean Latency (us)
    tau = 100  # Time constant (us) of the logarithmic part
    m_jitter = 30  # Mean jitter (us)
    m_bgn_pos = 0.1  # Mean positive background frequency (Hz)
    m_bgn_pos_per = np.uint64(1e6 / m_bgn_pos)  # Mean positive background period (us)
    m_bgn_neg = 0.01  # Mean negative background frequency (Hz)
    m_bgn_neg_per = np.uint64(1e6 / m_bgn_neg)  # Mean negative background period (us)
    ref = 50  # Refractory period (us)
    time = 0  # Time of the internal counter (us)
    noise_model = NOISE_FREQ # Model of noise used
    last_v = np.zeros(shape, dtype=np.double)  # Tension of each pixel during the last reset
    cur_v = np.zeros(shape, dtype=np.double)   # Tension of each pixel at the time t
    cur_th_pos = np.zeros(shape, dtype=np.double)  # Current Positive Threshold
    cur_th_neg = np.zeros(shape, dtype=np.double)   # Current Negative Threshold
    cur_ref = np.zeros(shape, dtype=np.uint64)  # Time when the pixel will have to be reset
    bgn_pos_next = np.zeros(shape, dtype=np.uint64)  # Next expected positive noise event
    bgn_neg_next = np.zeros(shape, dtype=np.uint64)   # Next expected negative noise event
    bgn_hist_pos = np.zeros((1, 1, 45), dtype=float)  # Positive noise cumulative distributions
    bgn_hist_neg = np.zeros((1, 1, 45), dtype=float)  # Negative noise cumulative distributions
    time_px = np.zeros(shape, dtype=np.uint64)  # Time t at the pixel (us)
    tau_p = np.zeros(shape, dtype=np.double)  # Time constant of each pixel (us)
    # Debug
    list_ts = []  # List of the timestamps used between two frames
    list_v = []  # List of the v values between two frames
    list_v_rst = []  # List of the reset v values between two frames

    def __init__(self, name):
        """ Init the sensor by creating the Blender Camera
        Args:
           name: string to identify the sensor
        """
        self.name = name

    def  set_shape(self, x, y):
        """ Set the shape of the sensor
            Args:
                x, y: size of the imager
        """
        self.shape = (x, y)

    def set_dvs_sensor(self, th_pos, th_neg, th_n, lat, tau, jit, bgn):
        """ Set the properties of the DVS sensor

            In this version the sensor positive and negative event's properties are symmetrical
        Args:
              th_pos: Mean threshold (log change)
              th_neg: Mean threshold (log change)
              th_n: Threshold noise (log change)
              lat: asymptotic (infinite contrast) latency (us)
              tau: Time constant of the log conversion (us)
              jit: asymptotic jitter (us)
              bgn: Mean frequency of the noise (Hz)
        """
        self.m_th_pos = th_pos
        self.m_th_neg = -th_neg
        self.m_th_noise = th_n
        self.m_latency = lat
        self.tau = tau
        self.m_jitter = jit
        self.m_bgn_pos = bgn
        self.m_bgn_pos_per = np.uint64(1e6 / bgn)
        self.m_bgn_neg_per = np.uint64(1e6 / bgn)
        self.m_bgn_neg = bgn
        self.shape = (self.shape[1], self.shape[0])
        self.last_v = np.zeros(self.shape, dtype=np.double)
        self.cur_v = np.zeros(self.shape, dtype=np.double)
        self.cur_ref = np.zeros(self.shape, dtype=np.uint64)
        self.time_px = np.zeros(self.shape, dtype=np.uint64)
        self.tau_p = np.zeros(self.shape, dtype=np.double)
        self.cur_ref[:] = -1
        self.init_bgn()
        self.init_tension()
        self.time = 0

    def init_bgn(self):
        """ Initialize the phases of the background noise
            This noise model does not include noise differences between pixel: every pixel will fire noise event a the
             same frequency but with a random phase
        """
        self.noise_model = NOISE_FREQ
        self.bgn_pos_next = np.array(np.random.randint(0, self.m_bgn_pos_per, self.shape), dtype=np.uint64)
        self.bgn_neg_next = np.array(np.random.randint(0, self.m_bgn_neg_per, self.shape), dtype=np.uint64)

    def init_bgn_hist(self, filename_noise_pos, filename_noise_neg):
        """ Load the distribution of the noise,
            Pick randomly one noise distribution for each pixel and initialize also randomly the phases of the
            background noise
            Args:
                filename_noise_pos: path of the positive noise's filename
                filename_noise_neg: path of the negative noise's filename
            """
        self.noise_model = NOISE_MEASURE
        noise_pos = np.load(filename_noise_pos)
        noise_pos = np.reshape(noise_pos, (noise_pos.shape[0] * noise_pos.shape[1], noise_pos.shape[2]))
        if len(noise_pos) == 0:
            print(filename_noise_pos, " is not correct")
            return
        noise_neg = np.load(filename_noise_neg)
        noise_neg = np.reshape(noise_neg, (noise_neg.shape[0] * noise_neg.shape[1], noise_neg.shape[2]))
        if len(noise_neg) == 0:
            print(filename_noise_neg, " is not correct")
            return
        # Load histogram and Update next noise event
        self.bgn_hist_pos = np.zeros((self.shape[0] * self.shape[1], 72), dtype=float)
        self.bgn_hist_neg = np.zeros((self.shape[0] * self.shape[1], 72), dtype=float)
        self.bgn_pos_next = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint64)
        self.bgn_neg_next = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint64)

        # Pick two spectrums for each pixel
        id_n = np.random.uniform(0, noise_neg.shape[0], size=(self.shape[0] * self.shape[1])).astype(np.int)
        id_p = np.random.uniform(0, noise_pos.shape[0], size=(self.shape[0] * self.shape[1])).astype(np.int)
        self.bgn_hist_pos = noise_pos[id_p, :]
        self.bgn_hist_neg = noise_neg[id_n, :]
        # Normalize spectrums
        s_p = np.sum(self.bgn_hist_pos, axis=1)
        s_n = np.sum(self.bgn_hist_neg, axis=1)
        id_p = np.where(s_p == 0)
        self.bgn_hist_pos[id_p, 0] = 1
        id_n = np.where(s_n == 0)
        self.bgn_hist_neg[id_n, 0] = 1
        id_p = np.where(s_p > 0)
        self.bgn_hist_pos[id_p, :] = self.bgn_hist_pos[id_p, :] / \
                                     np.repeat(self.bgn_hist_pos[id_p, -2].reshape((1, id_p[0].shape[0], 1)),
                                               self.bgn_hist_pos.shape[1],
                                               axis=2)
        id_n = np.where(s_n > 0)
        self.bgn_hist_neg[id_n, :] = self.bgn_hist_neg[id_n, :] / \
                                     np.repeat(self.bgn_hist_neg[id_n, -2].reshape((1, id_n[0].shape[0], 1)),
                                               self.bgn_hist_neg.shape[1],
                                               axis=2)

        for x in tqdm(range(0, self.shape[1], 1), desc="Noise Init"):
            for y in range(0, self.shape[0], 1):
                self.bgn_pos_next[y, x] = np.uint64(self.get_next_noise(x, y, 1) * np.random.uniform(0, 1))
                self.bgn_neg_next[y, x] = np.uint64(self.get_next_noise(x, y, 0) * np.random.uniform(0, 1))

    def init_tension(self):
        """ Initialize the thresholds of the comparators
            The positive and negative threshold share the same noise, which can be changed if necessary
        """
        self.cur_th_pos = np.clip(np.array(np.random.normal(self.m_th_pos, self.m_th_noise, self.shape),
                                           dtype=np.double), 0, 1000)
        self.cur_th_neg = np.clip(np.array(np.random.normal(self.m_th_neg, self.m_th_noise, self.shape),
                                           dtype=np.double), -1000, 0)

    def init_image(self, img):
        """ Initialize the first flux of the sensor
        Args:
            img: image whose greylevel corresponds to a radiometric value
        """
        if img.shape[1] != self.shape[1] or img.shape[0] != self.shape[0]:
            print("Error: the size of the image doesn't match with the sensor ")
            return
        if len(img.shape) == 3 and img.shape[2] == 3:
            print("Convert RGB image to Grey CV_RGB2LAB")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 0]
        ind = np.where(img > 0)
        if len(ind[0])== 0:
            print("ERROR: init_image: flux image with zeros data")
            return
        self.last_v[ind] = np.log(img[ind])
        self.cur_v[ind] = np.log(img[ind])
        self.tau_p[ind] = self.tau * 255 / img[ind]
        self.time_px[:, :] = 0
        self.time = 0

    def init_image_ESIM(self, img, time, log_eps=-1):
        """ Initialize the sensor
            Follows the algorithm: https://github.com/uzh-rpg/rpg_esim/blob/master/event_camera_simulator/esim/src/event_simulator.cpp
            Args:
                  img: Image of the light in the focal plane
                  time: time of the start (us)
                  log_eps: constant to avoid log(0)
        """
        if img.shape[1] != self.shape[1] or img.shape[0] != self.shape[0]:
            print("Error: the size of the image doesn't match with the sensor ")
            return
        if len(img.shape) == 3 and img.shape[2] == 3:
            print("Convert RGB image to Grey CV_RGB2LAB")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 0]
        if log_eps > 0:
            self.last_v = np.log(img + log_eps)
            self.cur_v = np.log(img + log_eps)
        else:
            self.last_v = img
            self.cur_v = img
        self.time = time
        self.cur_ref[:] = 0

    def check_noise(self, dt, img_d):
        """ Generate event packet of noise
            Check if the time of each pixel did not crossed a next noise event threshold during the next update
            In this method, every pixel has the same noise.
            Args:
                dt: delay between to images (us)
                img_d: log value of the input image
            Returns:
                A packet of events of type EventBuffer
        """
        ind_pos_noise = np.where(self.time + dt > self.bgn_pos_next)
        ind_neg_noise = np.where(self.time + dt > self.bgn_neg_next)
        pk_noise = EventBuffer(len(ind_pos_noise[0]) + len(ind_neg_noise[0]))
        if len(ind_pos_noise[0]) > 0:
            pk_noise.add_array(self.bgn_pos_next[ind_pos_noise], ind_pos_noise[0], ind_pos_noise[1], 1)
            self.time_px[ind_pos_noise] = self.bgn_pos_next[ind_pos_noise]
            self.bgn_pos_next[ind_pos_noise] += self.m_bgn_pos_per
            self.cur_v[ind_pos_noise] = img_d[ind_pos_noise]
            self.last_v[ind_pos_noise] = img_d[ind_pos_noise]
        if len(ind_neg_noise[0]) > 0:
            pk_noise.add_array(self.bgn_neg_next[ind_neg_noise], ind_neg_noise[0], ind_neg_noise[1], 0)
            self.time_px[ind_neg_noise] = self.bgn_neg_next[ind_neg_noise]
            self.bgn_neg_next[ind_neg_noise] += self.m_bgn_neg_per
            self.cur_v[ind_neg_noise] = img_d[ind_neg_noise]
            self.last_v[ind_neg_noise] = img_d[ind_neg_noise]
        pk_noise.sort()
        return pk_noise

    def check_noise_hist(self, dt, img_d):
        """ Generate event packet of noise
            Check if the time of each pixel did not crossed a next noise event threshold during the next update
            This version include the use of the histogram of the noise
            Args:
                  dt: delay between two updates (us)
                  img_d: logarithmic value of the input image
            Returns:
                A packet of events of type EventBuffer
        """
        ind_pos_noise = np.where(self.time + dt > self.bgn_pos_next)
        ind_neg_noise = np.where(self.time + dt > self.bgn_neg_next)
        pk_noise = EventBuffer(len(ind_pos_noise[0]) + len(ind_neg_noise[0]))
        if len(ind_pos_noise[0]) > 0:
            pk_noise.add_array(self.bgn_pos_next[ind_pos_noise], ind_pos_noise[0], ind_pos_noise[1], 1)
            self.time_px[ind_pos_noise] = self.bgn_pos_next[ind_pos_noise]
            self.cur_v[ind_pos_noise] = img_d[ind_pos_noise]
            self.last_v[ind_pos_noise] = img_d[ind_pos_noise]
            for i in range(0, len(ind_pos_noise[0]), 1):
                self.bgn_pos_next[ind_pos_noise[0][i], ind_pos_noise[1][i]] += \
                    self.get_next_noise(ind_pos_noise[1][i], ind_pos_noise[0][i], 1)
        if len(ind_neg_noise[0]) > 0:
            pk_noise.add_array(self.bgn_neg_next[ind_neg_noise], ind_neg_noise[0], ind_neg_noise[1], 0)
            self.time_px[ind_neg_noise] = self.bgn_neg_next[ind_neg_noise]
            self.cur_v[ind_neg_noise] = img_d[ind_neg_noise]
            self.last_v[ind_neg_noise] = img_d[ind_neg_noise]
            for i in range(0, len(ind_neg_noise[0]), 1):
                self.bgn_neg_next[ind_neg_noise[0][i], ind_neg_noise[1][i]] += \
                    self.get_next_noise(ind_neg_noise[1][i], ind_neg_noise[0][i], 0)
        pk_noise.sort()
        return pk_noise

    def get_next_noise(self, x, y, pol):
        """ Updates the next noise event
            Take a value between 0 and 1 and find the delay of the next noise event
            Args:
                x, y: coordonnate of the pixel
                pol: polarity of the noise
            Returns:
                the delay of the next noise event in us
        """
        val = np.random.uniform(0, 1)
        pos = y * self.shape[0] + x
        if pol == 1:
            ind = np.where(self.bgn_hist_pos[pos, :] >= val)
            next = FREQ[ind[0][0]]
        else:
            ind = np.where(self.bgn_hist_neg[pos, :] >= val)
            next = FREQ[ind[0][0]]
        return np.uint64(1e6 / next)

    def get_latency(self, time_end, last_v, cur_th_pos, cur_v, img_d, time_px):
        """ Obtain the latency of the pixel
            Method: Linearly interpolates the time when it crosses the threshold and add the constant latency of the
            comparator stage.
            Args:
                time_end: time of the change (us)
                last_v: voltage of the last spike (np.array)
                cur_th_pos: threshold (np.array)
                cur_v: voltage at time_px (np.array)
                last_v: voltage during the last spike (np.array)
            Returns:
                a np.array of the latencies in us
        """
        return np.uint64((last_v + cur_th_pos - cur_v) / (img_d - cur_v) * (time_end - time_px) + \
               np.clip(np.random.normal(self.m_latency, self.m_jitter, last_v.shape[0]), 0, 1e6))

    def get_latency_tau(self, cur_th_pos, cur_v, img_d, tau_p):
        """ Obtain the latency of the pixel
            Method: First order low pass filter interpolation of the time when it crosses the threshold and add the constant latency of the
            comparator stage
            Args:
                last_v: voltage of the last spike
                cur_th_pos: voltage to reach to cross the threshold
                cur_v: voltage at time_px
                tau_p: time constants of the pixels
            Returns:
                a np.array of the latencies in us
        """
        amp = np.divide(cur_th_pos - cur_v, img_d - cur_v)
        # Correct numerical errors
        jit = np.sqrt(self.m_jitter ** 2 + np.power(self.m_th_noise * tau_p / (img_d - cur_v), 2))
        t_ev = np.uint64(np.clip(np.random.normal(self.m_latency - tau_p*np.log(1 - amp), jit), 0, 10000))
        return t_ev

    def update(self, img, dt, debug=False):
        """ Update the sensor with a nef irradiance's frame
            Follow the ICNS model
            Args:
                img: radiometric value in the focal plane
                dt: delay between the frame and the last one (us)
                debug: return he intermediary values of the tension before the comparators
            Returns:
                EventBuffer of the created events
             """
        if debug:
            self.list_ts.append(np.array(self.time_px))
            self.list_v.append(np.array(self.cur_v))
            self.list_v_rst.append(np.array(self.last_v))
        if img.shape[1] != self.shape[1] or img.shape[0] != self.shape[0]:
            print("Error: the size of the image doesn't match with the sensor ")
            return
        if len(img.shape) == 3 and img.shape[2] == 3:
            print("Convert RGB image to Lab and use L")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img = img[:, :, 0]
        # Convert in the log domain
        img_d = np.array(img, dtype=np.double)
        ind = np.where(img > 0)
        if len(ind[0]) == 0:
            print("ERROR: update: flux image with only zeros data")
            return
        img_d[ind] = np.log(img[ind])
        ind = np.where(img_d != 0)
        # Update time constants
        self.tau_p[ind] = self.tau * np.log(255) / img_d[ind]
        # Update refractory and reset pixels
        ind_ref = np.where(self.cur_ref < self.time + dt)
        if len(ind_ref[0]) > 0:
            self.last_v[ind_ref] = self.cur_v[ind_ref] + (img_d[ind_ref] - self.cur_v[ind_ref]) * \
                                (1 - np.exp(-(np.array(self.cur_ref[ind_ref] - self.time, dtype=np.float)) / self.tau_p[ind_ref]))
            self.time_px[ind_ref] = self.cur_ref[ind_ref]
            self.cur_ref[ind_ref] = -1
            self.cur_v[ind_ref] = self.last_v[ind_ref]
            if debug:
                self.list_ts.append(np.array(self.time_px))
                self.list_v.append(np.array(self.cur_v))
                self.list_v_rst.append(np.array(self.last_v))
        # Check Noise
        if self.noise_model == NOISE_FREQ:
            pk_noise = self.check_noise(dt, img_d)
        else:
            pk_noise = self.check_noise_hist(dt, img_d)
        # Check contrast
        target = np.zeros(img_d.shape)
        target[ind] = self.cur_v[ind] + (img_d[ind] - self.cur_v[ind]) * (1 - np.exp(-np.array(self.time + dt - self.time_px[ind], dtype=np.float) / self.tau_p[ind]))  # Flux at the end of t + dt
        dif = target - self.last_v
        ind_pos = np.where((dif > self.cur_th_pos) & (self.cur_ref == np.uint64(-1)))
        ind_neg = np.where((dif < self.cur_th_neg) & (self.cur_ref == np.uint64(-1)))
        pk = EventBuffer(0)
        while len(ind_pos[0]) + len(ind_neg[0]) > 0:
            pk.increase(len(ind_pos[0]) + len(ind_neg[0]))
            # Positive
            if len(ind_pos[0]) > 0:
                t_event = self.get_latency_tau(self.last_v[ind_pos] + self.cur_th_pos[ind_pos], self.cur_v[ind_pos], img_d[ind_pos], self.tau_p[ind_pos])
                pk.add_array(self.time_px[ind_pos] + t_event, ind_pos[0], ind_pos[1], 1)
                self.cur_th_pos[ind_pos] = np.clip(np.random.normal(self.m_th_pos, self.m_th_noise, len(ind_pos[0])), 0, 1000)
                self.cur_ref[ind_pos] = self.time_px[ind_pos] + t_event + self.ref
            # Negative
            if len(ind_neg[0]) > 0:
                t_event = self.get_latency_tau(self.last_v[ind_neg] + self.cur_th_neg[ind_neg], self.cur_v[ind_neg], img_d[ind_neg], self.tau_p[ind_neg])
                pk.add_array(self.time_px[ind_neg] + t_event, ind_neg[0], ind_neg[1], 0)
                self.cur_th_neg[ind_neg] = np.clip(np.random.normal(self.m_th_neg, self.m_th_noise, len(ind_neg[0])), -1000, 0)
                self.cur_ref[ind_neg] = self.time_px[ind_neg] + t_event + self.ref
            pk.sort()
            # Update refractory
            ind_ref = np.where(self.cur_ref < self.time + dt)
            if len(ind_ref[0]) > 0:
                self.last_v[ind_ref] = self.cur_v[ind_ref] + (img_d[ind_ref] - self.cur_v[ind_ref]) * \
                                       (1 - np.exp(-(np.array(self.cur_ref[ind_ref]-self.time_px[ind_ref], dtype=np.float)) / self.tau_p[ind_ref]))
                self.time_px[ind_ref] = self.cur_ref[ind_ref]
                self.cur_ref[ind_ref] = -1
                self.cur_v[ind_ref] = self.last_v[ind_ref]
                if debug:
                    self.list_ts.append(np.array(self.time_px))
                    self.list_v.append(np.array(self.cur_v))
                    self.list_v_rst.append(np.array(self.last_v))
            dif = np.zeros((self.shape[0], self.shape[1]))
            target[ind_ref] = self.cur_v[ind_ref] + (img_d[ind_ref] - self.cur_v[ind_ref]) * (
                        1 - np.exp(-np.array(self.time + dt - self.time_px[ind_ref], dtype=np.float) / self.tau_p[ind_ref]))
            dif[ind_ref] = target[ind_ref] - self.last_v[ind_ref]
            ind_pos = np.where(dif > self.cur_th_pos)
            ind_neg = np.where(dif < self.cur_th_neg)
        self.cur_v[ind] = self.cur_v[ind] + (img_d[ind] - self.cur_v[ind]) * (1 - np.exp(-np.array(self.time + dt - self.time_px[ind], dtype=np.float) / self.tau_p[ind]))
        # Update Time
        self.time += dt
        self.time_px[:] = self.time
        if debug:
            self.list_ts.append(np.array(self.time_px))
            self.list_v.append(np.array(self.cur_v))
            self.list_v_rst.append(np.array(self.last_v))
        pk_end = EventBuffer(0)
        pk_end.merge(pk, pk_noise)
        if debug: print('{} Noise event, {} signal events'.format(pk_noise.i, pk.i))
        return pk_end

    def update_esim(self, im, time, log_eps=-1):
        """ Update the sensor with a nef irradiance's frame
            Follow the ESIM model:
            https://github.com/uzh-rpg/rpg_esim/blob/master/event_camera_simulator/esim/src/event_simulator.cpp
            In this algoritm, self.cur_ref is used to store the last event generated
            Args:
                img: radiometric value in the focal plane
                dt: delay between the frame and the last one (us)
            Returns:
                EventBuffer of the created events
             """
        tolerance = 1e-6
        preprocessed_img = np.array(im)
        if log_eps != -1:
            preprocessed_img = np.log(log_eps + im)
        if np.sum(self.last_v) == 0:
            self.init_image_ESIM(preprocessed_img, time)
            return
        ev = EventBuffer(0)
        delta_t = time - self.time
        for x in range(0, self.shape[1], 1):
            for y in range(0, self.shape[0], 1):
                itdt = preprocessed_img[y, x]
                it = self.cur_v[y, x]
                prev_cross = self.last_v[y, x]  # ref_values_ is self.last_v
                if abs(it - itdt) > tolerance:
                    pol = 1.0 if itdt >= it else -1.0
                    C = self.m_th_pos if pol > 0 else self.m_th_neg
                    sigma_C = self.m_th_noise  # Single reset noise
                    if sigma_C > 0:
                        C += np.random.normal(0, sigma_C, 1)
                        C = 0.01 if C  < 0.01 else C
                    curr_cross = prev_cross
                    while True:
                        curr_cross += pol * C
                        if (pol > 0 and  curr_cross > it and curr_cross <= itdt) or \
                                (pol < 0 and curr_cross < it and curr_cross >= itdt):
                            edt = (curr_cross - it) * delta_t / (itdt - it)
                            t = self.time + edt
                            last_stamp_at_xy = self.cur_ref[y, x]
                            dt = t - last_stamp_at_xy
                            if last_stamp_at_xy == 0 or dt >= self.ref:
                                ev.add(t, y, x, pol)
                                self.cur_ref[y, x] = t
                            else:
                                print("Dropping event because time since last event ",
                                      str(dt), " ns) < refractory period (", str(self.ref), " us).")
                            self.last_v[y, x] = curr_cross
                        else:
                            break
        self.time = time
        self.last_v = preprocessed_img
        ev.sort()
        return ev


def init_bgn_hist_cpp(filename_noise_pos, filename_noise_neg):
    """ Load the distribution of the noise for the cpp simulator
        Pick randomly one noise distribution for each pixel and initialize also randomly the phases of the
        background noise
        Args:
            filename_noise_pos: path of the positive noise's filename
            filename_noise_neg: path of the negative noise's filename
        """
    noise_pos = np.load(filename_noise_pos)
    if len(noise_pos) == 0: raise NameError(filename_noise_pos + " is not correct")
    noise_pos = np.reshape(noise_pos, (noise_pos.shape[0] * noise_pos.shape[1], noise_pos.shape[2]))
    ind_n = np.where(noise_pos[:, -2] == 0)
    noise_pos[ind_n, 1] = 1
    noise_pos[ind_n, -2] = 1
    div = np.tile(noise_pos[:, -2], [noise_pos.shape[1], 1])
    noise_pos = noise_pos / div.transpose()

    noise_neg = np.load(filename_noise_neg)
    if len(noise_neg) == 0: raise NameError(filename_noise_neg + " is not correct")
    noise_neg = np.reshape(noise_neg, (noise_neg.shape[0] * noise_neg.shape[1], noise_neg.shape[2]))
    ind_n = np.where(noise_neg[:, -2] == 0)
    noise_neg[ind_n, 1] = 1
    noise_neg[ind_n, -2] = 1
    div = np.tile(noise_neg[:, -2], [noise_neg.shape[1], 1])
    noise_neg = noise_neg / div.transpose()
    dsi.initNoise(noise_pos, noise_neg)