### This file contains the tools to use the v2e too with frames
### Source: https://github.com/SensorsINI/v2e (15-10-20)

import numba as nb
from numba import jit
import logging
import h5py
import atexit
import os
import sys
import time

import cv2
import numpy as np
import numba as nb
from numba import jit
import logging
import h5py

logger = logging.getLogger(__name__)

def checkAddSuffix(path: str, suffix: str):
    if path.endswith(suffix):
        return path
    else:
        return os.path.splitext(path)[0]+suffix

class EventEmulator(object):
    """compute events based on the input frame.
    - author: Zhe He
    - contact: zhehe@student.ethz.ch
    """

    # todo add refractory period

    def __init__(
            self,
            pos_thres=0.2,
            neg_thres=0.2,
            sigma_thres=0.03,
            cutoff_hz=0,
            leak_rate_hz=0.1,
            refractory_period_s=0,  # todo not yet modeled
            shot_noise_rate_hz=0,  # rate in hz of temporal noise events
            seed=42,
            output_folder: str = None,
            dvs_h5: str = None,
            dvs_aedat2: str = None,
            dvs_text: str = None,
            # change as you like to see 'baseLogFrame',
            # 'lpLogFrame', 'diff_frame'
            show_input: str = None
            # dvs_rosbag=None
    ):
        """
        Parameters
        ----------
        base_frame: np.ndarray
            [height, width]. If None, then it is initialized from first data
        pos_thres: float, default 0.21
            nominal threshold of triggering positive event in log intensity.
        neg_thres: float, default 0.17
            nominal threshold of triggering negative event in log intensity.
        sigma_thres: float, default 0.03
            std deviation of threshold in log intensity.
        cutoff_hz: float,
            3dB cutoff frequency in Hz of DVS photoreceptor
        leak_rate_hz: float
            leak event rate per pixel in Hz,
            from junction leakage in reset switch
        shot_noise_rate_hz: float
            shot noise rate in Hz
        seed: int, default=0
            seed for random threshold variations,
            fix it to nonzero value to get same mismatch every time
        dvs_aedat2, dvs_h5, dvs_text: str
            names of output data files or None
        show_input: str,
            None or 'new_frame' 'baseLogFrame','lpLogFrame', 'diff_frame'
        """

        logger.info(
            "ON/OFF log_e temporal contrast thresholds: "
            "{} / {} +/- {}".format(pos_thres, neg_thres, sigma_thres))
        self.baseLogFrame = None
        self.t_previous = None  # time of previous frame
        self.sigma_thres = sigma_thres
        # initialized to scalar, later overwritten by random value array
        self.pos_thres = pos_thres
        # initialized to scalar, later overwritten by random value array
        self.neg_thres = neg_thres
        self.pos_thres_nominal = pos_thres
        self.neg_thres_nominal = neg_thres
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.refractory_period_s = refractory_period_s
        self.shot_noise_rate_hz = shot_noise_rate_hz
        self.output_width = None
        self.output_height = None  # set on first frame
        self.show_input = show_input
        if seed > 0:
            np.random.seed(seed)

        #  if leak_rate_hz>0:
        #      logger.warning(
        #          'leak events not yet implemented; '
        #          'leak_rate_hz={} will be ignored'.format(leak_rate_hz))

        if refractory_period_s > 0:
            logger.warning(
                'refractory period not yet implemented; '
                'refractory_period_s={} will be ignored'.format(
                    refractory_period_s))

        self.output_folder = output_folder
        self.dvs_h5 = dvs_h5
        self.dvs_aedat2 = dvs_aedat2
        self.dvs_text = dvs_text
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0

        if self.output_folder:
            if dvs_h5:
                path = os.path.join(self.output_folder, dvs_h5)
                path = checkAddSuffix(path, '.h5')
                logger.info('opening event output dataset file ' + path)
                self.dvs_h5 = h5py.File(path, "w")
                self.dvs_h5_dataset = self.dvs_h5.create_dataset(
                    name="event",
                    shape=(0, 4),
                    maxshape=(None, 4),
                    dtype="uint32")
            if dvs_aedat2:
                path = os.path.join(self.output_folder, dvs_aedat2)
                path = checkAddSuffix(path, '.aedat')
                logger.info('opening AEDAT-2.0 output file ' + path)
                self.dvs_aedat2 = AEDat2Output(path)
            if dvs_text:
                path = os.path.join(self.output_folder, dvs_text)
                path = checkAddSuffix(path, '.txt')
                logger.info('opening text DVS output file ' + path)
                self.dvs_text = DVSTextOutput(path)
        atexit.register(self.cleanup)

    def cleanup(self):
        if self.dvs_h5 is not None:
            self.dvs_h5.close()

        if self.dvs_aedat2 is not None:
            self.dvs_aedat2.close()

        if self.dvs_text is not None:
            self.dvs_text.close()

    def _init(self, firstFrameLinear):
        logger.debug(
            'initializing random temporal contrast thresholds '
            'from from base frame')
        # base_frame are memorized lin_log pixel values
        self.baseLogFrame = lin_log(firstFrameLinear)
        # If leak is non-zero, then initialize each pixel memorized value
        # some fraction of on threshold below to create leak
        # events from the start; otherwise leak would only gradually
        # grow over time as pixels spike.
        if self.leak_rate_hz > 0:
            self.baseLogFrame -= np.random.uniform(
                0, self.pos_thres, firstFrameLinear.shape)
        # initialize first stage of 2nd order IIR to first input
        self.lpLogFrame0 = np.copy(self.baseLogFrame)
        # 2nd stage is initialized to same,
        # so diff will be zero for first frame
        self.lpLogFrame1 = np.copy(self.baseLogFrame)
        # take the variance of threshold into account.
        self.pos_thres = np.random.normal(
            self.pos_thres, self.sigma_thres, firstFrameLinear.shape)
        # to avoid the situation where the threshold is too small.
        self.pos_thres[self.pos_thres < 0.01] = 0.01
        self.neg_thres = np.random.normal(
            self.neg_thres, self.sigma_thres, firstFrameLinear.shape)
        self.neg_thres[self.neg_thres < 0.01] = 0.01

    def set_dvs_params(self, model: str):
        if model == 'clean':
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.02
            self.cutoff_hz = 0
            self.leak_rate_hz = 0
            self.shot_noise_rate_hz = 0  # rate in hz of temporal noise events
            self.refractory_period_s = 0  # TODO not yet modeled

        elif model == 'noisy':
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.05
            self.cutoff_hz = 30
            self.leak_rate_hz = 0.1
            # rate in hz of temporal noise events
            self.shot_noise_rate_hz = 0.1
            self.refractory_period_s = 0  # TODO not yet modeled

        else:
            logger.error(
                "dvs_params {} not known: "
                "use 'clean' or 'noisy'".format(model))
            sys.exit(1)
        logger.info("set DVS model params with option '{}' "
                    "to following values:\n"
                    "pos_thres={}\n"
                    "neg_thres={}\n"
                    "sigma_thres={}\n"
                    "cutoff_hz={}\n"
                    "leak_rate_hz={}\n"
                    "shot_noise_rate_hz={}\n"
                    "refractory_period_s={}".format(
                        model, self.pos_thres, self.neg_thres,
                        self.sigma_thres, self.cutoff_hz,
                        self.leak_rate_hz, self.shot_noise_rate_hz,
                        self.refractory_period_s))

    def reset(self):
        '''resets so that next use will reinitialize the base frame
        '''
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.baseLogFrame = None
        self.lpLogFrame0 = None  # lowpass stage 0
        self.lpLogFrame1 = None  # stage 1

    def _show(self, inp: np.ndarray):
        min = np.min(inp)
        img = ((inp - min) / (np.max(inp) - min))
        cv2.imshow(__name__, img)
        cv2.waitKey(30)

    def generate_events(
            self, new_frame: np.ndarray,
            t_frame: float) -> np.ndarray:
        """Compute events in new frame.

        Parameters
        ----------
        new_frame: np.ndarray
            [height, width]
        t_frame: float
            timestamp of new frame in float seconds

        Returns
        -------
        events: np.ndarray if any events, else None
            [N, 4], each row contains [timestamp, y cordinate,
            x cordinate, sign of event].
            # TODO validate that this order of x and y is correctly documented
        """

        #  base_frame: the change detector input,
        #              stores memorized brightness values
        # new_frame: the new intensity frame input
        # log_frame: the lowpass filtered brightness values
        if self.baseLogFrame is None:
            self._init(new_frame)
            self.t_previous = t_frame
            return None

        if t_frame <= self.t_previous:
            raise ValueError("this frame time={} must be later than previous frame time={}".format(t_frame,self.t_previous))

        # lin-log mapping
        logNewFrame = lin_log(new_frame)

        # Apply nonlinear lowpass filter here.
        # Filter is 2nd order lowpass IIR
        # that uses two internal state variables
        # to store stages of cascaded first order RC filters.
        # Time constant of the filter is proportional to
        # the intensity value (with offset to deal with DN=0)
        deltaTime = t_frame - self.t_previous
        # logger.debug('deltaTime={}'.format(deltaTime))

        inten01 = None  # define for later
        if self.cutoff_hz > 0 or self.shot_noise_rate_hz > 0:  # will use later
            # make sure we get no zero time constants
            inten01 = (np.array(new_frame, float)+20)/275 # limit max time constant to ~1/10 of white intensity level
        if self.cutoff_hz <= 0:
            eps = 1
        else:
            tau = (1 / (np.pi * 2 * self.cutoff_hz))
            # make the update proportional to the local intensity
            eps = inten01 * (deltaTime / tau)
            eps[eps[:] > 1] = 1  # keep filter stable
        # first internal state is updated
        self.lpLogFrame0 = (1-eps)*self.lpLogFrame0+eps*logNewFrame
        # then 2nd internal state (output) is updated from first
        self.lpLogFrame1 = (1-eps)*self.lpLogFrame1+eps*self.lpLogFrame0

        # # Noise: add infinite bandwidth white noise to samples
        # # after lowpass filtering,
        # # so that the noise scales up as intensity goes down.
        # # It will model the fact that total noise power is concentrated
        # # to low frequencies when photocurrent (and transconductance)
        # # is smaller.
        #  if self.shot_noise_rate_hz > 0:
        #      noise = (self.shot_noise_rate_hz)*(
        #          np.divide(np.random.randn(
        #              logNewFrame.shape[0],
        #              logNewFrame.shape[1]), inten01 + .1))
        #      # makes the darkest pixels (with DN=0) have sample
        #      # to sample 1-sigma noise of self.shot_noise_rate_hz
        #  else:
        #      noise = 0

        # Leak events: switch in diff change amp leaks at some rate
        # equivalent to some hz of ON events.
        # Actual leak rate depends on threshold for each pixel.
        # We want nominal rate leak_rate_Hz, so
        # R_l=(dI/dt)/Theta_on, so
        # R_l*Theta_on=dI/dt, so
        # dI=R_l*Theta_on*dt
        if self.leak_rate_hz > 0:
            deltaLeak = deltaTime*self.leak_rate_hz * self.pos_thres_nominal  # scalars
            self.baseLogFrame -= deltaLeak  # subract so it increases ON events

        # log intensity (brightness) change from memorized values is computed
        # from the difference between new input
        # (from lowpass of lin-log input) and the memorized value
        diff_frame = self.lpLogFrame1 - self.baseLogFrame

        if self.show_input:
            if self.show_input == 'new_frame':
                self._show(new_frame)
            elif self.show_input == 'lpLogFrame':
                self._show(self.lpLogFrame1)
            elif self.show_input == 'lpLogFrame':
                self._show(self.lpLogFrame1)
            elif self.show_input == 'diff_frame':
                self._show(diff_frame)
            else:
                logger.error("don't know about showing {}".format(
                    self.show_input))
        pos_frame = np.zeros_like(diff_frame)  # initialize
        neg_frame = np.zeros_like(diff_frame)
        poxIdxs = diff_frame > 0
        pos_frame[poxIdxs] = diff_frame[poxIdxs]  # pixels with ON changes
        negIdxs = diff_frame < 0
        neg_frame[negIdxs] = np.abs(diff_frame[negIdxs])

        # compute quantized numbers of ON events for each pixel
        pos_evts_frame = pos_frame//self.pos_thres
        # compute number of times to pass over array to compute
        # separated ON events
        pos_iters = int(pos_evts_frame.max())
        neg_evts_frame = neg_frame // self.neg_thres  # same for OFF events
        neg_iters = int(neg_evts_frame.max())

        # ERROR: why are you here?
        #  pos_evts_frame.argmax()
        # need to iterative this many times
        num_iters = max(pos_iters, neg_iters)

        events = []

        for i in range(num_iters):
            events_curr_iters = np.zeros((0, 4), dtype=np.float32)
            # intermediate timestamps are linearly spaced
            # they start after the t_start to make sure
            # that there is space from previous frame
            # they end at t_end
            # e.g. t_start=0, t_end=1, num_iters=2, i=0,1
            # ts=1*1/2, 2*1/2
            ts = self.t_previous + deltaTime * (i + 1) / (num_iters)

            # for each iteration, compute the ON and OFF event locations
            # for that threshold amount of change
            pos_cord = (pos_frame >= self.pos_thres * (i + 1))
            neg_cord = (neg_frame >= self.neg_thres * (i + 1))

            # generate events
            pos_event_xy = np.where(pos_cord)
            num_pos_events = pos_event_xy[0].shape[0]
            neg_event_xy = np.where(neg_cord)
            num_neg_events = neg_event_xy[0].shape[0]
            num_events = num_pos_events + num_neg_events

            self.num_events_on += num_pos_events
            self.num_events_off += num_neg_events
            self.num_events_total += num_events

            # sort out the positive event and negative event
            if num_pos_events > 0:
                pos_events = np.hstack(
                    (np.ones((num_pos_events, 1), dtype=np.float32) * ts,
                     pos_event_xy[1][..., np.newaxis],
                     pos_event_xy[0][..., np.newaxis],
                     np.ones((num_pos_events, 1), dtype=np.float32) * 1))
            else:
                pos_events = np.zeros((0, 4), dtype=np.float32)

            if num_neg_events > 0:
                neg_events = np.hstack(
                    (np.ones((num_neg_events, 1), dtype=np.float32) * ts,
                     neg_event_xy[1][..., np.newaxis],
                     neg_event_xy[0][..., np.newaxis],
                     np.ones((num_neg_events, 1), dtype=np.float32) * -1))
            else:
                neg_events = np.zeros((0, 4), dtype=np.float32)

            events_tmp = np.vstack((pos_events, neg_events))

            # randomly order events to prevent bias to one corner
            #  if events_tmp.shape[0] != 0:
            #      np.random.shuffle(events_tmp)

            if num_events > 0:
                events_curr_iters = events_tmp
                #  events.append(events_tmp)

                if self.shot_noise_rate_hz > 0:
                    # NOISE: add temporal noise here by
                    # simple Poisson process that has a base noise rate
                    # self.shot_noise_rate_hz.
                    # If there is such noise event,
                    # then we output event from each such pixel

                    # the shot noise rate varies with intensity:
                    # for lowest intensity the rate rises to parameter.
                    # the noise is reduced by factor
                    # SHOT_NOISE_INTEN_FACTOR for brightest intensities
                    SHOT_NOISE_INTEN_FACTOR = 0.25
                    shotNoiseFactor = (
                        (self.shot_noise_rate_hz/2)*deltaTime/num_iters) * \
                        ((SHOT_NOISE_INTEN_FACTOR-1)*inten01+1)
                    # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

                    rand01 = np.random.uniform(
                        size=self.baseLogFrame.shape)  # draw samples

                    # probability for each pixel is
                    # dt*rate*nom_thres/actual_thres.
                    # That way, the smaller the threshold,
                    # the larger the rate
                    shotOnProbThisSample = shotNoiseFactor*np.divide(
                        self.pos_thres_nominal, self.pos_thres)
                    # array with True where ON noise event
                    shotOnCord = rand01 > (1-shotOnProbThisSample)

                    shotOffProbThisSample = shotNoiseFactor*np.divide(
                        self.neg_thres_nominal, self.neg_thres)
                    # array with True where OFF noise event
                    shotOffCord = rand01 < shotOffProbThisSample

                    shotOnXy = np.where(shotOnCord)
                    shotOnCount = shotOnXy[0].shape[0]

                    shotOffXy = np.where(shotOffCord)
                    shotOffCount = shotOffXy[0].shape[0]

                    self.num_events_on += shotOnCount
                    self.num_events_off += shotOffCount
                    self.num_events_total += shotOnCount+shotOffCount

                    if shotOnCount > 0:
                        shotEvents = np.hstack(
                            (np.ones((shotOnCount, 1), dtype=np.float32) * ts,
                             shotOnXy[1][..., np.newaxis],
                             shotOnXy[0][..., np.newaxis],
                             np.ones((shotOnCount, 1), dtype=np.float32) * 1))
                        events_curr_iters = np.append(
                            events_curr_iters, shotEvents, axis=0)
                        #  events.append(shotEvents)
                        self.baseLogFrame[shotOnCord] += \
                            shotOnCord[shotOnCord] * self.pos_thres[shotOnCord]
                    if shotOffCount > 0:
                        shotEvents = np.hstack(
                            (np.ones((shotOffCount, 1), dtype=np.float32) * ts,
                             shotOffXy[1][..., np.newaxis],
                             shotOffXy[0][..., np.newaxis],
                             np.ones((shotOffCount, 1), dtype=np.float32) * 1))
                        events_curr_iters = np.append(
                            events_curr_iters, shotEvents, axis=0)
                        #  events.append(shotEvents)
                        self.baseLogFrame[shotOffCord] -= \
                            shotOffCord[shotOffCord] * \
                            self.neg_thres[shotOffCord]
                    # end temporal noise

            # shuffle and append to the events collectors
            np.random.shuffle(events_curr_iters)
            events.append(events_curr_iters)

            if i == 0:
                # update the base frame only once,
                # after we know how many events per pixel
                # add to memorized brightness values
                # just the events we emitted.
                # don't add the remainder.
                # the next aps frame might have sufficient value
                # to trigger another event or it might not,
                # but we are correct in not storing
                # the current frame brightness
                if num_pos_events > 0:
                    self.baseLogFrame[pos_cord] += \
                        pos_evts_frame[pos_cord] * self.pos_thres[pos_cord]
                if num_neg_events > 0:
                    self.baseLogFrame[neg_cord] -= \
                        neg_evts_frame[neg_cord] * self.neg_thres[neg_cord]
                    # neg_thres is >0

        if len(events) > 0:
            events = np.vstack(events)
            if self.dvs_h5 is not None:
                # convert data to uint32 (microsecs) format
                temp_events = np.copy(events)
                temp_events[:, 0] = temp_events[:, 0] * 1e6
                temp_events[temp_events[:, 3] == -1, 3] = 0
                temp_events = temp_events.astype(np.uint32)

                # save events
                self.dvs_h5_dataset.resize(
                   self.dvs_h5_dataset.shape[0] + temp_events.shape[0],
                   axis=0)

                self.dvs_h5_dataset[-temp_events.shape[0]:] = temp_events
                self.dvs_h5.flush()
            if self.dvs_aedat2 is not None:
                self.dvs_aedat2.appendEvents(events)
            if self.dvs_text is not None:
                self.dvs_text.appendEvents(events)

        self.t_previous = t_frame
        if len(events) > 0:
            return events
        else:
            return None


class AEDat2Output:
    '''
    outputs AEDAT-2.0 jAER format DVS data from v2e
    '''

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file=None
        # edit below to match your device from https://inivation.com/support/software/fileformat/#aedat-20
        CAMERA = 'Davis346BMono'  # edit for your desired output
        if CAMERA == 'Davis346BMono':
            # DAVIS
            # In the 32-bit address:
            # bit 32 (1-based) being 1 indicates an APS sample
            # bit 11 (1-based) being 1 indicates a special event
            # bits 11 and 32 (1-based) both being zero signals a polarity event
            self.yShiftBits = 22
            self.xShiftBits = 12
            self.polShiftBits = 11  # see https://inivation.com/support/software/fileformat/#aedat-20
            self.sizex = 346
            self.sizey = 260
            self.flipy = True  # v2e uses computer vision matrix printing convention of UL pixel being 0,0, but jAER uses original graphics and graphing convention that 0,0 is LL
            self.flipx = True # not 100% sure why this is needed. Observed for tennis example
        else:
            raise ValueError('CAMERA type not found, add your camera to {}'.format(__name__))

        self.numEventsWritten = 0
        self.numOnEvents=0
        self.numOffEvents=0
        logging.info('opening AEDAT-2.0 output file {} in binary mode'.format(filepath))
        try:
            self.file = open(filepath, 'wb')
            self._writeHeader()
            atexit.register(self.cleanup)
            logger.info('opened {} for DVS output data for jAER'.format(filepath))
        except OSError as err:
            logger.error('caught {}:\n  could not open {} for writing; maybe jAER has it open?'.format(err,filepath))
            quit()

    def cleanup(self):
        self.close()

    def close(self):
        if self.file:
            logger.info("Closing {} after writing {} events ({} on, {} off)".
                        format(self.filepath,
                               EngNumber(self.numEventsWritten),
                               EngNumber(self.numOnEvents),
                               EngNumber(self.numOffEvents),
                               ))
            self.file.close()
            self.file = None

    def _writeHeader(self):
        import datetime, time, getpass
        # CRLF \r\n is needed to not break header parsing in jAER
        date = datetime.datetime.now().strftime('# Creation time: %I:%M%p %B %d %Y\r\n')  # Tue Jan 26 13:57:06 CET 2016
        time = '# Creation time: System.currentTimeMillis() {}\r\n'.format(int(time.time() * 1000.))
        user = '# User name: {}\r\n'.format(getpass.getuser())
        header = ('#!AER-DAT2.0\r\n',
                  '# This is a raw AE data file created by AEDat2Output in v2e (see https://github.com/SensorsINI/v2e) as specified at https://inivation.com/support/software/fileformat/#aedat-20\r\n',
                  '# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n',
                  '# Timestamps tick is 1 us\r\n',
                  date, time,
                  user
                  )
        for s in header:
            bytes = s.encode('UTF-8')
            self.file.write(bytes)

    def appendEvents(self, events: np.ndarray):
        if self.file is None:
            return

        if len(events) == 0:
            return
        n = events.shape[0]
        t = (1e6 * events[:, 0]).astype(np.int32)   # to us from seconds
        x = events[:, 1].astype(np.int32)
        if self.flipx: x = (self.sizex - 1) - x  # 0 goes to sizex-1
        y = events[:, 2].astype(np.int32)
        if self.flipy: y = (self.sizey - 1) - y
        p = ((events[:, 3] + 1) / 2).astype(np.int32) # 0=off, 1=on

        a = (x << self.xShiftBits | y << self.yShiftBits | p << self.polShiftBits)
        out = np.empty(2 * n, dtype=np.int32)
        out[0::2] = a  # addresses even
        out[1::2] = t  # timestamps odd
        #make sure we don't write comment char as first event
        bytes=out.byteswap().tobytes(order='C')
        if self.numEventsWritten==0:
            chopped=False
            while bytes[0:1].decode('utf-8',errors='ignore')=='#':
                logger.warning('first event would write a # comment char, dropping it')
                bytes=bytes[8:]
                chopped=True
        # now out is numpy array holding int32 timestamp,address array, i.e. ts0, ad0, ts1, ad1, etc
        self.file.write(bytes)  # java is big-endian, so  byteswap to get this
        self.numEventsWritten += n
        onCount=np.count_nonzero(p)
        offCount=n-onCount
        self.numOnEvents+=onCount
        self.numOffEvents+=offCount
        self.file.flush()
        # logger.info('wrote {} events'.format(n))


class DVSTextOutput:
    '''
    outputs text format DVS events to file according to events.txt format in http://rpg.ifi.uzh.ch/davis_data.html

    The RPG DVS text file datatset looks like this. Each line has (time(float s), x, y, polarity (0=off,1=on)

        0.000000000 33 39 1
        0.000011001 158 145 1
        0.000050000 88 143 0
        0.000055000 174 154 0
        0.000080001 112 139 1
        0.000123000 136 171 0
        0.000130001 173 90 0
        0.000139001 106 140 0
        0.000148001 192 79 1
    '''

    def __init__(self, filepath: str):
        self.filepath = filepath
        # edit below to match your device from https://inivation.com/support/software/fileformat/#aedat-20
        self.numEventsWritten = 0
        logging.info('opening text DVS output file {}'.format(filepath))
        self.file = open(filepath, 'w')
        self._writeHeader()
        atexit.register(self.cleanup)
        self.flipx = False  # set both flipx and flipy to rotate TODO replace with rotate180
        self.flipy = False
        self.sizex = 346
        self.sizey = 260  # adjust to your needs

    def cleanup(self):
        self.close()

    def close(self):
        if self.file:
            logger.info("Closing {} after writing {} events".format(self.filepath, EngNumber(self.numEventsWritten)))
            self.file.close()
            self.file = None

    def _writeHeader(self):
        import datetime, time, getpass
        date = datetime.datetime.now().strftime('# Creation time: %I:%M%p %B %d %Y\n')  # Tue Jan 26 13:57:06 CET 2016
        time = '# Creation time: System.currentTimeMillis() {}\n'.format(int(time.time() * 1000.))
        user = '# User name: {}\n'.format(getpass.getuser())
        header = ('#!events.txt\n',
                  '# This is a text DVS created by v2e (see https://github.com/SensorsINI/v2e)\n',
                  '# Format is time (float s), x, y, polarity (0=off, 1=on) as specified at http://rpg.ifi.uzh.ch/davis_data.html\n',
                  date, time,
                  user
                  )
        for s in header:
            self.file.write(s)

    def appendEvents(self, events: np.ndarray):
        if self.file is None:
            raise Exception('output file closed already')

        if len(events) == 0:
            return
        n = events.shape[0]
        t = (events[:, 0]).astype(np.float)
        x = events[:, 1].astype(np.int32)
        if self.flipx: x = (self.sizex - 1) - x  # 0 goes to sizex-1
        y = events[:, 2].astype(np.int32)
        if self.flipy: y = (self.sizey - 1) - y
        p = ((events[:, 3] + 1) / 2).astype(np.int32)  # go from -1/+1 to 0,1
        for i in range(n):
            self.file.write('{} {} {} {}\n'.format(t[i], x[i], y[i], p[i]))  # todo there must be vector way
        self.numEventsWritten += n

def lin_log(x, threshold=20):
    """
    linear mapping + logrithmic mapping.
    @author: Zhe He
    @contact: hezhehz@live.cn
    """

    # converting x into np.float32.
    if x.dtype is not np.float32:
        x = x.astype(np.float32)
    f = (1 / (threshold)) * np.log(threshold)

    y = np.piecewise(
        x,
        [x < threshold, x >= threshold],
        [lambda x: x * f,
         lambda x: np.log(x)]
    )

    return y
