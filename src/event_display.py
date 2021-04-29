# Damien JOUBERT 17-01-2020
import numpy as np
import cv2


class EventDisplay():
    """ Structure to handle the thread created by OpenCV to render an image """
    name = "test"  # Name of the window
    time = 0  # Internal counter of the display (us)
    last_frame = 0  # Time of the last frame
    frametime = 100000  # Time to refresh the display (us)
    time_surface = np.zeros((10, 10), dtype=np.uint64)  # Timestamp of the last event in the focal plane
    pol_surface = np.zeros((10, 10), dtype=np.uint8)  # Polarity of the last event in the focal plane
    im = np.zeros((10, 10, 3), dtype=np.uint8)  # Image to render
    render = 0  # 0: binary image, 1: ts
    render_tau = 40000  # tau decay of the time surface (us)
    display_time = True

    def __init__(self, name, dx, dy, frametime, render=0):
        """ Initialize the Display by reseting the internal timer of the structure and providing the right size of
            buffers
            Args:
                name: name of the windows
                dy dx: size of the data
                frametime: delay between two frames (us)
                render: rendering method: 0 = binary, 1 = timesurface
            """
        self.name = name
        self.time = 0
        self.last_frame = 0
        self.frametime = frametime
        self.time_surface = np.zeros((int(dy), int(dx)), dtype=np.uint64)
        self.pol_surface = np.zeros((int(dy), int(dx)), dtype=np.uint8)
        self.im = np.zeros((int(dy), int(dx), 3), dtype=np.uint8)
        self.render = 0
        self.render_tau = 3 * frametime

    def reset(self):
        """ Reset timers and buffers to 0 """
        self.time = 0
        self.last_frame = 0
        self.time_surface[:] = 0
        self.pol_surface[:] = 0

    def update(self, pk, dt):
        """  During the time dt, the EventBuffer was created. This function adds these events to the structure and
            triggers a display if needed
            Args:
                pk: EventBuffer
                dt: delay since the last update
            """
        self.time_surface[pk.y[:pk.i], pk.x[:pk.i]] = pk.ts[:pk.i]
        self.pol_surface[pk.y[:pk.i], pk.x[:pk.i]] = pk.p[:pk.i]
        self.time += dt
        self.last_frame += dt
        if self.last_frame > self.frametime:
            self.last_frame = 0
            self.im[:] = 125
            if self.render == 0:
                ind = np.where((self.time_surface > self.time - self.frametime) & (self.time_surface <= self.time))
                self.im[:, :, 0][ind] = self.pol_surface[ind]*255
                self.im[:, :, 1][ind] = self.pol_surface[ind]*255
                self.im[:, :, 2][ind] = self.pol_surface[ind]*255
            if self.render == 1:
                self.im[:, :, 0] = (self.pol_surface * 2 - 1) * 125 * np.exp(-(self.time - self.time_surface.astype(np.double)) / self.render_tau)
            if self.display_time: self.im = cv2.putText(self.im, '{} s'.format(self.time / 1e6), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
            cv2.imshow(self.name, self.im)
            cv2.waitKey(10)










