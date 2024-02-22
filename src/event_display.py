# Damien JOUBERT 17-01-2020
import numpy as np
import cv2


class EventDisplay():
    """ Class to handle the thread created by OpenCV to render an image """

    def __init__(self, name, dx, dy, frametime, render=0, display_time=True):
        """ Initialise the display by reseting the internal timer of the structure and providing the right size of
            buffers
            Args:
                name: name of the windows
                dx, dy: size of the data
                frametime: delay between two frames (us)
                render: rendering method: 0 = binary, 1 = time surface
            """
        self.name = name
        self.time = 0
        self.last_frame = 0
        self.frametime = frametime
        self.time_surface = np.zeros((int(dy), int(dx)), dtype=np.uint64)
        self.pol_surface = np.zeros((int(dy), int(dx)), dtype=np.uint8)
        self.im = np.zeros((int(dy), int(dx), 3), dtype=np.uint8)
        self.render = render
        self.display_time = display_time
        self.render_tau = 3 * frametime

    def reset(self):
        """ Reset timers and buffers to 0 """
        self.time = 0
        self.last_frame = 0
        self.time_surface[:] = 0
        self.pol_surface[:] = 0

    def update(self, pk, dt):
        """  During the time dt, a new EventBuffer pk was created. This method adds these events and
            triggers a display if needed
            Args:
                pk: EventBuffer
                dt: delay since the last update
            """
        # print(self.time, self.time_surface.max())
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
                self.im[:, :, 1] = self.im[:, :, 0]
                self.im[:, :, 2] = self.im[:, :, 0]
            if self.render == 1:
                print(self.time - self.time_surface.max(), self.time - self.time_surface.min())
                self.im[:, :, 0] = 128 - (self.pol_surface * 2 - 1) * 128 * np.exp(-(self.time - self.time_surface.astype(np.double)) / self.render_tau)
                self.im[:, :, 1] = 128
                self.im[:, :, 2] = 128
            if self.display_time: 
                self.im = cv2.putText(self.im, '{} s'.format(self.time / 1e6), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
            cv2.imshow(self.name, self.im)
            cv2.waitKey(10)










