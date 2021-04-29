# Damien JOUBERT 17-01-2020
import bpy
import math
import mathutils
import numpy as np
from dvs_sensor import DvsSensor

# Global variable
bins = []
for dec in range(-3, 2, 1):
    bins.append(np.arange(10 ** dec, 10 ** (dec + 1), 10 ** dec))
bins = np.array(bins)
FREQ = bins.reshape(bins.shape[0] * bins.shape[1])


class Blender_DvsSensor(DvsSensor):
    """ Structure to handle the Camera with Blender parameters such as position etc. """
    pixel_pitch = 0.015  # mm
    focal = 8.0  # mm
    def_x = 640  # pixel
    def_y = 640  # pixel
    position = np.array([0.0, 0.0, 0.0], np.float)  # Position of the sensor in the Blender's convention (Blender Unit)
    angle = np.array([0.0, 0.0, 0.0], np.float)  # Angle of the sensor (Euler) (Blender Unit)
    speed = np.array([0.0, 0.0, 0.0], np.float)  # Speed of the position (Blender Unit / s)
    angular_speed = np.array([0.0, 0.0, 0.0], np.float)  # Angular Speed (Blender Unit / s)

    def __init__(self, name):
        """ Create a new Blender camera
            Args:
                name: name of the camera
        """
        self.name = name
        cam_data = bpy.data.cameras.new(name)
        self.cam = bpy.data.objects.new(name, cam_data)

    def set_sensor(self, nx, ny, pp):
        """ Initialize the properties of the sensor:
            the definition (nx, ny) and the pixel pitch
            Args:
                 (nx, ny): the definition in number of pixels
                 pp: pixel pitch (mm)
        """
        self.pixel_pitch = pp
        self.def_x = nx
        self.def_y = ny
        self.cam.data.sensor_height = self.pixel_pitch * self.def_y
        self.cam.data.sensor_width = self.pixel_pitch * self.def_x
        self.shape = (nx, ny)

    def set_sensor_optics(self, focal):
        """ Set the optics of the camera according to the focal length
            Args:
                focal: focal lenght (mm)
        """
        self.focal = focal
        self.cam.data.angle_x = 2 * math.atan(self.cam.data.sensor_width / (2 * self.focal))
        self.cam.data.angle_y = 2 * math.atan(self.cam.data.sensor_height / (2 * self.focal))

    def update_cam(self):
        """ Update the potion of the camera in Blender according to the values stored in the structure """
        self.cam.location = mathutils.Vector((self.position[0], self.position[1], self.position[2]))
        self.cam.rotation_euler = mathutils.Euler((self.angle[0], self.angle[1], self.angle[2]))

    def set_position(self, position):
        """ Update the position in the structure and update the Blender's world
            Args:
                 position: list of positions in Blender units [x, y, z]
        """
        self.position = np.array(position)
        self.cam.location = mathutils.Vector((position[0], position[1], position[2]))

    def set_angle(self, angle):
        """ Update the angle in the structure and update the Blender's world
            Args:
                 angle: list of angles [wx, ,wy, wz] in radian
        """
        self.angle = np.array(angle)
        self.cam.rotation_euler = mathutils.Euler((angle[0], angle[1], angle[2]))

    def set_speeds(self, speed, angular_speed):
        """ Set the speeds of the camera
            Args:
            speed: list of speeds [dx, dy, dz] expressed in Blender Unit / s
            angular_speed: list of angular speeds expressed in radian/s
        """
        self.speed = np.array(speed)
        self.angular_speed = np.array(angular_speed)

    def update_time(self, dt):
        """ Update the time of the Blender's world by updating the position of the camera in Blender
            Args:
                dt: delay since the last update in s
        """
        self.position = self.position + self.speed * dt
        self.angle = self.angle + self.angular_speed * dt
        self.update_cam()

    def print_position(self):
        """ Print positon and Euler angle of the camera """
        s1 = " x : %f, y : %f, z : %f \n" % (self.position[0], self.position[1], self.position[2])
        print(s1)
        s2 = " a1 : %f, a2 : %f, a3 : %f \n" % (self.angle[0], self.angle[1], self.angle[2])
        print(s2)




