"""
    Simulate saccadic movement along several axis directions in front of One NMNIST digit
"""
import bpy
from mathutils import Vector
import cv2
import math
import numpy as np
import os
import sys
cur_dir = os.getcwd()
sys.path.append(cur_dir + "/../../src")
from dvs_sensor_blender import Blender_DvsSensor
from event_display import EventDisplay
from event_buffer import EventBuffer
from tqdm import tqdm

path_NMNIST = os.path.abspath(os.getcwd()) + "/../../data/"

if not os.path.exists(path_NMNIST + "tmp"):
    os.mkdir(path_NMNIST + "tmp")
if not os.path.exists(path_NMNIST + "tmp/text"):
    os.mkdir(path_NMNIST + "tmp/text")
if not os.path.exists(path_NMNIST + "tmp/labels"):
    os.mkdir(path_NMNIST + "tmp/labels")
if not os.path.exists(path_NMNIST + "tmp/events"):
    os.mkdir(path_NMNIST + "tmp/events")

text_start = 0
text_end = 1
print(path_NMNIST + 'tmp/tmp_{}_{}'.format(text_start, text_end))
if not os.path.exists(path_NMNIST + 'tmp/tmp_{}_{}'.format(text_start, text_end)):
    os.mkdir(path_NMNIST + 'tmp/tmp_{}_{}'.format(text_start, text_end))

scene = bpy.context.scene

bpy.ops.object.select_all(action='DESELECT')

if "Camera" in bpy.data.objects:
    bpy.data.objects['Camera'].select_set(True)
    bpy.ops.object.delete()

if "Cube" in bpy.data.objects:
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()

if "Light" in bpy.data.objects:
    light = bpy.data.objects['Light']
    light.location = Vector((5, 5, 5))
    light.data.energy = 10

    light.data.type = 'SUN'

offset = 100

text_start = 0
bpy.ops.mesh.primitive_plane_add(size=20.0, calc_uvs=True, enter_editmode=False,
                                 location=(10 - 20.0 * text_start, 0.0, 5.0), rotation=(0, 0, 0))
pl = bpy.context.active_object
pl.name = 'NMNIST_PLANE'

material = bpy.data.materials.new(name='NMNIST_PLANE_MATERIAL')
material.use_nodes = True
nodes = material.node_tree.nodes

for node in nodes:
    nodes.remove(node)

texture = bpy.data.textures.new('NMNIST_PLANE_TEX', 'IMAGE')
texture.extension = 'REPEAT'

texture.image = bpy.data.images.load(path_NMNIST + 'tmp/text/mnist_plane_texture.png')

shader_node = nodes.new(type='ShaderNodeBsdfPrincipled')
shader_node.location = (0, 0)

texture_node = nodes.new(type='ShaderNodeTexImage')
texture_node.location = (-300, 0)
texture_node.image = texture.image

material_output_node = nodes.new(type='ShaderNodeOutputMaterial')
material_output_node.location = (300, 0)

links = material.node_tree.links
link = links.new(texture_node.outputs["Color"], shader_node.inputs["Base Color"])

link = links.new(shader_node.outputs["BSDF"], material_output_node.inputs["Surface"])

pl.data.materials.append(material)

ppsee = Blender_DvsSensor("PropheseeSensor")
ppsee.set_shape(28, 28)
ppsee.set_sensor(28, 28, 0.015)
ppsee.set_dvs_sensor(th_pos=0.15, th_neg=0.15, th_n=0.05, lat=100, tau=300, jit=30, bgn=0.1)
ppsee.init_tension()
ppsee.init_bgn_hist("../../data/noise_pos_0.1lux.npy", "../../data/noise_neg_0.1lux.npy")
ppsee.set_sensor_optics(20)
scene = bpy.context.scene
master_collection = bpy.context.collection

master_collection.objects.link(ppsee.cam)
scene.camera = ppsee.cam
ppsee.set_angle([math.pi, 0, 0.0])
ppsee.set_position([0.0, 0.0, 0.0])
ppsee.init_tension()

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = path_NMNIST + 'tmp/tmp_{}_{}/image_anim'.format(text_start, text_end)
scene.render.resolution_x = ppsee.def_x
scene.render.resolution_y = ppsee.def_y

ed = EventDisplay("Events", 28, 28, 2000)
ed.display_time = False
ev = EventBuffer(ppsee.def_x * ppsee.def_y)

# Saccade viwed as a state machine
phase_saccade = 0
size_digit = 0.1
off_y = 0.02
pos = np.array([size_digit*0.5 + 2 * size_digit * text_start, off_y, 0.0])  # x, y
speed = 0.01
t = 0
pos_img = 0
dt = 1000  # us
labels = np.load(path_NMNIST + "tmp/text/NMNIST_lab.npy")
lab = np.zeros((2, len(labels)))  # [ ts, lab]
bpy.context.scene.frame_set(pos_img)
ppsee.set_position([pos[0], pos[1], pos[2]])
# Constants
nb_img_hory = int(size_digit / (speed))
nb_img_diag = int(size_digit / (speed * np.sqrt(2)))
off = np.zeros((4, 3))
off[0, :] = np.array([size_digit, 0.0, 0.0])
off[1, :] = np.array([-size_digit / 2, -size_digit / 2, 0.0])
off[2, :] = np.array([-size_digit / 2, size_digit / 2, 0.0])
off[3, :] = np.array([size_digit * 2, 0.0, 0.0])
for nb_digit in tqdm(range(0, 10, 1)):
    lab[0, nb_digit] = t
    lab[1, nb_digit] = labels[nb_digit]
    # Saccade state machine
    # 0
    pos = pos + off[0, :]
    pos_img = pos_img + nb_img_hory
    bpy.context.scene.frame_set(pos_img)
    ppsee.set_position([pos[0], pos[1], pos[2]])
    ppsee.cam.keyframe_insert(data_path="location", index=-1)
    # 1
    pos = pos + off[1, :]
    pos_img = pos_img + nb_img_diag
    bpy.context.scene.frame_set(pos_img)
    ppsee.set_position([pos[0], pos[1], pos[2]])
    ppsee.cam.keyframe_insert(data_path="location", index=-1)
    # 2
    pos = pos + off[2, :]
    pos_img = pos_img + nb_img_diag
    bpy.context.scene.frame_set(pos_img)
    ppsee.set_position([pos[0], pos[1], pos[2]])
    ppsee.cam.keyframe_insert(data_path="location", index=-1)
    # 3
    pos = pos + off[3, :]
    pos_img = pos_img + nb_img_hory
    bpy.context.scene.frame_set(pos_img)
    ppsee.set_position([pos[0], pos[1], pos[2]])
    ppsee.cam.keyframe_insert(data_path="location", index=-1)
    t = t + dt * 2 * (nb_img_diag + nb_img_hory)

bpy.context.scene.frame_end = pos_img
bpy.context.scene.frame_current = bpy.context.scene.frame_start

# Specify the filepath for the .blend file
filepath = path_NMNIST + "/tmp/nmnist_experiment.blend"
bpy.ops.wm.save_as_mainfile(filepath=filepath)

bpy.ops.render.render(animation=True)

# Events
t = 0
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('nmnist.avi', fourcc, 20.0, (ppsee.def_x, ppsee.def_y))
for i in range(1, pos_img, 1):
    t = t + dt
    filename = path_NMNIST + 'tmp/tmp_{}_{}/'.format(text_start, text_end) + 'image_anim{0:04}.png'.format(i)
    im = cv2.imread(filename)
    out.write(im)
    if t == dt:
        ppsee.init_image(im)
    else:
        ev_buf = ppsee.update(im, dt)
        ev.increase_ev(ev_buf)
        ed.update(ev_buf, dt)
    cv2.imshow("Blender", im)
    cv2.waitKey(10)
out.release()
ev.write(path_NMNIST + 'tmp/events/MNIST_ev_{}_{}_td.dat'.format(text_start, text_end))
np.save(path_NMNIST + 'tmp/labels/MNIST_lab_events_{}_{}.npy'.format(text_start, text_end), lab)
