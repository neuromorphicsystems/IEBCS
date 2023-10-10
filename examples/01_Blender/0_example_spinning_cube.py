# Joubert Damien, 03-02-2020
import bpy
from mathutils import Vector
import cv2
import math
import os, sys
sys.path.append("../../src")
from dvs_sensor import *
from dvs_sensor_blender import Blender_DvsSensor
from event_display import EventDisplay


path_Cube = os.path.abspath(os.getcwd()) + "/../../data/"

if not os.path.exists(path_Cube + "tmp"):
    os.mkdir(path_Cube + "tmp")
if not os.path.exists(path_Cube + "tmp/events"):
    os.mkdir(path_Cube + "tmp/events")
if not os.path.exists(path_Cube + "tmp/render"):
    os.mkdir(path_Cube + "tmp/render")


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
    light.data.energy = 1

    light.data.type = 'SUN'
# Add texture
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
matname = "CheeseMapping"
texname = "texCheeseMapping"

newcube = bpy.context.active_object
newcube.name = 'Cube'

material = bpy.data.materials.new(name=matname)
material.use_nodes = True
nodes = material.node_tree.nodes

for node in nodes:
    nodes.remove(node)

texture = bpy.data.textures.new(texname, 'IMAGE')
texture.extension = 'REPEAT'

texture.image = bpy.data.images.load(os.path.abspath(os.getcwd()) + "/CheeseTextureImg.png")

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

newcube.data.materials.append(material)

# Move objects
bpy.data.objects['Cube'].location = Vector((0, 0, 10))
bpy.data.objects['Cube'].rotation_euler = Vector((3.14/4, 3.14/4, 0))
# Create the camera
ppsee = Blender_DvsSensor("Sensor")
ppsee.set_sensor(nx=360, ny=160, pp=0.015)
ppsee.set_dvs_sensor(th_pos=0.15, th_neg=0.15, th_n=0.05, lat=500, tau=300, jit=100, bgn=0.0001)
ppsee.set_sensor_optics(8)
scene = bpy.context.scene
master_collection = bpy.context.collection

master_collection.objects.link(ppsee.cam)
scene.camera = ppsee.cam
ppsee.set_angle([math.pi, 0.0, 0.0])
ppsee.set_position([0.0, 0.0, 0.0])
ppsee.set_speeds([0.0, 0, 0], [0.0, 0.0, 10])
ppsee.init_tension()
ppsee.init_bgn_hist("../../data/noise_pos_161lux.npy", "../../data/noise_pos_161lux.npy")
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = path_Cube + "tmp/render/cube_image.png"
scene.render.resolution_x = ppsee.def_x
scene.render.resolution_y = ppsee.def_y
ed = EventDisplay("Events", ppsee.def_x, ppsee.def_y, 10000)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('cube.avi', fourcc, 20.0, (ppsee.def_x, ppsee.def_y))
ev = EventBuffer(0)
for p in range(0, 100, 1):
    ppsee.update_time(1/1000)
    ppsee.print_position()
    bpy.ops.render.render(write_still=1)
    im = cv2.imread(path_Cube + "tmp/render/cube_image.png")
    out.write(im)
    if p == 0:
        ppsee.init_image(im)
    else:
        pk = ppsee.update(im, 1000)
        ed.update(pk, 1000)
        ev.increase_ev(pk)
        bpy.data.objects['Light'].data.energy += 0.01

    cv2.imshow("Blender", im)
    cv2.waitKey(1)
out.release()
ev.write(path_Cube + "tmp/events/" + 'ev_cube.dat')
