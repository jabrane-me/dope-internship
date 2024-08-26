import blenderproc as bp  # must be first!
from blenderproc.python.utility.Utility import Utility
import bpy
"""
things needed 

predictions for image 
ground truth for that image 
3d model loaded 

compare the poses.
"""

import argparse
import os
import numpy as np 
import glob
import math 
from scipy import spatial
import simplejson as json 
import copy 
from pyquaternion import Quaternion
import pickle 
import bpy
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--data_prediction', 
    default = "../train2/output2/GREEN4/000", 
    help='path to prediction data')
parser.add_argument('--data', 
    default="../data_generation/blenderproc_data_gen/output2/test44/000", 
    help='path to data ground truth')
parser.add_argument('--models',
    default="content/", 
    help='path to the 3D grocery models')
parser.add_argument("--outf",
    default="results/",
    help="where to put the data"
    )
parser.add_argument('--adds',
    action='store_true',
    help="run ADDS, this might take a while"
    )
parser.add_argument("--cuboid",
    action='store_true',
    help="use cuboid to compute the ADD"
    )
parser.add_argument("--show",
    action='store_true',
    help="show the graph at the end. "
    )

opt = parser.parse_args()

if opt.outf is None:
    opt.outf = opt.data_prediction

if not os.path.isdir(opt.outf):
    print(f'creating the folder: {opt.outf}')
    os.mkdir(opt.outf)

if os.path.isdir(opt.outf + "/tmp"):
    print(f'folder {opt.outf + "/tmp"}/ exists')
else:
    os.mkdir(opt.outf + "/tmp")
    print(f'created folder {opt.outf + "/tmp"}/')

def get_all_entries(path_to_explore, what='*.json'):

    imgs = []

    def add_images(path): 
        for j in sorted(glob.glob(path+"/"+what)):
            imgs.append(j)

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path) 
                        if os.path.isdir(os.path.join(path,o))]
        for path_entry in folders:                
            explore(path_entry)
        add_images(path)

    explore(path_to_explore)
    return imgs

def create_obj(name='name', path_obj="", path_tex=None, scale=1, rot_base=None, pos_base=(-10,-10,-10)):
    bpy.ops.import_scene.obj(filepath=path_obj)
    obj = bpy.context.selected_objects[0]
    obj.name = name
    bpy.context.view_layer.objects.active = obj

    # Set the material
    mat = bpy.data.materials.new(name + "_material")
    obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Metallic'].default_value = 0
        bsdf.inputs['Roughness'].default_value = 1

    if path_tex:
        tex_image = nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(path_tex)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    obj.scale = (scale, scale, scale)
    if rot_base:
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = (rot_base.w, rot_base.x, rot_base.y, rot_base.z)
    if pos_base:
        obj.location = pos_base

    print(f' created: {obj.name}')
    return obj

def add_cuboid(obj_name, debug=False):
    obj = bpy.data.objects[obj_name]
    min_obj = np.min([v.co for v in obj.data.vertices], axis=0)
    max_obj = np.max([v.co for v in obj.data.vertices], axis=0)
    centroid_obj = np.mean([v.co for v in obj.data.vertices], axis=0)

    cuboid = [
        [max_obj[0], max_obj[1], max_obj[2]],
        [min_obj[0], max_obj[1], max_obj[2]],
        [max_obj[0], min_obj[1], max_obj[2]],
        [max_obj[0], max_obj[1], min_obj[2]],
        [min_obj[0], min_obj[1], max_obj[2]],
        [max_obj[0], min_obj[1], min_obj[2]],
        [min_obj[0], max_obj[1], min_obj[2]],
        [min_obj[0], min_obj[1], min_obj[2]],
        [centroid_obj[0], centroid_obj[1], centroid_obj[2]], 
    ]

    cuboid = [cuboid[2], cuboid[0], cuboid[3],
              cuboid[5], cuboid[4], cuboid[1],
              cuboid[6], cuboid[7], cuboid[-1]]

    cuboid.append([centroid_obj[0], centroid_obj[1], centroid_obj[2]])

    for i, p in enumerate(cuboid):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=p)
        sphere = bpy.context.object
        sphere.name = f"{obj_name}_cuboid_{i}"
        sphere.parent = obj
        if not debug:
            sphere.hide_viewport = True
            sphere.hide_render = True

    for i, v in enumerate(cuboid):
        cuboid[i] = [v[0], v[1], v[2]]

    return cuboid

def get_models(path, suffix=""):
    models = {}
    for folder in glob.glob(path+"/*/"):
        model_name = folder.replace(path,"").replace('/',"")
        print('loading',model_name + suffix)
        models[model_name] = create_obj(
            name=model_name + suffix,
            path_obj=folder + "/google_16k/textured.obj",
            path_tex=folder + "/google_16k/texture_map_flat.png",
            scale=0.01
        )
        if opt.cuboid:
            add_cuboid(model_name + suffix)
    if 'gu' in suffix: 
        models[model_name].data.materials[0].metallic = 1
        models[model_name].data.materials[0].roughness = 0.05
    return models

# Function to transform vertices
def transform_vertices(obj, position, rotation):
    q = Quaternion(rotation)
    transformed_vertices = []
    for v in obj.data.vertices:
        v_world = obj.matrix_world @ v.co
        v_rotated = q.rotate(v_world - obj.location) + position
        transformed_vertices.append(v_rotated)
    return np.array(transformed_vertices)

data_thruth = get_all_entries(opt.data, "*.json")
data_prediction = get_all_entries(opt.data_prediction, "*.json")

print('number of ground truths found', len(data_thruth))
print("number of predictions found", len(data_prediction))

meshes_gt = get_models(opt.models, '_gt')
meshes_gu = get_models(opt.models, '_gu')

adds_objects = {}
adds_all = []
all_gts = []
count_all_annotations = 0
count_by_object = {}
count_all_guesses = 0
count_by_object_guesses = {}

for gt_file in data_thruth:
    scene_gt = gt_file.replace(opt.data,"").replace('.json','')
    pred_scene = None

    for d in data_prediction:
        scene_d = d.replace(opt.data_prediction,'').replace('json','').replace('.','')
        if scene_d.split('/')[-1] == scene_gt.split('/')[-1]:
            pred_scene = d
            break

    if pred_scene is None:
        continue

    gt_json = None
    with open(gt_file) as json_file:
        gt_json = json.load(json_file)

    gu_json = None
    with open(pred_scene) as json_file:
        gu_json = json.load(json_file)

    objects_gt = [] #name obj, pose

    for obj in gt_json['objects']:
        name_gt = obj['class']
        if name_gt == '003':
            name_gt = "003_cracker_box_16k"
        objects_gt.append(
            [
                name_gt,
                {
                    "rotation": obj['quaternion_xyzw'],
                    "position": np.array(obj['location'])
                }
            ]
        )
        
        count_all_annotations += 1
        
        if name_gt in count_by_object: 
            count_by_object[name_gt] +=1 
        else:
            count_by_object[name_gt] = 1

    for obj_guess in gu_json['objects']:
        name_guess = obj_guess['class']
        name_look_up = obj_guess['class']

        try:
            pose_mesh = {
                "rotation": obj_guess['quaternion_xyzw'],
                "position": np.array([
                    float(str(obj_guess['location'][0]))/100.0,
                    float(str(obj_guess['location'][1]))/100.0,
                    float(str(obj_guess['location'][2]))/100.0
                ])
            }
        except:
            pose_mesh = {
                "rotation": obj_guess['quaternion_xyzw'],
                "position": np.array([1000000, 1000000, 1000000])
            }

        count_all_guesses += 1
        
        if name_guess in count_by_object_guesses: 
            count_by_object_guesses[name_guess] +=1 
        else:
            count_by_object_guesses[name_guess] = 1

        candidates = []
        for i_obj_gt, obj_gt in enumerate(objects_gt):
            name_gt, pose_mesh_gt = obj_gt

            if name_look_up == name_gt:
                candidates.append([i_obj_gt, pose_mesh_gt, name_gt])

        best_dist = 10000000000 
        best_index = -1 

        for candi_gt in candidates:
            i_gt, pose_gt, name_gt = candi_gt

            visii_gt = meshes_gt[name_gt]
            transformed_gt = transform_vertices(visii_gt, pose_gt['position'], pose_gt['rotation'])

            visii_gu = meshes_gu[name_look_up]
            transformed_gu = transform_vertices(visii_gu, pose_mesh['position'], pose_mesh['rotation'])

            if opt.adds:
                if opt.cuboid:
                    dist = 0
                    for i_p in range(9):
                        corner_gt = transformed_gt[i_p]
                        dist_s = []
                        for i_ps in range(9):
                            corner_gu = transformed_gu[i_ps]
                            dist_now = np.linalg.norm(corner_gt - corner_gu)
                            dist_s.append(dist_now)
                        dist += min(dist_s) 
                    dist /= 9
                    print(dist)
                else:
                    dist = np.mean(spatial.distance_matrix(
                                        np.array(transformed_gt), 
                                        np.array(transformed_gu), p=2).min(axis=1))
            else:
                if opt.cuboid:
                    dist = 0
                    for i_p in range(9):
                        corner_gt = transformed_gt[i_p]
                        corner_gu = transformed_gu[i_p]
                        dist += np.linalg.norm(corner_gt - corner_gu)
                    dist /= 9
                else:
                    dist = np.mean([np.linalg.norm(v1 - v2) for v1, v2 in zip(transformed_gt, transformed_gu)])

            if dist < best_dist:
                best_dist = dist
                best_index = i_gt

        if best_index != -1:
            if not name_guess in adds_objects.keys():
                 adds_objects[name_guess] = []
            adds_all.append(best_dist)
            adds_objects[name_guess].append(best_dist)

# save the data
if len(opt.outf.split("/"))>1:
    path = None
    for folder in opt.outf.split("/"):
        if path is None:
            path = folder
        else:
            path = path + "/" + folder 
        try:
            os.mkdir(path)
        except:
            pass        
else:
    try:
        os.mkdir(opt.outf)
    except:
        pass
print(adds_objects.keys())
count_by_object["all"] = count_all_annotations
pickle.dump(count_by_object,open(f'{opt.outf}/count_all_annotations.p','wb'))
pickle.dump(adds_all,open(f'{opt.outf}/adds_all.p','wb'))

count_by_object_guesses["all"] = count_all_guesses
pickle.dump(count_by_object,open(f'{opt.outf}/count_all_guesses.p','wb'))

labels = []
data = []
for key in adds_objects.keys():
    pickle.dump(adds_objects[key],open(f'{opt.outf}/adds_{key}.p','wb'))
    labels.append(key)
    data.append(f'{opt.outf}/adds_{key}.p')

array_to_call = ["python", "make_graphs.py","--outf", opt.outf,'--labels']

for label in labels:
    array_to_call.append(label)

array_to_call.append('--data')
for d_p in data:
    array_to_call.append(d_p)

array_to_call.append('--colours')
for i in range(len(data)):
    array_to_call.append(str(i))
if opt.show:
    array_to_call.append('--show')

print(array_to_call)
subprocess.call(array_to_call)

bpy.ops.wm.quit_blender()
