import argparse
import math
import os
import pickle as pk
import json

import bpy
import numpy as np

# from mathutils import Matrix


def rot2quat(rot):
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rot.reshape(9)
    q_abs = np.array(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ]
    )
    q_abs = np.sqrt(np.maximum(q_abs, 0))

    quat_by_rijk = np.vstack(
        [
            np.array([q_abs[0] ** 2, m21 - m12, m02 - m20, m10 - m01]),
            np.array([m21 - m12, q_abs[1] ** 2, m10 + m01, m02 + m20]),
            np.array([m02 - m20, m10 + m01, q_abs[2] ** 2, m12 + m21]),
            np.array([m10 - m01, m20 + m02, m21 + m12, q_abs[3] ** 2]),
        ]
    )
    flr = 0.1
    quat_candidates = quat_by_rijk / np.maximum(2.0 * q_abs[:, None], 0.1)

    idx = q_abs.argmax(axis=-1)

    quat = quat_candidates[idx]
    return quat


def deg2rad(angle):
    return -np.pi * (angle + 90) / 180.0


part_match = {
    "root": "root",
    "bone_00": "Pelvis",
    "bone_01": "L_Hip",
    "bone_02": "R_Hip",
    "bone_03": "Spine1",
    "bone_04": "L_Knee",
    "bone_05": "R_Knee",
    "bone_06": "Spine2",
    "bone_07": "L_Ankle",
    "bone_08": "R_Ankle",
    "bone_09": "Spine3",
    "bone_10": "L_Foot",
    "bone_11": "R_Foot",
    "bone_12": "Neck",
    "bone_13": "L_Collar",
    "bone_14": "R_Collar",
    "bone_15": "Head",
    "bone_16": "L_Shoulder",
    "bone_17": "R_Shoulder",
    "bone_18": "L_Elbow",
    "bone_19": "R_Elbow",
    "bone_20": "L_Wrist",
    "bone_21": "R_Wrist",
    "bone_22": "L_Hand",
    "bone_23": "R_Hand",
}

x_part_match = {
    "root": "root",
    "bone_00": "pelvis",
    "bone_01": "left_hip",
    "bone_02": "right_hip",
    "bone_03": "spine1",
    "bone_04": "left_knee",
    "bone_05": "right_knee",
    "bone_06": "spine2",
    "bone_07": "left_ankle",
    "bone_08": "right_ankle",
    "bone_09": "spine3",
    "bone_10": "left_foot",
    "bone_11": "right_foot",
    "bone_12": "neck",
    "bone_13": "left_collar",
    "bone_14": "right_collar",
    "bone_15": "head",
    "bone_16": "left_shoulder",
    "bone_17": "right_shoulder",
    "bone_18": "left_elbow",
    "bone_19": "right_elbow",
    "bone_20": "left_wrist",
    "bone_21": "right_wrist",
    "bone_22": "jaw",
    "bone_23": "left_eye_smplhf",
    "bone_24": "right_eye_smplhf",
    "bone_25": "left_index1",
    "bone_26": "left_index2",
    "bone_27": "left_index3",
    "bone_28": "left_middle1",
    "bone_29": "left_middle2",
    "bone_30": "left_middle3",
    "bone_31": "left_pinky1",
    "bone_32": "left_pinky2",
    "bone_33": "left_pinky3",
    "bone_34": "left_ring1",
    "bone_35": "left_ring2",
    "bone_36": "left_ring3",
    "bone_37": "left_thumb1",
    "bone_38": "left_thumb2",
    "bone_39": "left_thumb3",
    "bone_40": "right_index1",
    "bone_41": "right_index2",
    "bone_42": "right_index3",
    "bone_43": "right_middle1",
    "bone_44": "right_middle2",
    "bone_45": "right_middle3",
    "bone_46": "right_pinky1",
    "bone_47": "right_pinky2",
    "bone_48": "right_pinky3",
    "bone_49": "right_ring1",
    "bone_50": "right_ring2",
    "bone_51": "right_ring3",
    "bone_52": "right_thumb1",
    "bone_53": "right_thumb2",
    "bone_54": "right_thumb3",
}


def init_scene(scene, root_path, gender="m", angle=0):
    # load fbx model
    if gender == "smplx-neutral":
        bpy.ops.import_scene.fbx(
            filepath=os.path.join(root_path, "data", "smplx-neutral.fbx"),
            axis_forward="-Y",
            axis_up="-Z",
            global_scale=1,
        )
        obname = "SMPLX-mesh-neutral"
        arm_obname = "SMPLX-neutral"
    else:
        bpy.ops.import_scene.fbx(
            filepath=os.path.join(root_path, "data", f"basicModel_{gender}_lbs_10_207_0_v1.0.2.fbx"),
            axis_forward="-Y",
            axis_up="-Z",
            global_scale=100,
        )
        obname = "%s_avg" % gender[0]
        arm_obname = "Armature"
    print("success load")
    ob = bpy.data.objects[obname]
    ob.data.use_auto_smooth = False  # autosmooth creates artifacts

    # assign the existing spherical harmonics material
    ob.active_material = bpy.data.materials["Material"]

    # delete the default cube (which held the material)
    # bpy.ops.object.select_all(action='DESELECT')
    # bpy.data.objects['Cube'].select = True
    # bpy.ops.object.delete(use_global=False)

    # set camera properties and initial position
    # bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects["Camera"]
    cam_ob.location = [0, 0, 0]
    cam_ob.rotation_euler = [np.pi / 2, 0, 0]
    # scn = bpy.context.sceneobname
    # scn.objects.active = cam_ob

    # th = deg2rad(angle)
    # cam_ob = init_location(cam_ob, th, params['camera_distance'])

    """
    cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']+dis),
                                 (0., -1, 0., -1.0),
                                 (-1., 0., 0., 0.),
                                 (0.0, 0.0, 0.0, 1.0)))
    """
    # cam_ob.data.angle = math.radians(60)
    # cam_ob.data.lens = 60
    # cam_ob.data.clip_start = 0.1
    # cam_ob.data.sensor_width = 32

    # # setup an empty object in the center which will be the parent of the Camera
    # # this allows to easily rotate an object around the origin
    # scn.cycles.film_transparent = True
    # scn.render.layers["RenderLayer"].use_pass_vector = True
    # scn.render.layers["RenderLayer"].use_pass_normal = True
    # scene.render.layers['RenderLayer'].use_pass_emit = True
    # scene.render.layers['RenderLayer'].use_pass_emit = True
    # scene.render.layers['RenderLayer'].use_pass_material_index = True

    # # set render size
    # # scn.render.resolution_x = params['resy']
    # # scn.render.resolution_y = params['resx']
    # scn.render.resolution_percentage = 100
    # scn.render.image_settings.file_format = 'PNG'

    # clear existing animation data
    # ob.data.shape_keys.animation_data_clear()
    arm_ob = bpy.data.objects[arm_obname]
    arm_ob.animation_data_clear()

    return (ob, obname, arm_ob)


def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0.0 else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat


def rotate180(rot):
    xyz_convert = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    return np.dot(xyz_convert.T, rot)


def convert_transl(transl):
    xyz_convert = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    return transl.dot(xyz_convert)


def rodrigues2bshapes(pose):
    if pose.size == 24 * 9:
        rod_rots = np.asarray(pose).reshape(24, 3, 3)
        mat_rots = [rod_rot for rod_rot in rod_rots]
    elif pose.size == 55 * 9:
        rod_rots = np.asarray(pose).reshape(55, 3, 3)
        mat_rots = [rod_rot for rod_rot in rod_rots]
    else:
        rod_rots = np.asarray(pose).reshape(24, 3)
        mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
    return (mat_rots, bshapes)


def load_bvh(res_db, mesh_ob, obname, arm_ob):
    scene = bpy.data.scenes["Scene"]

    # unblocking both the pose and the blendshape limits
    for k in mesh_ob.data.shape_keys.key_blocks.keys():
        mesh_ob.data.shape_keys.key_blocks[k].slider_min = -10
        mesh_ob.data.shape_keys.key_blocks[k].slider_max = 10

    # scene.objects.active = arm_ob

    # animation
    arm_ob.animation_data_clear()
    mesh_ob.animation_data_clear()
    # cam_ob.animation_data_clear()
    # load smpl params:

    # all_betas = res_db['pred_betas']
    # avg_beta = np.mean(all_betas, axis=0)

    print("nFrames ", len(res_db))
    for nframe, frame in enumerate(res_db):
        scene.frame_set(nframe)
        # apply
        trans = np.asarray(frame["transl_camsys"])
        shape = np.asarray(frame["pred_betas"])
        pose = np.asarray(frame["pred_thetas"])

        # transform pose into rotation matrices (for pose) and pose blendshapes
        mrots, bsh = rodrigues2bshapes(pose)
        mrots[0] = rotate180(mrots[0])
        trans = convert_transl(trans)

        if obname == "SMPLX-mesh-neutral":
            prefix = ""
            selected_part_match = x_part_match
        else:
            prefix = obname + "_"
            selected_part_match = part_match

        # set the location of the first bone to the translation parameter
        # arm_ob.pose.bones[obname + '_Pelvis'].location = trans
        arm_ob.pose.bones[prefix + "root"].location = trans
        arm_ob.pose.bones[prefix + "root"].keyframe_insert("location", frame=nframe)
        # set the pose of each bone to the quaternion specified by pose
        for ibone, mrot in enumerate(mrots):
            bone = arm_ob.pose.bones[prefix + selected_part_match["bone_%02d" % ibone]]
            bone.rotation_quaternion = rot2quat(mrot)
            if frame is not None:
                bone.keyframe_insert("rotation_quaternion", frame=nframe)
                bone.keyframe_insert("location", frame=nframe)

        # apply pose blendshapes
        for ibshape, bshape in enumerate(bsh):
            mesh_ob.data.shape_keys.key_blocks["Pose%03d" % ibshape].value = bshape
            if frame is not None:
                mesh_ob.data.shape_keys.key_blocks["Pose%03d" % ibshape].keyframe_insert("value", index=-1, frame=nframe)

        # apply shape blendshapes
        for ibshape, shape_elem in enumerate(shape):
            if ibshape == 0:
                key_name = "Base"
            elif ibshape <= 10:
                key_name = "Shape%03d" % (ibshape - 1)
            else:
                key_name = "Exp%03d" % (ibshape - 11)
            mesh_ob.data.shape_keys.key_blocks[key_name].value = shape_elem
            if frame is not None:
                mesh_ob.data.shape_keys.key_blocks[key_name].keyframe_insert("value", index=-1, frame=nframe)


if __name__ in ("__main__", "<run_path>"):
    # with open("test/res.pk", 'rb') as fid:
    #     res_db = pk.load(fid)
    with open("/tmp/hybrikx_frame.json", "r") as fid:
        res = json.load(fid)
    arm_ob = bpy.data.objects["SMPLX-neutral"]
    mesh_ob = arm_ob.children[0]
    cam_ob = bpy.data.objects["Camera"]
    cam_ob.location = [0, 0, 0]
    cam_ob.rotation_euler = [np.pi / 2, 0, 0]
    load_bvh([res], mesh_ob, "SMPLX-mesh-neutral", arm_ob)