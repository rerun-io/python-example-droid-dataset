#!/usr/bin/env python3

from __future__ import annotations

import rerun as rr
import argparse
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
import os
import sys
from pathlib import Path
from rerun_loader_urdf import URDFLogger
import scipy.spatial.transform as st
import pyzed.sl as sl
import cv2
import json
import glob
import h5py

SVO_PATH = Path("data/2023-08-31/Thu_Aug_31_13:55:38_2023/recordings/SVO")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds


def path_to_link(link: int) -> str:
    return "/".join(f"panda_link{i}" for i in range(link + 1))


def log_angle_rot(urdf_logger: URDFLogger, link: int, angle_rad: int) -> None:
    entity_path = path_to_link(link)
    start_translation, start_rotation_mat = urdf_logger.entity_to_transform[entity_path]

    link_to_rot_axis = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],  # 1
            [0, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, -1, 0],  # 4
            [0, 1, 0],  # 5
            [0, -1, 0],  # 6
            [0, 1, 0],  # 7
            [0, 0, 1],  # 8
        ]
    )
    vec = np.array(link_to_rot_axis[link] * angle_rad)
    rot = st.Rotation.from_rotvec(vec).as_matrix()
    rotation_mat = rot @ start_rotation_mat
    rr.log(
        entity_path, rr.Transform3D(translation=start_translation, mat3x3=rotation_mat)
    )


def h5_tree(val, pre=""):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + "└── " + key)
                h5_tree(val, pre + "    ")
            else:
                print(pre + "└── " + key + " (%d)" % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + "├── " + key)
                h5_tree(val, pre + "│   ")
            else:
                print(pre + "├── " + key + " (%d)" % len(val))


class SVOCamera:

    left_images: list[np.array]
    right_images: list[np.array]
    depth_images: list[np.array]
    width: float
    height: float
    left_dist_coeffs: np.array
    left_intrinsic_mat: np.array

    right_dist_coeffs: np.array
    right_intrinsic_mat: np.array

    def __init__(self, svo_path: Path):
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_path)
        init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        init_params.svo_real_time_mode = True
        init_params.coordinate_units = (
            sl.UNIT.METER
        )
        # init_params.depth_minimum_distance = 200
        init_params.depth_minimum_distance = 0.2

        zed = sl.Camera()
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Error reading camera data: {err}")
            sys.exit(1)

        params = (
            zed.get_camera_information().camera_configuration.calibration_parameters
        )

        # Assumes both the cameras have the same resolution.
        resolution = zed.get_camera_information().camera_configuration.resolution
        self.width = resolution.width
        self.height = resolution.height

        self.left_intrinsic_mat = np.array(
            [
                [params.right_cam.fx, 0, params.right_cam.cx],
                [0, params.right_cam.fy, params.right_cam.cy],
                [0, 0, 1],
            ]
        )
        self.right_intrinsic_mat = np.array(
            [
                [params.right_cam.fx, 0, params.right_cam.cx],
                [0, params.right_cam.fy, params.right_cam.cy],
                [0, 0, 1],
            ]
        )

        self.left_dist_coeffs = params.left_cam.disto
        self.right_dist_coeffs = params.right_cam.disto

        left_image = sl.Mat()
        right_image = sl.Mat()
        depth_image = sl.Mat()

        rt_param = sl.RuntimeParameters()

        self.left_images = []
        self.right_images = []
        self.depth_images = []

        while True:
            err = zed.grab(rt_param)
            if err == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                self.left_images.append(np.array(left_image.numpy()))

                zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                self.right_images.append(np.array(right_image.numpy()))

                zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
                self.depth_images.append(np.array(depth_image.numpy()))
            else:
                break
    
    def log_params(self, entity_path: str, n: int):
        rr.log(entity_path, rr.Transform3D(
                translation = self.translation,
                mat3x3=self.rot_mat
            ),
            timeless=True
        )
        rr.log(entity_path+"/camera", rr.Pinhole(
                image_from_camera=self.left_intrinsic_mat,
            ),
            timeless=True
        )

    def log_images_and_depth(self, entity_path: str, n: int):
        rr.log(entity_path+"/", rr.Pinhole(
            
        ))

class DROIDScene:

    dir_path: Path
    trajectory_length: int
    metadata: dict
    cameras: dict[str, SVOCamera]
    
    def __init__(self, dir_path: Path):
        self.dir_path = dir_path

        json_file_paths = glob.glob(str(self.dir_path) + "/*.json")
        if len(json_file_paths) < 1:
            raise Exception(f"Unable to find metadata file at '{self.dir_path}'")

        with open(json_file_paths[0], "r") as metadata_file:
            self.metadata = json.load(metadata_file)

        self.trajectory = h5py.File(str(self.dir_path / "trajectory.h5"), "r")
        h5_tree(self.trajectory)

        self.trajectory_length = self.metadata["trajectory_length"]

        camera_names = ["ext1", "ext2"]

        self.serial = {
            camera_name: self.metadata[f'{camera_name}_cam_serial'] for camera_name in camera_names
        }

        self.cameras = {}
        for camera_name in camera_names:
            self.cameras[camera_name] = SVOCamera(
                str(
                    self.dir_path
                    / "recordings"
                    / "SVO"
                    / f"{self.serial[camera_name]}.svo"
                ),
            )

    def log_frame_n(self, n: int):
        for (camera_name, camera) in self.cameras.items():
            if n < len(camera.left_images):
                left_image = camera.left_images[n]

                extrinsics = self.trajectory["observation"]["camera_extrinsics"][f"{self.serial[camera_name]}_left"][n]
                rotation = Rotation.from_euler('xyz', np.array(extrinsics[3:])).as_matrix()
                
                rr.log(f"{camera_name}_camera/", rr.Pinhole(
                    image_from_camera=camera.left_intrinsic_mat,
                )),
                rr.log(f"{camera_name}_camera/", rr.Transform3D(
                    translation=np.array(extrinsics[:3]),
                    mat3x3=rotation,
                )),

                rr.log(f"{camera_name}_camera/rgb", rr.Image(left_image))
                depth = camera.depth_images[n]
                depth[depth > 2.0] = 0.0
                rr.log(f"{camera_name}_camera/depth", rr.DepthImage(depth))

def main() -> None:
    rr.init("DROID-visualized")
    rr.connect()

    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--urdf", type=Path)
    args = parser.parse_args()

    scene = DROIDScene(args.data)

    urdf_logger = URDFLogger(args.urdf)
    urdf_logger.log()

    for i in range(scene.trajectory_length):
        scene.log_frame_n(i)

    return

    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("droid_100") / "1.0.0")
    parser.add_argument("--urdf-path", type=Path, default=Path("Panda") / "panda.urdf")
    args = parser.parse_args()

    urdf_logger = URDFLogger(args.urdf_path)
    urdf_logger.log()

    # ds = tfds.load("droid_raw", data_dir="gs://gresearch/robotics")
    # print(ds)
    # print(ds.info.features)


#     builder = tfds.builder_from_directory(builder_dir=args.data_dir)
#     print(builder.info.features)

#     dataset = builder.as_dataset()["train"]

#     for series in list(dataset)[:4]:
#         for i, step in enumerate(series["steps"]):
#             # rr.set_time_sequence("step", i+1)

#             rr.log("observation/exterior_image_1_left", rr.Image(np.array(step['observation']['exterior_image_1_left'])))
#             rr.log("observation/exterior_image_2_left", rr.Image(np.array(step['observation']['exterior_image_2_left'])))
#             rr.log("observation/wrist_image_left", rr.Image(np.array(step['observation']['wrist_image_left'])))

#             joint_positions = step["observation"]["joint_position"]
#             for (joint_idx, angle) in enumerate(joint_positions):
#                 log_angle_rot(urdf_logger, joint_idx+1, angle)

#             rr.log("observation/gripper_position", rr.Scalar(step['observation']['gripper_position']))

#             rr.log("discount", rr.Scalar(step['discount']))

#             rr.log("language_instructions", rr.TextDocument(
#                 f'''
# **instruction 1**: {bytearray(step["language_instruction"].numpy()).decode()}\n
# **instruction 2**: {bytearray(step["language_instruction_2"].numpy()).decode()}\n
# **instruction 3**: {bytearray(step["language_instruction_3"].numpy()).decode()}\n
#     ''', media_type="text/markdown"
#             ))

#             action_dict = step["action_dict"]
#             rr.log("/action_dict/gripper_velocity", rr.Scalar(action_dict["gripper_velocity"]))
#             rr.log("/action_dict/gripper_position", rr.Scalar(action_dict["gripper_position"]))
#             for (i, vel) in enumerate(action_dict["joint_velocity"]):
#                 rr.log(f'/action_dict/joint_velocity/{i}', rr.Scalar(vel))

#             for (i, vel) in enumerate(action_dict["joint_velocity"]):
#                 rr.log(f'/action_dict/joint_velocity/{i}', rr.Scalar(vel))


if __name__ == "__main__":
    main()
