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

"""
videos/5030472d/2023-07-18/Tue_Jul_18_14:04:48_2023/27904255.mp4
ILIAD+5e938e3b+2023-07-18-14h-04m-48s

2023-06-14/Wed_Jun_14_16:26:36_2023

videos/5030472d/2023-06-11/Sun_Jun_11_15:52:37_2023/27904255.mp4
ILIAD

"""

def path_to_link(link: int) -> str:
    return "/".join(f"panda_link{i}" for i in range(link + 1))


def log_angle_rot(urdf_logger: URDFLogger, link: int, angle_rad: int) -> None:
    entity_path = path_to_link(link)
    start_translation, start_rotation_mat = urdf_logger.entity_to_transform[entity_path]

    link_to_rot_axis = np.array(
        [
            [1, 0, 0], # unused
            [0, 0, 1],  # 1
            [0, 0, 1],  # 2
            [0, 0, 1],  # 3
            [0, 0, 1],  # 4
            [0, 0, 1],  # 5
            [0, 0, 1],  # 6
            [0, 0, 1],  # 7
            [0, 0, 1],  # 8
        ]
    )
    vec = np.array(link_to_rot_axis[link] * angle_rad)
    rot = st.Rotation.from_rotvec(vec).as_matrix()
    rotation_mat = start_rotation_mat @ rot
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
        
        self.zed = zed

    def get_next_frame(self) -> tuple[np.array, np.array, np.array] |  None:
        left_image = sl.Mat()
        right_image = sl.Mat()
        depth_image = sl.Mat()

        rt_param = sl.RuntimeParameters()
        err = self.zed.grab(rt_param)
        if err == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
            left_image = np.array(left_image.numpy())

            self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            right_image = np.array(right_image.numpy())

            self.zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            depth_image = np.array(depth_image.numpy())
            return (left_image, right_image, depth_image)
        else:
            return None

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

        camera_names = ["ext1", "ext2", "wrist"]

        self.serial = {
            camera_name: self.metadata[f'{camera_name}_cam_serial'] for camera_name in camera_names
        }

        self.joint_positions = self.trajectory["observation"]["robot_state"]["joint_positions"]
        self.joint_velocities = self.trajectory["observation"]["robot_state"]["joint_velocities"]

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

    def log_robot_states(self, urdf_logger: URDFLogger):
        time_stamps_nanos = self.trajectory["observation"]["timestamp"]["robot_state"]["robot_timestamp_nanos"]
        time_stamps_seconds = self.trajectory["observation"]["timestamp"]["robot_state"]["robot_timestamp_seconds"]
        for i in range(self.trajectory_length):
            time_stamp = time_stamps_seconds[i]*int(1e9) + time_stamps_nanos[i]
            rr.set_time_nanos("real_time", time_stamp)

            if i == 0:
                # We want to log the robot here so that it appears in the right timeline
                urdf_logger.log()

            for (joint_idx, angle) in enumerate(self.joint_positions[i]):
                log_angle_rot(urdf_logger, joint_idx+1, angle)

            for (camera_name, camera) in self.cameras.items():
                time_stamp_camera = self.trajectory["observation"]["timestamp"]["cameras"][f"{self.serial[camera_name]}_estimated_capture"][i]
                rr.set_time_nanos("real_time", time_stamp_camera*int(1e6))

                extrinsics_left = self.trajectory["observation"]["camera_extrinsics"][f"{self.serial[camera_name]}_left"][i]
                rotation = Rotation.from_euler('xyz', np.array(extrinsics_left[3:])).as_matrix()

                rr.log(f"{camera_name}_camera/left", rr.Pinhole(
                    image_from_camera=camera.left_intrinsic_mat,
                )),
                rr.log(f"{camera_name}_camera/left", rr.Transform3D(
                    translation=np.array(extrinsics_left[:3]),
                    mat3x3=rotation,
                )),

                extrinsics_right = self.trajectory["observation"]["camera_extrinsics"][f"{self.serial[camera_name]}_right"][i]
                rotation = Rotation.from_euler('xyz', np.array(extrinsics_right[3:])).as_matrix()

                rr.log(f"{camera_name}_camera/right", rr.Pinhole(
                    image_from_camera=camera.right_intrinsic_mat,
                )),
                rr.log(f"{camera_name}_camera/right", rr.Transform3D(
                    translation=np.array(extrinsics_right[:3]),
                    mat3x3=rotation,
                )),

                depth_translation = (extrinsics_left[:3]+extrinsics_right[:3])/2
                rotation = Rotation.from_euler('xyz', np.array(extrinsics_right[3:])).as_matrix()

                rr.log(f"{camera_name}_camera/depth", rr.Pinhole(
                    image_from_camera=camera.left_intrinsic_mat,
                )),
                rr.log(f"{camera_name}_camera/depth", rr.Transform3D(
                    translation=depth_translation,
                    mat3x3=rotation,
                )),

                frames = camera.get_next_frame()
                if frames:
                    left_image, right_image, depth_image = frames
                    depth_image[depth_image > 1.5] = 0

                    rr.log(f"{camera_name}_camera/left", rr.Image(left_image))
                    rr.log(f"{camera_name}_camera/right", rr.Image(right_image))
                    rr.log(f"{camera_name}_camera/depth", rr.DepthImage(depth_image))


def main() -> None:
    rr.init("DROID-visualized")
    rr.connect()

    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--urdf", default="franka_description/panda.urdf", type=Path)
    args = parser.parse_args()

    scene = DROIDScene(args.data)    
    
    urdf_logger = URDFLogger(args.urdf)
    
    scene.log_robot_states(urdf_logger)
    

    return

if __name__ == "__main__":
    main()
