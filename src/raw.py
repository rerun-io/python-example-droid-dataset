#!/usr/bin/env python3
import pyzed.sl as sl
import numpy as np
from pathlib import Path
import rerun as rr
from scipy.spatial.transform import Rotation
import glob
import h5py
import json
from common import h5_tree, CAMERA_NAMES, log_angle_rot, blueprint_row_images, extract_extrinsics, log_cartesian_velocity, POS_DIM_NAMES
from rerun_loader_urdf import URDFLogger
import argparse

class SVOCamera:
    left_images: list[np.ndarray]
    right_images: list[np.ndarray]
    depth_images: list[np.ndarray]
    width: float
    height: float
    left_dist_coeffs: np.ndarray
    left_intrinsic_mat: np.ndarray

    right_dist_coeffs: np.ndarray
    right_intrinsic_mat: np.ndarray

    def __init__(self, svo_path: Path):
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(str(svo_path))
        init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = 0.2

        zed = sl.Camera()
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Error reading camera data: {err}")

        params = (
            zed.get_camera_information().camera_configuration.calibration_parameters
        )

        # Assumes both the cameras have the same resolution.
        resolution = zed.get_camera_information().camera_configuration.resolution
        self.width = resolution.width
        self.height = resolution.height

        self.left_intrinsic_mat = np.array(
            [
                [params.left_cam.fx, 0, params.left_cam.cx],
                [0, params.left_cam.fy, params.left_cam.cy],
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

    def get_next_frame(self) -> tuple[np.array, np.array, np.array] | None:
        """Gets the the next from both cameras and computes the depth."""

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


class RawScene:
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
        self.action = self.trajectory['action']

        # We ignore the robot_state under action/, don't know why where is two different robot_states.
        self.robot_state = self.trajectory['observation']['robot_state']
        h5_tree(self.trajectory)

        self.trajectory_length = self.metadata["trajectory_length"]

        self.serial = {
            camera_name: self.metadata[f"{camera_name}_cam_serial"]
            for camera_name in CAMERA_NAMES
        }

        self.joint_positions = self.trajectory["observation"]["robot_state"][
            "joint_positions"
        ]
        self.joint_velocities = self.trajectory["observation"]["robot_state"][
            "joint_velocities"
        ]

        self.motor_torques_measured = self.trajectory["observation"]["robot_state"][
            "motor_torques_measured"
        ]

        self.cameras = {}
        for camera_name in CAMERA_NAMES:
            self.cameras[camera_name] = SVOCamera(
                self.dir_path / "recordings" / "SVO" / f"{self.serial[camera_name]}.svo"
            )

    def log_cameras_next(self, i: int) -> None:
        """
        Log data from cameras at step `i`.
        It should be noted that it logs the next camera frames that haven't been 
        read yet, this means that this method must only be called once for each step 
        and it must be called in order (log_cameras_next(0), log_cameras_next(1)). 

        The motivation behind this is to avoid storing all the frames in a `list` because
        that would take up too much memory.
        """

        for camera_name, camera in self.cameras.items():
                time_stamp_camera = self.trajectory["observation"]["timestamp"][
                    "cameras"
                ][f"{self.serial[camera_name]}_estimated_capture"][i]
                rr.set_time_nanos("real_time", time_stamp_camera * int(1e6))

                extrinsics_left = self.trajectory["observation"]["camera_extrinsics"][
                    f"{self.serial[camera_name]}_left"
                ][i]
                rotation = Rotation.from_euler(
                    "xyz", np.array(extrinsics_left[3:])
                ).as_matrix()

                rr.log(
                    f"cameras/{camera_name}/left",
                    rr.Pinhole(
                        image_from_camera=camera.left_intrinsic_mat,
                    ),
                ),
                rr.log(
                    f"cameras/{camera_name}/left",
                    rr.Transform3D(
                        translation=np.array(extrinsics_left[:3]),
                        mat3x3=rotation,
                    ),
                ),
                
                extrinsics_right = self.trajectory["observation"]["camera_extrinsics"][
                    f"{self.serial[camera_name]}_right"
                ][i]
                rotation = Rotation.from_euler(
                    "xyz", np.array(extrinsics_right[3:])
                ).as_matrix()

                rr.log(
                    f"cameras/{camera_name}/right",
                    rr.Pinhole(
                        image_from_camera=camera.right_intrinsic_mat,
                    ),
                ),
                rr.log(
                    f"cameras/{camera_name}/right",
                    rr.Transform3D(
                        translation=np.array(extrinsics_right[:3]),
                        mat3x3=rotation,
                    ),
                ),

                depth_translation = (extrinsics_left[:3] + extrinsics_right[:3]) / 2
                rotation = Rotation.from_euler(
                    "xyz", np.array(extrinsics_right[3:])
                ).as_matrix()

                rr.log(
                    f"cameras/{camera_name}/depth",
                    rr.Pinhole(
                        image_from_camera=camera.left_intrinsic_mat,
                    ),
                ),
                rr.log(
                    f"cameras/{camera_name}/depth",
                    rr.Transform3D(
                        translation=depth_translation,
                        mat3x3=rotation,
                    ),
                ),

                frames = camera.get_next_frame()
                if frames:
                    left_image, right_image, depth_image = frames

                    # Ignore points that are far away.
                    depth_image[depth_image > 1.3] = 0

                    rr.log(f"cameras/{camera_name}/left", rr.Image(left_image))
                    rr.log(f"cameras/{camera_name}/right", rr.Image(right_image))
                    rr.log(f"cameras/{camera_name}/depth", rr.DepthImage(depth_image))

    def log_action(self, i: int) -> None:
        pose = self.trajectory['action']['cartesian_position'][i]
        trans, mat = extract_extrinsics(pose)
        rr.log('action/cartesian_position', rr.Transform3D(translation=trans, mat3x3=mat))
        rr.log('action/cartesian_position', rr.Points3D([0, 0, 0], radii=0.02))

        log_cartesian_velocity('action/cartesian_velocity', self.action['cartesian_velocity'][i])

        rr.log('action/gripper_position', rr.Scalar(self.action['gripper_position'][i]))
        rr.log('action/gripper_velocity', rr.Scalar(self.action['gripper_velocity'][i]))

        for j, vel in enumerate(self.trajectory['action']['cartesian_position'][i]):
            rr.log(f'action/joint_velocity/{j}', rr.Scalar(vel))

        pose = self.trajectory['action']['target_cartesian_position'][i]
        trans, mat = extract_extrinsics(pose)
        rr.log('action/target_cartesian_position', rr.Transform3D(translation=trans, mat3x3=mat))
        rr.log('action/target_cartesian_position', rr.Points3D([0, 0, 0], radii=0.02))

        rr.log('action/target_gripper_position', rr.Scalar(self.action['target_gripper_position'][i]))
        
    def log_robot_state(self, i: int, entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        for joint_idx, angle in enumerate(self.joint_positions[i]):
            log_angle_rot(entity_to_transform, joint_idx + 1, angle)

        rr.log('robot_state/gripper_position', rr.Scalar(self.robot_state['gripper_position'][i]))

        for j, vel in enumerate(self.joint_velocities[i]):
            rr.log(f"robot_state/joint_velocities/{j}", rr.Scalar(vel))

        for j, vel in enumerate(self.robot_state['joint_torques_computed'][i]):
            rr.log(f"robot_state/joint_torques_computed/{j}", rr.Scalar(vel))

        for j, vel in enumerate(self.motor_torques_measured[i]):
            rr.log(f"robot_state/motor_torques_measured/{j}", rr.Scalar(vel))

    def log(self, urdf_logger) -> None:
        time_stamps_nanos = self.trajectory["observation"]["timestamp"]["robot_state"][
            "robot_timestamp_nanos"
        ]
        time_stamps_seconds = self.trajectory["observation"]["timestamp"][
            "robot_state"
        ]["robot_timestamp_seconds"]
        for i in range(self.trajectory_length):
            time_stamp = time_stamps_seconds[i] * int(1e9) + time_stamps_nanos[i]
            rr.set_time_nanos("real_time", time_stamp)

            if i == 0:
                # We want to log the robot model here so that it appears in the right timeline
                urdf_logger.log()

            self.log_action(i)
            self.log_cameras_next(i)
            self.log_robot_state(i, urdf_logger.entity_to_transform)

def blueprint_raw():
    from rerun.blueprint import (
        Blueprint,
        BlueprintPanel,
        Horizontal,
        Vertical,
        SelectionPanel,
        Spatial3DView,
        TimePanel,
        TimeSeriesView,
        Tabs,
    )

    blueprint = Blueprint(
        Horizontal(
            Vertical(
                Spatial3DView(name="robot view", origin="/", contents=["/**"]),
                blueprint_row_images(
                    [
                        "cameras/ext1/left",
                        "cameras/ext1/right",
                        "cameras/ext1/depth",
                        "cameras/wrist/left",
                    ]
                ),
                blueprint_row_images(
                    [
                        "cameras/ext2/left",
                        "cameras/ext2/right",
                        "cameras/ext2/depth",
                        "cameras/wrist/right",
                    ]
                ),
                row_shares=[3, 1, 1],
            ),
            Tabs(
                Vertical(
                    *(
                        TimeSeriesView(origin=f'action/cartesian_velocity/{dim_name}') for dim_name in POS_DIM_NAMES
                    ),
                    name='cartesian_velocity',
                ),
                Vertical(
                    TimeSeriesView(origin='action/', contents=['action/gripper_position', 'action/target_gripper_position']),
                    TimeSeriesView(origin='action/gripper_velocity'),
                    name='action/gripper' 
                ),
                Vertical(
                    *(
                        TimeSeriesView(origin=f'action/joint_velocity/{i}') for i in range(7)
                    ),
                    name='action/joint_velocity'
                ),
                Vertical(
                    *(
                        TimeSeriesView(origin=f'robot_state/joint_torques_computed/{i}') for i in range(7)
                    ),
                    name='joint_torques_computed',
                ),
                Vertical(
                    *(
                        TimeSeriesView(origin=f'robot_state/joint_velocities/{i}') for i in range(7)
                    ),
                    name='robot_state/joint_velocities'
                ),
                Vertical(
                    *(
                        TimeSeriesView(origin=f'robot_state/motor_torques_measured/{i}') for i in range(7)
                    ),
                    name='motor_torques_measured',
                ),
            ),
            column_shares=[3, 2],
        ),
        BlueprintPanel(expanded=False),
        SelectionPanel(expanded=False),
        TimePanel(expanded=False),
        auto_space_views=False,
    )
    return blueprint

def main():
    rr.init("DROID-visualized")
    rr.connect()

    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )

    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--urdf", default="franka_description/panda.urdf", type=Path)
    args = parser.parse_args()

    urdf_logger = URDFLogger(args.urdf)

    from raw import RawScene, blueprint_raw

    raw_scene = RawScene(args.scene)
    rr.send_blueprint(blueprint_raw())
    raw_scene.log(urdf_logger)

if __name__ == "__main__":
    main()
