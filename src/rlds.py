#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from rerun_loader_urdf import URDFLogger
from scipy.spatial.transform import Rotation
from common import log_angle_rot, blueprint_row_images
import rerun as rr
import argparse
import sys

# Hide those pesky warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds


class RLDSDataset:
    def __init__(self, data: Path | None = None):
        if data is None:
            print("Warning: no scene picked, downloads the droid_100 dataset")

            ds = tfds.load(
                "droid_100", data_dir="gs://gresearch/robotics", split="train"
            )
        else:
            ds = tfds.builder_from_directory(builder_dir=data).as_dataset()["train"]

        # builder.download_and_prepare()
        # ds = builder.as_dataset()

        self.ds = ds

    def log_images(self, step):
        for cam in [
            "exterior_image_1_left",
            "exterior_image_2_left",
            "wrist_image_left",
        ]:
            rr.log(f"/cameras/{cam}", rr.Image(step["observation"][cam].numpy()))

    def log_robot_states(self, step, entity_to_transform):
        for joint_idx, angle in enumerate(step["observation"]["joint_position"]):
            log_angle_rot(entity_to_transform, joint_idx + 1, angle)

    def log_action_dict(self, step):
        pose = step["action_dict"]["cartesian_position"]
        translation = pose[:3]
        rotation_mat = Rotation.from_euler("xyz", pose[3:]).as_matrix()
        rr.log(
            "/action_dict/cartesian_position",
            rr.Transform3D(translation=translation, mat3x3=rotation_mat),
        )

        for i, vel in enumerate(step["action_dict"]["cartesian_velocity"]):
            rr.log(f"/action_dict/cartesian_velocity/{i}", rr.Scalar(vel))

        for i, vel in enumerate(step["action_dict"]["joint_velocity"]):
            rr.log(f"/action_dict/joint_velocity/{i}", rr.Scalar(vel))

        rr.log(
            "/action_dict/gripper_position",
            rr.Scalar(step["action_dict"]["gripper_position"]),
        )
        rr.log(
            "/action_dict/gripper_velocity",
            rr.Scalar(step["action_dict"]["gripper_velocity"]),
        )
        rr.log("/reward", rr.Scalar(step["reward"]))

    def log_robot_dataset(
        self, entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]]
    ):
        cur_time_ns = 0
        for episode in self.ds:
            for step in episode["steps"]:
                rr.set_time_nanos("real_time", cur_time_ns)
                cur_time_ns += int((1e9 * 1 / 15))
                self.log_images(step)
                self.log_robot_states(step, entity_to_transform)
                self.log_action_dict(step)
                rr.log("discount", rr.Scalar(step["discount"]))

    def blueprint(self):
        from rerun.blueprint import (
            Blueprint,
            Horizontal,
            Vertical,
            Spatial3DView,
            TimeSeriesView,
            Tabs,
            SelectionPanel,
            TimePanel,
        )

        return Blueprint(
            Horizontal(
                Vertical(
                    Spatial3DView(name="spatial view", origin="/", contents=["/**"]),
                    blueprint_row_images(
                        [
                            f"/cameras/{cam}"
                            for cam in [
                                "exterior_image_1_left",
                                "exterior_image_2_left",
                                "wrist_image_left",
                            ]
                        ]
                    ),
                    row_shares=[3, 1],
                ),
                Tabs( # Tabs for all the different time serieses.
                    Vertical(
                        *(
                            TimeSeriesView(origin=f"/action_dict/joint_velocity/{i}")
                            for i in range(7)
                        ),
                        name="joint velocity",
                    ),
                    Vertical(
                        *(
                            TimeSeriesView(origin=f"/action_dict/cartesian_velocity/{i}")
                            for i in range(6)
                        ),
                        name="cartesian position",
                    ),
                    Vertical(
                        TimeSeriesView(origin="/action_dict/gripper_position"),
                        TimeSeriesView(origin="/action_dict/gripper_velocity"),
                        name="gripper",
                    ),
                    TimeSeriesView(origin="/discount"),
                    TimeSeriesView(origin="/reward"),
                ),
                column_shares=[3, 1],
            ),
            SelectionPanel(expanded=False),
            TimePanel(expanded=False),
        )
    
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )

    parser.add_argument("--data", required=False, type=Path)
    parser.add_argument("--urdf", default="franka_description/panda.urdf", type=Path)
    args = parser.parse_args()

    urdf_logger = URDFLogger(args.urdf)
    rlds_scene = RLDSDataset(args.data)

    rr.init("DROID-visualized", spawn=True)

    rr.send_blueprint(rlds_scene.blueprint())

    urdf_logger.log()
    rlds_scene.log_robot_dataset(urdf_logger.entity_to_transform)


if __name__ == "__main__":
    main()
