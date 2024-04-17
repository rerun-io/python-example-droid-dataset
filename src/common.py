#!/usr/bin/env python3

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation
import h5py

CAMERA_NAMES = ["ext1", "ext2", "wrist"]

# Maps indices in X_position and X_velocity to their meaning.
POS_DIM_NAMES = ['x', 'y', 'z', 'rot_x', 'rot_y', 'rot_z']

def blueprint_col_cartesian_velocity(root: str, name: str):
    from rerun.blueprint import Vertical, TimeSeriesView

    if not root.endswith('/'):
        root += '/'

    return Vertical(
        *(
            TimeSeriesView(origin=root+dim_name) for dim_name in POS_DIM_NAMES
        ),
        name=name,
    )

def log_cartesian_velocity(root: str, cartesian_velocity: np.ndarray):
    if not root.endswith('/'):
        root += '/'

    for (vel, name) in zip(cartesian_velocity, POS_DIM_NAMES):
        rr.log(root + name, rr.Scalar(vel))

def blueprint_row_images(origins):
    from rerun.blueprint import Horizontal, Spatial2DView
    return Horizontal(
        *(Spatial2DView(
            name=org,
            origin=org,
        ) for org in origins),
    )

def extract_extrinsics(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Takes a vector with dimension 6 and extracts the translation vector and the rotation matrix"""
    translation = pose[:3]
    rotation = Rotation.from_euler(
        "xyz", np.array(pose[3:])
    ).as_matrix()
    return (translation, rotation)

def path_to_link(link: int) -> str:
    return "/".join(f"panda_link{i}" for i in range(link + 1))

def log_angle_rot(entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]], link: int, angle_rad: int) -> None:
    """Logs an angle for the franka panda robot"""
    entity_path = path_to_link(link)
    
    start_translation, start_rotation_mat = entity_to_transform[entity_path]

    # All angles describe rotations around the transformed z-axis.
    vec = np.array(np.array([0, 0, 1]) * angle_rad)

    rot = Rotation.from_rotvec(vec).as_matrix()
    rotation_mat = start_rotation_mat @ rot

    rr.log(
        entity_path, rr.Transform3D(translation=start_translation, mat3x3=rotation_mat)
    )

def h5_tree(val, pre=""):
    # Taken from https://stackoverflow.com/questions/61133916/is-there-in-python-a-single-function-that-shows-the-full-structure-of-a-hdf5-fi
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
