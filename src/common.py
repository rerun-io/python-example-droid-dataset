#!/usr/bin/env python3

from rerun_loader_urdf import URDFLogger
import numpy as np
import rerun as rr
import scipy.spatial.transform as st
import h5py

CAMERA_NAMES = ["ext1", "ext2", "wrist"]

def blueprint_row_images(origins):
    from rerun.blueprint import Horizontal, Spatial2DView
    return Horizontal(
        *(Spatial2DView(
            name=org,
            origin=org,
        ) for org in origins),
    )

def path_to_link(link: int) -> str:
    return "/".join(f"panda_link{i}" for i in range(link + 1))

def log_angle_rot(urdf_logger: URDFLogger, link: int, angle_rad: int) -> None:
    entity_path = path_to_link(link)
    start_translation, start_rotation_mat = urdf_logger.entity_to_transform[entity_path]
    vec = np.array(np.array([0, 0, 1]) * angle_rad)
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
