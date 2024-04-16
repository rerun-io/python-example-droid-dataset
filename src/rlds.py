#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from rerun_loader_urdf import URDFLogger

class RLDSScene:
    def __init__(self, data: Path):
        pass

    def log_robot_states(self, urdf_logger: URDFLogger):
        pass