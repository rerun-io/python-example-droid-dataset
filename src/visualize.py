#!/usr/bin/env python3

from __future__ import annotations

import rerun as rr
import argparse
import numpy as np
import sys
from pathlib import Path
from rerun_loader_urdf import URDFLogger

from rlds import RLDSScene

def main() -> None:
    
    rr.init("DROID-visualized")
    rr.connect()

    parser = argparse.ArgumentParser(
        description="Visualizes the DROID dataset using Rerun."
    )

    parser.add_argument("--raw-data", required=False, type=Path)
    parser.add_argument("--rlds-data", required=False, type=Path)
    parser.add_argument("--urdf", default="franka_description/panda.urdf", type=Path)
    args = parser.parse_args()

    urdf_logger = URDFLogger(args.urdf)

    if args.raw_data is not None:
        from raw import RawScene, blueprint_raw
        raw_scene = RawScene(args.raw_data)
        rr.send_blueprint(blueprint_raw())
        raw_scene.log_robot_states(urdf_logger)
    elif args.rlds_data is not None:
        rlds_scene = RLDSScene(args.rlds_data)
        rlds_scene.log_robot_states(urdf_logger)
    else:
        print("You must specify a path to the data, either by using --raw-data or --rlds-data")
        sys.exit()
        
    return

if __name__ == "__main__":
    main()
