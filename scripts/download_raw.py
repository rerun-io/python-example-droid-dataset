#!/usr/bin/env python3

from pathlib import Path
import subprocess
import argparse
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        "Downloads a succesfull scene from the raw version of the dataset"
    )
    parser.add_argument("--out", default=None, type=Path, help="where to store data, by default it gets puts in data/")
    parser.add_argument(
        "--date", type=str, help="date of recording, format: %Y_%b_%d_%H:%M:%S"
    )

    args = parser.parse_args()

    if args.out is None:
        root_dir = Path(__file__).parent.parent
        target_dir = root_dir / "data" / "droid_raw" / "1.0.1"
    else:
        target_dir = args.out

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    annotations_file_name = "aggregated-annotations-030724.json"
    if not (target_dir / annotations_file_name).exists():
        command = [
            "gsutil",
            "-m",
            "cp",
            f"gs://gresearch/robotics/droid_raw/1.0.1/{annotations_file_name}",
            str(target_dir),
        ]
        print(f'Annotation file not found, running {" ".join(command)}')
        subprocess.run(command)

    date = datetime.strptime(args.date, "%Y-%b-%d_%H:%M:%S")
    formated_date = date.strftime("%Y-%m-%d-%Hh-%Mm-%Ss")

    with open(target_dir / annotations_file_name) as f:
        annotations = json.load(f)

    org = None
    for key in annotations.keys():
        if formated_date in key:
            org = key.split("+")[0]

    rel_path = f"success/{date.year}-{date.month:0>2}-{date.day:0>2}"
    src_path = (
        f"gs://gresearch/robotics/droid_raw/1.0.1/{org}/"
        + rel_path
        + "/"
        + date.strftime("%a_%b_%d_%H:%M:%S_%Y")
    )
    dst_path = target_dir / rel_path
    dst_path.mkdir(parents=True, exist_ok=True)
    command = ["gsutil", "-m", "cp", "-r", src_path, dst_path]
    print(f'Running: "{" ".join(map(str, command))}"')
    subprocess.run(command)

if __name__ == "__main__":
    main()
