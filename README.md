# Visualization of the [DROID dataset](https://droid-dataset.github.io/) using [Rerun](https://www.rerun.io/)

https://github.com/rerun-io/python-example-droid-dataset/assets/28707703/95bdb869-1252-4bf6-af9b-e913829fb949

## Viewing the RLDS version
RLDS is a version of the DROID dataset preprocessed to be more suitable for machine learning models. It contains downscaleded images from the left camera of each stereo camera and removes unnecessary data such as `motor_torques_measured`.

The easiest way to get started is just by running:
```bash
pip install -r requirements.txt
src/rlds.py
```
This will download a sample of 100 episodes and display them in the rerun viewer.

You need Python 3.11 to run this demo (newer versions are not supported right now).

## Viewing the raw dataset with depth images (requires CUDA)
Viewing the raw dataset is a bit trickier because the metadata for stereo cameras are stored in `.svo` files which is a proprietary file format. To read these files one must install the ZED SDK and to run it you must have CUDA installed. To easily do both of these things we will use Docker.

First we must download the data
```bash
mkdir -p data && gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/ILIAD/success/2023-06-11/Sun_Jun_11_15:52:37_2023 data/
```
then we can build and run the docker container.
```bash
pip install -r requirements.txt
rerun &>/dev/null & # Start a rerun viewer outside of the docker container

./build.sh # Build the docker container
./start.sh # Start the docker container
./attach.sh # Enter the docker container

# inside of container...

src/raw.py --scene data/Sun_Jun_11_15:52:37_2023/
```

## Viewing the raw dataset without depth images (doesn't require CUDA)
We can view the raw dataset without using ZED SDK by reading the mp4 files, the problem is that the intrinsic parameters of the cameras are stored in the `.svo` file so we will have to guess what they are.

To download and view an episode run the following:
```bash
mkdir -p data && gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/ILIAD/success/2023-06-11/Sun_Jun_11_15:52:37_2023 data/
pip install -r requirements.txt
src/raw.py --scene data/Sun_Jun_11_15:52:37_2023/
```