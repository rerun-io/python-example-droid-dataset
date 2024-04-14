FROM stereolabs/zed:4.1-devel-cuda12.1-ubuntu22.04

RUN apt update -y && apt upgrade -y
RUN apt install -y fish python3-pip python3-opencv

WORKDIR /root/droid-example

RUN pip install rerun-sdk numpy scipy trimesh urdf_parser_py tensorflow tensorflow-datasets pycollada

