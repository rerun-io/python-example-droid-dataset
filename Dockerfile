FROM stereolabs/zed:4.1-devel-cuda12.1-ubuntu22.04

RUN apt update -y && apt upgrade -y
RUN apt install -y fish python3-pip python3-opencv gsutil

WORKDIR /root/droid-example

COPY requirements.txt .

RUN pip install -r requirements.txt
