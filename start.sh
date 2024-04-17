#!/usr/bin/env bash

docker run -it -d --rm --privileged \
  --name ubuntu_container \
  --net=host \
  --gpus all \
  --volume="$PWD/src:/root/droid-example/src:rw" \
  --volume="$PWD/robotiq_arg85_description:/root/droid-example/robotiq_arg85_description:rw" \
  --volume="$PWD/franka_description:/root/droid-example/franka_description:rw" \
  --volume="$PWD/data:/root/droid-example/data:rw" \
  --volume="$PWD/franka_description:/root/droid-example/franka_description:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  -e DISPLAY=:0 \
  ubuntu_container
