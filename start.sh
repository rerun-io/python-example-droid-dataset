#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ "$(docker ps -a | grep ubuntu_container)" ]; then
  echo
  echo
  echo "Docker container is already running."
  echo "Use ./attach_to_container.sh to enter the container or use ./shutdown_container.sh to shutdown the container."
  echo "To restart container after shutdown, use ./start_container.sh" 
  echo
  echo

  exit 1
fi

# Run the container
docker run -it -d --rm --privileged \
  --name ubuntu_container \
  --net=host \
  --gpus all \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --volume="$PWD/src:/root/droid-example/src:rw" \
  --volume="$PWD/data:/root/droid-example/data:rw" \
  --volume="$PWD/franka_description:/root/droid-example/franka_description:rw" \
  -e DISPLAY="localhost:10.0" \
  ubuntu_container

echo
echo
echo "Docker container started!"
echo "Use ./attach.sh to enter the container!"
echo
echo

