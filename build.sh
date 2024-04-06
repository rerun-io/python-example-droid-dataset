#!/usr/bin/env bash
cd "$(dirname "$0")"

docker build . -t ubuntu_container

