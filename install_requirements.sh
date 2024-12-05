#!/bin/bash

sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip -y
pip3 install -r requirements.txt
sudo apt install libatlas-base-dev libopencv-dev -y
python3 detect_camera.py