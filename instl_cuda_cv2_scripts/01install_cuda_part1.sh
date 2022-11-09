#! /bin/bash
# run with sudo
VER=4.6.0
KEYRING_VERSION=1.0-1
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation-network
cd ~/Downloads
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64//cuda-keyring_1.0-1_all.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64//cuda-keyring_${KEYRING_VERSION}_all.deb
sudo dpkg -i cuda-keyring_${KEYRING_VERSION}_all.deb
sudo apt-get update
sudo apt -y install cuda
sudo apt -y install nvidia-gds

sudo reboot
