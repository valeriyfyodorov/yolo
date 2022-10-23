#! /bin/bash
# run with sudo
VER=4.6.0
CUDA_VERSION=11.8
# CUDNN_VERSION_LEFT=ubuntu2204
CUDNN_VERSION_RIGHT=8.6.0.163
KEYRING_VERSION=1.0-1


nvccs --version
nvidia-smi

cd ~/Downloads
echo 'export PATH=/usr/local/cuda-'${CUDA_VERSION}'/bin${PATH:+:${PATH}}:~/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-'${CUDA_VERSION}'/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
# echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}:~/bin' >> ~/.bashrc

source $HOME/.bashrc
echo "Installation complete"

sudo apt install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev

# Download cuda deb from developer site
# https://developer.nvidia.com/cudnn
# replace with actual file name

# sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-${CUDNN_VERSION_RIGHT}_${KEYRING_VERSION}_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update
# sudo apt install libcudnn8=8.6.0.163-1+cuda11.8
# sudo apt install libcudnn8-dev=8.6.0.163-1+cuda11.8
# sudo apt install libcudnn8-samples=8.6.0.163-1+cuda11.8
sudo apt install libcudnn8=${CUDNN_VERSION_RIGHT}-1+cuda11.8
sudo apt install libcudnn8-dev=${CUDNN_VERSION_RIGHT}-1+cuda11.8
sudo apt install libcudnn8-samples=${CUDNN_VERSION_RIGHT}-1+cuda11.8

sudo reboot
