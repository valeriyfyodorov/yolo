#! /bin/bash
# run with sudo
VER=4.6.0
PYTHON_VERSION=3.9.15
CUDA_VERSION=11.8
CORES=2
PREFIX=`pyenv prefix`
PREFIX_MAIN=`pyenv virtualenv-prefix`
ARCH_BIN=6.1
# CUDNN_VERSION_LEFT=ubuntu2204
CUDNN_VERSION_RIGHT=8.6.0.163
KEYRING_VERSION=1.0-1

# run tests
cd ~/Downloads
cp -r /usr/src/cudnn_samples_v8/ $HOME/Downloads/tests/cudnn_samples_v8
cd  $HOME/Downloads/tests/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN

nvccs --version
nvidia-smi
