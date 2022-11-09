#! /bin/bash
# do not run with sudo otherwise wrong HOME will be used
# IMPORTANT ACTIVATE VENV !!
# pyenv activate YOURENVIRONMENT
# instal CUDA before continue
VER=4.6.0
PYTHON_VERSION=3.9.15
PYTHON_LIB=libpython3.9.a
CUDA_VERSION=11.8
CORES=2
# /home/railcar/.pyenv/versions/rcars
PREFIX=`pyenv prefix`
# /home/railcar/.pyenv/versions/3.9.15
PREFIX_MAIN=`pyenv virtualenv-prefix`
# for grtx1070 ARCH_BIN is 6.1 for others check nvidia site
ARCH_BIN=6.1
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation-network
cd ~/Downloads
pip install -U numpy
echo "Make sure cuda is installed right"
nvccs --version
nvidia-smi

echo "Script for installing the OpenCV $VER on Ubuntu 22.04 LTS"
echo "Updating the OS..."
sudo apt update 
sudo apt upgrade -y
echo "Installing dependencies..."
sudo apt-get install build-essential cmake pkg-config unzip yasm git checkinstall
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev libopenjp2-7-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get install libfaac-dev libvorbis-dev
sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libtbb-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libprotobuf-dev protobuf-compiler
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
sudo apt-get install libgtkglext1 libgtkglext1-dev
sudo apt-get install libopenblas-dev liblapacke-dev libva-dev libopenjp2-tools libopenjpip-dec-server libopenjpip-server libqt5opengl5-dev libtesseract-dev 
sudo apt-get install cmake libeigen3-dev libgflags-dev libgoogle-glog-dev libsuitesparse-dev libatlas-base-dev libmetis-dev
# sudo apt install python-is-python3

echo "To install ceres solver manually uncomment lines in script"
# git clone https://ceres-solver.googlesource.com/ceres-solver
# cd ceres-solver
# mkdir build && cd build
# cmake ..
# make -j${CORES}
# make test
# sudo make install

echo "Fetching and unpacking OpenCV $VER..."
mkdir -p $HOME/Downloads/repositories
cd $HOME/Downloads/repositories
wget -O opencv.zip https://github.com/opencv/opencv/archive/${VER}.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${VER}.zip
unzip opencv.zip
unzip opencv_contrib.zip
rm opencv.zip
rm opencv_contrib.zip
cd opencv-${VER}
mkdir -p build
cd build

# hostname=$(sudo cat /etc/hostname)
echo "Compiling OpenCV $VER... this will take several minutes..."
rm CMakeCache.txt
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D BUILD_NEW_PYTHON_SUPPORT=ON \
	-D BUILD_opencv_python3=ON \
	-D BUILD_opencv_legacy=OFF \
	-D CMAKE_INSTALL_PREFIX=$HOME/.pyenv/versions/${PYTHON_VERSION}/usr/local/ \
	-D PYTHON_EXECUTABLE=$HOME/.pyenv/versions/${PYTHON_VERSION}/bin/python \
	-D PYTHON_LIBRARY=$HOME/.pyenv/versions/${PYTHON_VERSION}/lib/${PYTHON_LIB} \
	-D PYTHON_INCLUDE_DIR=$HOME/.pyenv/versions/${PYTHON_VERSION}/include/python3.9 \
	-D PYTHON_INCLUDE_DIRS=$HOME/.pyenv/versions/${PYTHON_VERSION}/include/python3.9 \
	-D PYTHON_INCLUDE_DIRS2=$HOME/.pyenv/versions/${PYTHON_VERSION}/include/python3.9 \
	-D INCLUDE_DIRS=$HOME/.pyenv/versions/${PYTHON_VERSION}/include/python3.9 \
	-D INCLUDE_DIRS2=$HOME/.pyenv/versions/${PYTHON_VERSION}/include/python3.9 \
	-D PYTHON_PACKAGES_PATH=$HOME/.pyenv/versions/${PYTHON_VERSION}/lib/python3.9/site-packages \
	-D PYTHON_NUMPY_INCLUDE_DIR=$HOME/.pyenv/versions/${PYTHON_VERSION}/lib/python3.9/site-packages/numpy/core/include \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D WITH_TBB=ON \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D CUDA_ARCH_BIN=${ARCH_BIN} \
	-D WITH_CUBLAS=1 \
	-D WITH_OPENGL=ON \
	-D WITH_QT=ON \
	-D OpenGL_GL_PREFERENCE=LEGACY \
	-D OPENCV_EXTRA_MODULES_PATH=$HOME/Downloads/repositories/opencv_contrib-${VER}/modules \
	-D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
	-D BUILD_EXAMPLES=ON ..


echo "Compilation has started ..."
make -j$CORES
echo "Installing OpenCV $VER ..."
sudo make -j$CORES install
# sudo cp unix-install/opencv4.pc /usr/lib/x86_64-linux-gnu/pkgconfig/
# sudo cp unix-install/opencv4.pc /usr/local/lib/pkgconfig
sudo ldconfig
source $HOME/.bashrc
echo "Installation complete"


echo "Create symlinks"
ls /usr/local/lib/python3.9/site-packages/cv2/python-3.9/
ls $PREFIX_MAIN/lib/python3.9/site-packages/cv2/python-3.9/

echo "CMake sure so file for cv2 is present in any of those locations. Then uncomment in part 2 the right one and complete tests"



# echo "Check that it works. Quit with quit()"
# python
# import cv2
# cv2.__version__


	# -D CMAKE_INSTALL_PREFIX=/usr/local \


# cmake -D CMAKE_BUILD_TYPE=RELEASE \
# 	-D CMAKE_INSTALL_PREFIX=/usr/local \
# 	-D INSTALL_PYTHON_EXAMPLES=OFF \
# 	-D INSTALL_C_EXAMPLES=OFF \
# 	-D OPENCV_ENABLE_NONFREE=ON \
# 	-D OPENCV_EXTRA_MODULES_PATH=~/repositories/opencv_contrib-${VER}/modules \
# 	-D BUILD_NEW_PYTHON_SUPPORT=ON \
# 	-D PYTHON3_PACKAGES_PATH="$PREFIX"/lib/python${PYTHON_VERSION}/site-packages \
# 	-D BUILD_opencv_python3=ON \
# 	-D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
# 	# -D PYTHON_DEFAULT_EXECUTABLE=/home/jlee/pyenvs/opencv_cuda/bin/python \
# 	-D PYTHON3_LIBRARY="$PREFIX_MAIN"/lib/libpython3.9 \
#     -D PYTHON3_INCLUDE_DIR="$PREFIX_MAIN"/include/python3.9 \
# 	-D OPENCV_GENERATE_PKGCONFIG=ON \
# 	-D OPENCV_PC_FILE_NAME=opencv4.pc \
# 	-D WITH_TBB=ON \
# 	-D ENABLE_FAST_MATH=1 \
# 	-D CUDA_FAST_MATH=1 \
# 	-D WITH_CUBLAS=1 \
# 	-D WITH_CUDA=ON \
# 	-D BUILD_opencv_cudacodec=OFF \
# 	-D WITH_CUDNN=ON \
# 	-D OPENCV_DNN_CUDA=ON \
# 	-D CUDA_ARCH_BIN=6.1 \
# 	-D WITH_V4L=ON \
# 	-D WITH_QT=OFF \
# 	-D WITH_OPENGL=ON \
# 	-D WITH_QT=OFF \
# 	-D WITH_GSTREAMER=ON \
# 	-D WITH_FFMPEG=ON \
# 	-D WITH_OPENCL=ON \
# 	-D OPENCV_ENABLE_NONFREE=ON \
# 	-D ENABLE_PRECOMPILED_HEADERS=YES \
# 	-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
# 	-D BUILD_EXAMPLES=ON ..



# -D CMAKE_INSTALL_PREFIX=~/.pyenv/versions/${PYTHON_VERSION}/usr/local/ \
# -D INSTALL_C_EXAMPLES=OFF \
# -D BUILD_NEW_PYTHON_SUPPORT=ON \
# -D BUILD_opencv_python3=ON \
# -D BUILD_opencv_legacy=OFF \
# -D INSTALL_PYTHON_EXAMPLES=ON \
# -D OPENCV_EXTRA_MODULES_PATH=~/tmp/opencv_contrib/modules \
# -D BUILD_EXAMPLES=ON \
# -D PYTHON_EXECUTABLE=~/.pyenv/versions/${PYTHON_VERSION}/bin/python \
# -D PYTHON_LIBRARY=~/.pyenv/versions/${PYTHON_VERSION}/lib/${PYTHON_LIB} \
# -D PYTHON_INCLUDE_DIR=~/.pyenv/versions/${PYTHON_VERSION}/include/python3.5m \
# -D PYTHON_INCLUDE_DIRS=~/.pyenv/versions/${PYTHON_VERSION}/include/python3.5m \
# -D PYTHON_INCLUDE_DIRS2=~/.pyenv/versions/${PYTHON_VERSION}/include/python3.5m \
# -D INCLUDE_DIRS=~/.pyenv/versions/${PYTHON_VERSION}/include/python3.5m \
# -D INCLUDE_DIRS2=~/.pyenv/versions/${PYTHON_VERSION}/include/python3.5m \
# -D PYTHON_PACKAGES_PATH=~/.pyenv/versions/${PYTHON_VERSION}/lib/python3.5/site-packages \
# -D PYTHON_NUMPY_INCLUDE_DIR=~/.pyenv/versions/${PYTHON_VERSION}/lib/python3.5/site-packages/numpy/core/include