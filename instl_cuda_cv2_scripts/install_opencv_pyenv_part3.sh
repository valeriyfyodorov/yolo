#! /bin/bash

PREFIX=`pyenv prefix`
echo "Create symlinks"
echo "make sure you found so file before continue otherwise symlink will lead to null"
# cd ~/pyenvs/opencv_cuda/lib/python3.10/site-packages/
# ln -s /usr/local/lib/python3.9/site-packages/cv2/python-3.10/cv2.cpython-39-x86_64-linux-gnu.so cv2.so
cd ${PREFIX}/lib/python3.9/site-packages/
# use only one of those below, find the so file first
ln -s $HOME/.pyenv/versions/3.9.15/usr/local/lib/python3.9/site-packages/cv2/python-3.9/cv2.cpython-39-x86_64-linux-gnu.so cv2.so
# ln -s /usr/local/lib/python3.9/site-packages/cv2/python-3.9/cv2.cpython-39-x86_64-linux-gnu.so cv2.so
# ln -s $PREFIX_MAIN/lib/python3.9/site-packages/cv2/python-3.9/cv2.cpython-39-x86_64-linux-gnu.so cv2.so

echo "Check that it works. Quit with quit()"
python
import cv2
cv2.__version__

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