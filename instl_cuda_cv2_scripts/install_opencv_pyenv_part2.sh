#! /bin/bash

PREFIX=`pyenv prefix`
echo "Create symlinks"
echo "make sure you found so file before continue otherwise symlink will lead to null"
# cd ~/pyenvs/opencv_cuda/lib/python3.9/site-packages/
ls /usr/local/lib/python3.9/site-packages/cv2/python-3.10/cv2.cpython-*
cd ${PREFIX}/lib/python3.9/site-packages/
# use only one of those below, find the so file first

