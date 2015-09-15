#!/bin/bash
set -e

VERSION=3.0.0

sudo apt-get update -q
sudo apt-get install -y build-essential libgtk2.0-dev pkg-config libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen3-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev

wget -nc https://github.com/Itseez/opencv/archive/$VERSION.tar.gz
tar xfz $VERSION.tar.gz
cd opencv-$VERSION
rm -rf build install
mkdir build install
export OPENCV_INSTALL_PREFIX=`pwd`/install
cd build
cmake -D CMAKE_INSTALL_PREFIX=$OPENCV_INSTALL_PREFIX ENABLE_SSE41=ON -D ENABLE_SSE42=ON -D ENABLE_SSSE3=ON ENABLE_SSE=ON ENABLE_SSE2=ON ENABLE_SSE3=ON ENABLE_AVX=ON ENABLE_AVX2=ON -D CMAKE_BUILD_TYPE=Release -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -D WITH_QT=OFF -D WITH_OPENGL=OFF -D WITH_VTK=OFF -D BUILD_DOCS=OFF -D BUILD_TESTS=OFF -D BUILD_WITH_DEBUG_INFO=ON -D WITH_CUDA=OFF -D WITH_CUFFT=OFF -D WITH_EIGEN=ON -D WITH_OPENCL=OFF ..
make all install -j 8
