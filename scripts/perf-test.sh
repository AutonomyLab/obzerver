#!/bin/bash
set -e

OPENCV_DIR=/home/autolab/Dev/opencv/opencv-2.4.10.1/build
TMP_DIR=`mktemp -d`

echo "OpenCV Dir: $OPENCV_DIR"
echo "Temp Dir: $TMP_DIR"

cd $TMP_DIR

echo "Installing dependencies ..."

sudo apt-get update -qq
# Except for glog, all other are ccv deps (not all required though)
sudo apt-get install -qy libgoogle-glog-dev clang libjpeg-dev libpng-dev libdispatch-dev libgsl0-dev liblas-dev libfftw3-dev liblinear-dev libavcodec-dev libavformat-dev libswscale-dev

echo "Downloading samples ..."
wget --quiet -nc http://autolab.cmpt.sfu.ca/files/waving_sample_1.tar.gz
tar xfz waving_sample_1.tar.gz

echo "Cloning obzerver ..."
git clone --quiet --recursive https://bitbucket.org/mani-monaj/obzerver.git
cd obzerver
mkdir build
cd build

echo "Building obzerver ..."
cmake -DUSE_CUSTOM_OPENCV=1 -DOpenCV_DIR=$OPENCV_DIR -DCMAKE_BUILD_TYPE=Release ..
make -j2

cd ../..
mkdir log
mkdir result

for file in *.avi
do
  echo "Processing $file ..."
  ./obzerver/build/demo --fps=29.97 -v $file -l ./log/$file.log 1>&2 2>./result/$file.txt
done
