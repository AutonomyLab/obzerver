#!/bin/bash
set -e

OPENCV_DIR=/local_home/opencv/opencv-2.4.10.1/install
#OPENCV_DIR=/home/autolab/Dev/opencv/opencv-2.4.10.1/install

TMP_DIR=`mktemp -d`

echo "OpenCV Dir: $OPENCV_DIR"
echo "Temp Dir: $TMP_DIR"

cd $TMP_DIR

echo "Installing dependencies ..."

sudo apt-get update -qq
# Except for glog, all other are ccv deps (not all required though)
sudo apt-get install -qy libgoogle-glog-dev libgsl0-dev
#sudo apt-get install -qy libgoogle-glog-dev clang libjpeg-dev libpng-dev libdispatch-dev libgsl0-dev liblas-dev libfftw3-dev liblinear-dev libavcodec-dev libavformat-dev libswscale-dev

echo "Downloading samples ..."
wget --quiet -nc http://autolab.cmpt.sfu.ca/files/waving_sample_1.tar.gz
tar xfz waving_sample_1.tar.gz

echo "Cloning obzerver ..."
git clone --quiet --recursive https://bitbucket.org/AutonomyLab/obzerver.git
cd obzerver
mkdir build
cd build

echo "Building obzerver ..."
cmake -DCMAKE_PREFIX_PATH=${OPENCV_DIR} -DCMAKE_BUILD_TYPE=Release ..
make -j2

cd ../..
mkdir log
mkdir result

for file in *.avi
do
  echo "Processing $file ..."
  ./obzerver/build/demo \
    --fps=29.97 \
    --video $file \
    --logfile ./log/$file.log \
    --config ./obzerver/config/ucf_arg.ini \
    #1>&2 2>/dev/null
done

echo "Generating Results ..."

cd log
for file in *
do
  # ignore symlinks
  test -h $file || fgrep "(ms)" $file > ../result/`basename $file`.txt
done

echo "Results in: $TMP_DIR/result"
