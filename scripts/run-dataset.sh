#!/bin/bash
set -e

DEMO=/home/autolab/Dev/obzerver/build/demo
#CONFIG=/home/mani/Dev/obzerver/config/ucf_arg.ini
#CONFIG=/home/autolab/Dev/obzerver/config/kth.ini
CONFIG=$1
DATA=$2
#DATA=/home/autolab/datasets/KTH/walking
#DATA=/local_home/datasets/UCF_ARG/aerial_clips/waving

#TMP_DIR=`mktemp -d`
#TMP_DIR="/home/autolab/tmp/kth/walking"
TMP_DIR=$3
LOG_DIR=$TMP_DIR/logs
RES_DIR=$TMP_DIR/results

echo "Using executable: $DEMO"
echo "Config file: $CONFIG"
echo "Data folder: $DATA"
echo "Log Folder: $LOG_DIR"
echo "Result folder: $RES_DIR"

mkdir -p $LOG_DIR
mkdir -p $RES_DIR

cd $DATA

for file in *.avi
do
  echo "Processing $file ..."
  $DEMO \
    --fps=29.97 \
    --video $file \
    --logfile $LOG_DIR/$file.log \
    --config $CONFIG \
    --eval.enabled \
    --eval.file $RES_DIR/$file.png
    1>&2 2>$LOG_DIR/$file.log.stderr
done
