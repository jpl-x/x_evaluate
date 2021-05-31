#!/usr/bin/env bash

BASE_DIR=$(pwd -P)

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
SCRIPT_PATH="$( cd -- "$(dirname "$SCRIPT_PATH/../test/evaluate.py")" >/dev/null 2>&1 ; pwd -P )"

EXPLORE_XVIO_PATCH_SIZE=0
EXPLORE_XVIO_IMU_OFFSET=0
EXPLORE_XVIO_MSCKF_BASELINE=0
EXPLORE_XVIO_RHO_0=0

DATE=$(date '+%Y-%m-%d')

echo "We are in '$BASE_DIR'"
echo "Assuming location of evaluation script to be '$SCRIPT_PATH'"
echo "Today is $DATE"

echo "Taking first argument as results folder: '$1'"

cd "$SCRIPT_PATH" || exit

trap 'cd "$BASE_DIR"' EXIT

if [ $EXPLORE_XVIO_PATCH_SIZE -gt 0 ]
then
  echo
  echo "Performing frame based XVIO patch size exploration"
  echo

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/000-baseline --frontend \
   XVIO --name "XVIO baseline"

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/001-patch-size-3 --frontend \
   XVIO --name "XVIO p=3" --overrides block_half_length=3 margin=3

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/002-patch-size-5 --frontend \
   XVIO --name "XVIO p=5" --overrides block_half_length=5 margin=5

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/003-patch-size-6 --frontend \
   XVIO --name "XVIO p=6" --overrides block_half_length=6 margin=6

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/004-patch-size-8 --frontend \
   XVIO --name "XVIO p=8" --overrides block_half_length=8 margin=8

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/005-patch-size-10 --frontend \
   XVIO --name "XVIO p=10" --overrides block_half_length=10 margin=10

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/006-patch-size-12 --frontend \
   XVIO --name "XVIO p=12" --overrides block_half_length=12 margin=12

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/007-patch-size-15 --frontend \
   XVIO --name "XVIO p=15" --overrides block_half_length=15 margin=15

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/008-patch-size-16 --frontend \
   XVIO --name "XVIO p=16" --overrides block_half_length=16 margin=16

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/009-patch-size-18 --frontend \
   XVIO --name "XVIO p=18" --overrides block_half_length=18 margin=18

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/010-patch-size-20 --frontend \
   XVIO --name "XVIO p=20" --overrides block_half_length=20 margin=20

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/011-patch-size-22 --frontend \
   XVIO --name "XVIO p=22" --overrides block_half_length=22 margin=22

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-patch-size/012-patch-size-25 --frontend \
   XVIO --name "XVIO p=25" --overrides block_half_length=25 margin=25

  python ../scripts/compare.py --input_folder $1/$DATE-xvio-patch-size/ --output_folder $1/$DATE-xvio-patch-size/results

fi


if [ $EXPLORE_XVIO_IMU_OFFSET -gt 0 ]
then
  echo
  echo "Performing frame based XVIO IMU time offset calibration"
  echo

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/000-baseline --frontend \
   XVIO --name "XVIO baseline"

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/001-offset-0.01 --frontend \
   XVIO --name "XVIO IMU offset -0.01" --overrides cam1_time_offset=-0.01

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/002-offset-0.009 --frontend \
   XVIO --name "XVIO IMU offset -0.009" --overrides cam1_time_offset=-0.009

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/003-offset-0.008 --frontend \
   XVIO --name "XVIO IMU offset -0.008" --overrides cam1_time_offset=-0.008

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/004-offset-0.007 --frontend \
   XVIO --name "XVIO IMU offset -0.007" --overrides cam1_time_offset=-0.007

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/005-offset-0.006 --frontend \
   XVIO --name "XVIO IMU offset -0.006" --overrides cam1_time_offset=-0.006

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/006-offset-0.005 --frontend \
   XVIO --name "XVIO IMU offset -0.005" --overrides cam1_time_offset=-0.005

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/007-offset-0.004 --frontend \
   XVIO --name "XVIO IMU offset -0.004" --overrides cam1_time_offset=-0.004

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/008-offset-0.003 --frontend \
   XVIO --name "XVIO IMU offset -0.003" --overrides cam1_time_offset=-0.003

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/009-offset-0.002 --frontend \
   XVIO --name "XVIO IMU offset -0.002" --overrides cam1_time_offset=-0.002

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/010-offset-0.001 --frontend \
   XVIO --name "XVIO IMU offset -0.001" --overrides cam1_time_offset=-0.001

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/011-offset0.0 --frontend \
   XVIO --name "XVIO IMU offset +0.0" --overrides cam1_time_offset=0.0

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/012-offset0.001 --frontend \
   XVIO --name "XVIO IMU offset +0.001" --overrides cam1_time_offset=0.001

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/013-offset0.002 --frontend \
   XVIO --name "XVIO IMU offset +0.002" --overrides cam1_time_offset=0.002

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/014-offset0.003 --frontend \
   XVIO --name "XVIO IMU offset +0.003" --overrides cam1_time_offset=0.003

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/015-offset0.004 --frontend \
   XVIO --name "XVIO IMU offset +0.004" --overrides cam1_time_offset=0.004

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/016-offset0.005 --frontend \
   XVIO --name "XVIO IMU offset +0.005" --overrides cam1_time_offset=0.005

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/017-offset0.006 --frontend \
   XVIO --name "XVIO IMU offset +0.006" --overrides cam1_time_offset=0.006

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/018-offset0.007 --frontend \
   XVIO --name "XVIO IMU offset +0.007" --overrides cam1_time_offset=0.007

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/019-offset0.008 --frontend \
   XVIO --name "XVIO IMU offset +0.008" --overrides cam1_time_offset=0.008

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/020-offset0.009 --frontend \
   XVIO --name "XVIO IMU offset +0.009" --overrides cam1_time_offset=0.009

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-imu-offset/021-offset0.01 --frontend \
   XVIO --name "XVIO IMU offset +0.01" --overrides cam1_time_offset=0.01

  python ../scripts/compare.py --input_folder $1/$DATE-xvio-imu-offset/ --output_folder $1/$DATE-xvio-imu-offset/results

fi


if [ $EXPLORE_XVIO_MSCKF_BASELINE -gt 0 ]
then
  echo
  echo "Performing frame based XVIO MSCKF baseline exploration"
  echo

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-msckf-baseline/000-baseline --frontend \
   XVIO --name "XVIO baseline"

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-msckf-baseline/001-msckf-baseline-5 --frontend \
   XVIO --name "XVIO baseline" --overrides msckf_baseline=5

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-msckf-baseline/002-msckf-baseline-10 --frontend \
   XVIO --name "XVIO baseline" --overrides msckf_baseline=10

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-msckf-baseline/003-msckf-baseline-15 --frontend \
   XVIO --name "XVIO baseline" --overrides msckf_baseline=15

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-msckf-baseline/004-msckf-baseline-20 --frontend \
   XVIO --name "XVIO baseline" --overrides msckf_baseline=20

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-msckf-baseline/005-msckf-baseline-25 --frontend \
   XVIO --name "XVIO baseline" --overrides msckf_baseline=25

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-msckf-baseline/006-msckf-baseline-30 --frontend \
   XVIO --name "XVIO baseline" --overrides msckf_baseline=30

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-msckf-baseline/007-msckf-baseline-35 --frontend \
   XVIO --name "XVIO baseline" --overrides msckf_baseline=35

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-msckf-baseline/008-msckf-baseline-40 --frontend \
   XVIO --name "XVIO baseline" --overrides msckf_baseline=40

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-msckf-baseline/009-msckf-baseline-45 --frontend \
   XVIO --name "XVIO baseline" --overrides msckf_baseline=45

   python ../scripts/compare.py --input_folder $1/$DATE-xvio-msckf-baseline/ --output_folder $1/$DATE-xvio-msckf-baseline/results

fi


if [ $EXPLORE_XVIO_RHO_0 -gt 0 ]
then
  echo
  echo "Performing frame based XVIO rho_0 exploration"
  echo

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-rho-0/000-baseline --frontend \
   XVIO --name "XVIO baseline"

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-rho-0/001-rho-0.3 --frontend \
   XVIO --name "XVIO baseline" --overrides rho_0=0.3 sigma_rho_0=0.15

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-rho-0/001-rho-0.4 --frontend \
   XVIO --name "XVIO baseline" --overrides rho_0=0.4 sigma_rho_0=0.2

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-rho-0/001-rho-0.5 --frontend \
   XVIO --name "XVIO baseline" --overrides rho_0=0.5 sigma_rho_0=0.25

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-rho-0/001-rho-0.6 --frontend \
   XVIO --name "XVIO baseline" --overrides rho_0=0.6 sigma_rho_0=0.3

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-rho-0/001-rho-0.7 --frontend \
   XVIO --name "XVIO baseline" --overrides rho_0=0.7 sigma_rho_0=0.35

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-rho-0/001-rho-0.8 --frontend \
   XVIO --name "XVIO baseline" --overrides rho_0=0.8 sigma_rho_0=0.4

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-rho-0/001-rho-0.9 --frontend \
   XVIO --name "XVIO baseline" --overrides rho_0=0.9 sigma_rho_0=0.45

  python evaluate.py --configuration evaluate.yaml --output_folder $1/$DATE-xvio-rho-0/001-rho-1.0 --frontend \
   XVIO --name "XVIO baseline" --overrides rho_0=1.0 sigma_rho_0=0.5

   python ../scripts/compare.py --input_folder $1/$DATE-xvio-rho-0/ --output_folder $1/$DATE-xvio-rho-0/results

fi
