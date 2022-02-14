#!/usr/bin/env bash

BASE_DIR=$(pwd -P)

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
SCRIPT_PATH="$( cd -- "$(dirname "$SCRIPT_PATH/../test/evaluate.py")" >/dev/null 2>&1 ; pwd -P )"

CONFIGURATION="evaluate_wells.yaml --skip_feature_tracking"

COMPARISONS_ONLY=0
QUALITATIVE_COMPARISON=1
EXPLORE_BASELINES=1
EXPLORE_XVIO_PATCH_SIZE=0
EXPLORE_XVIO_IMU_OFFSET=0
EXPLORE_XVIO_MSCKF_BASELINE=0
EXPLORE_XVIO_RHO_0=0
EXPLORE_XVIO_FAST_DETECTION_DELTA=0
EXPLORE_XVIO_NON_MAX_SUPP=0
EXPLORE_XVIO_N_FEAT_MIN=0
EXPLORE_XVIO_OUTLIER_METHOD=0
EXPLORE_XVIO_TILING=0
EXPLORE_XVIO_N_POSES_MAX=0
EXPLORE_XVIO_N_SLAM_FEATURES_MAX=0
EXPLORE_XVIO_SIGMA_IMG=0
EXPLORE_XVIO_IMU_NOISE=0
EXPLORE_XVIO_ACC_SPIKE=0
EXPLORE_XVIO_INITIAL_SIGMA_P=0
EXPLORE_XVIO_INITIAL_SIGMA_V=0
EXPLORE_XVIO_INITIAL_SIGMA_THETA=0
EXPLORE_XVIO_INITIAL_SIGMA_BW=0
EXPLORE_XVIO_INITIAL_SIGMA_BA=0
EXPLORE_EKLT_PATCH_SIZE=0
EXPLORE_EKLT_IMU_OFFSET=0
EXPLORE_EKLT_OUTLIER_REMOVAL=0
EXPLORE_EKLT_TRACKING_QUALITY=0
EXPLORE_EKLT_UPDATE_STRATEGY_N_MSEC=0
EXPLORE_EKLT_UPDATE_STRATEGY_N_EVENTS=0
EXPLORE_EKLT_INTERPOLATION_TIMESTAMP=0
EXPLORE_EKLT_FEATURE_INTERPOLATION=0
EXPLORE_EKLT_FEATURE_INTERPOLATION_RELATIVE_LIMIT=0
EXPLORE_EKLT_FEATURE_INTERPOLATION_ABSOLUTE_LIMIT=0
EXPLORE_EKLT_LINLOG_SCALE=0
EXPLORE_EKLT_PATCH_TIMESTAMP_ASSIGNMENT=0
EXPLORE_EKLT_SIGMA_IMG=0
EXPLORE_EKLT_HARRIS_K=0
EXPLORE_EKLT_HARRIS_QL=0
EXPLORE_HASTE_TYPE=0
EXPLORE_HASTE_HARRIS_K=0
EXPLORE_HASTE_HARRIS_QL=0
EXPLORE_HASTE_OUTLIER_METHOD=0
EXPLORE_HASTE_OUTLIER_METHOD_95=0
EXPLORE_HASTE_OUTLIER_METHOD_98=0
EXPLORE_HASTE_OUTLIER_METHOD_99=0
EXPLORE_HASTE_OUTLIER_METHOD_EVRY_MSG=0
EXPLORE_HASTE_DIFF_HASTE_OUTLIER_METHOD=0
EXPLORE_HASTE_INTERPOLATION_TIMESTAMP=0
EXPLORE_HASTE_FEATURE_INTERPOLATION=0
EXPLORE_HASTE_UPDATE_STRATEGY_N_MSEC=0
EXPLORE_HASTE_UPDATE_STRATEGY_N_EVENTS=0
EXPLORE_HASTE_BEST=0



cleanup () {
  d=`dirname $1/evaluation.pickle`
  echo "Related to $1"
  find $d -type d | grep -v $d$ | grep -v results | sort | while read j
  do
    echo "   going to delete `du -sh $j`"
    rm -rf $j
  done
}



DATE=$2

if [ -z "$2" ]
then
  DATE=$(date '+%Y-%m-%d')
fi

echo "Using $DATE as date"


echo "We are in '$BASE_DIR'"
echo "Assuming location of evaluation script to be '$SCRIPT_PATH'"
echo "Using the following configuration file: '$CONFIGURATION'"
echo "Today is $DATE"

echo "Taking first argument as results folder: '$1'"

cd "$SCRIPT_PATH" || exit

trap 'cd "$BASE_DIR"' EXIT

COMPARISON_SCRIPT="../scripts/compare.py"

if [ $QUALITATIVE_COMPARISON -gt 0 ]
then
  COMPARISON_SCRIPT="../scripts/qualitative_evaluation.py"
fi

if [ $EXPLORE_BASELINES -gt 0 ]
then
  echo
  echo "Running all frontends baselines"
  echo

  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-baselines/000-xvio-baseline --frontend \
     XVIO --name "XVIO"

    cleanup $1/$DATE-baselines/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-baselines/001-eklt-baseline --frontend \
     EKLT --name "EKLT"

    cleanup $1/$DATE-baselines/001-eklt-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-baselines/002-haste-baseline --frontend \
     HASTE --name "HASTE"

    cleanup $1/$DATE-baselines/002-haste-baseline

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-baselines/ --output_folder $1/$DATE-baselines/results

fi


if [ $EXPLORE_HASTE_BEST -gt 0 ]
then
  echo
  echo "Running HASTE best guesses"
  echo

  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-best/001-haste-guess --frontend \
     HASTE --name "HASTE Best 01" --overrides haste_harris_k=0 haste_outlier_method=8 haste_outlier_param1=0.4 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-best/001-haste-guess

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-best/002-haste-guess --frontend \
     HASTE --name "HASTE Best 02" --overrides haste_detection_min_distance=40 haste_harris_k=0

    cleanup $1/$DATE-haste-best/002-haste-guess

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-best/003-haste-guess --frontend \
     HASTE --name "HASTE Best 03" --overrides haste_outlier_param1=0.3 haste_outlier_param2=0.95 haste_detection_min_distance=40 haste_harris_k=0

    cleanup $1/$DATE-haste-best/003-haste-guess

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-best/ --output_folder $1/$DATE-haste-best/results

fi






if [ $EXPLORE_XVIO_PATCH_SIZE -gt 0 ]
then
  echo
  echo "Performing frame based XVIO patch size exploration"
  echo

  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/001-patch-size-3 --frontend \
     XVIO --name "XVIO p=3" --overrides block_half_length=3 margin=3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/002-patch-size-5 --frontend \
     XVIO --name "XVIO p=5" --overrides block_half_length=5 margin=5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/003-patch-size-6 --frontend \
     XVIO --name "XVIO p=6" --overrides block_half_length=6 margin=6

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/004-patch-size-8 --frontend \
     XVIO --name "XVIO p=8" --overrides block_half_length=8 margin=8

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/005-patch-size-10 --frontend \
     XVIO --name "XVIO p=10" --overrides block_half_length=10 margin=10

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/006-patch-size-12 --frontend \
     XVIO --name "XVIO p=12" --overrides block_half_length=12 margin=12

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/007-patch-size-15 --frontend \
     XVIO --name "XVIO p=15" --overrides block_half_length=15 margin=15

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/008-patch-size-16 --frontend \
     XVIO --name "XVIO p=16" --overrides block_half_length=16 margin=16

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/009-patch-size-18 --frontend \
     XVIO --name "XVIO p=18" --overrides block_half_length=18 margin=18

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/010-patch-size-20 --frontend \
     XVIO --name "XVIO p=20" --overrides block_half_length=20 margin=20

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/011-patch-size-22 --frontend \
     XVIO --name "XVIO p=22" --overrides block_half_length=22 margin=22

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-patch-size/012-patch-size-25 --frontend \
     XVIO --name "XVIO p=25" --overrides block_half_length=25 margin=25

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-patch-size/ --output_folder $1/$DATE-xvio-patch-size/results

fi


if [ $EXPLORE_XVIO_IMU_OFFSET -gt 0 ]
then
  echo
  echo "Performing frame based XVIO IMU time offset calibration"
  echo

  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/001-offset0.0027626 --frontend \
    #  XVIO --name "XVIO IMU offset 0.0027626" --overrides cam1_time_offset=0.0027626

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/001-offset-0.0027626 --frontend \
    #  XVIO --name "XVIO IMU offset -0.0027626" --overrides cam1_time_offset=-0.0027626

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/001-offset-0.01 --frontend \
    XVIO --name "XVIO IMU offset -0.01" --overrides cam1_time_offset=-0.01

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/002-offset-0.009 --frontend \
    XVIO --name "XVIO IMU offset -0.009" --overrides cam1_time_offset=-0.009

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/003-offset-0.008 --frontend \
    XVIO --name "XVIO IMU offset -0.008" --overrides cam1_time_offset=-0.008

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/004-offset-0.007 --frontend \
    XVIO --name "XVIO IMU offset -0.007" --overrides cam1_time_offset=-0.007

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/005-offset-0.006 --frontend \
    XVIO --name "XVIO IMU offset -0.006" --overrides cam1_time_offset=-0.006

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/006-offset-0.005 --frontend \
    XVIO --name "XVIO IMU offset -0.005" --overrides cam1_time_offset=-0.005

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/007-offset-0.004 --frontend \
    XVIO --name "XVIO IMU offset -0.004" --overrides cam1_time_offset=-0.004

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/008-offset-0.003 --frontend \
    XVIO --name "XVIO IMU offset -0.003" --overrides cam1_time_offset=-0.003

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/009-offset-0.002 --frontend \
    XVIO --name "XVIO IMU offset -0.002" --overrides cam1_time_offset=-0.002

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/010-offset-0.001 --frontend \
    XVIO --name "XVIO IMU offset -0.001" --overrides cam1_time_offset=-0.001

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/011-offset0.0 --frontend \
    XVIO --name "XVIO IMU offset +0.0" --overrides cam1_time_offset=0.0

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/012-offset0.001 --frontend \
    XVIO --name "XVIO IMU offset +0.001" --overrides cam1_time_offset=0.001

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/013-offset0.002 --frontend \
    XVIO --name "XVIO IMU offset +0.002" --overrides cam1_time_offset=0.002

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/014-offset0.003 --frontend \
    XVIO --name "XVIO IMU offset +0.003" --overrides cam1_time_offset=0.003

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/015-offset0.004 --frontend \
    XVIO --name "XVIO IMU offset +0.004" --overrides cam1_time_offset=0.004

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/016-offset0.005 --frontend \
    XVIO --name "XVIO IMU offset +0.005" --overrides cam1_time_offset=0.005

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/017-offset0.006 --frontend \
    XVIO --name "XVIO IMU offset +0.006" --overrides cam1_time_offset=0.006

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/018-offset0.007 --frontend \
    XVIO --name "XVIO IMU offset +0.007" --overrides cam1_time_offset=0.007

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/019-offset0.008 --frontend \
    XVIO --name "XVIO IMU offset +0.008" --overrides cam1_time_offset=0.008

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/020-offset0.009 --frontend \
    XVIO --name "XVIO IMU offset +0.009" --overrides cam1_time_offset=0.009

   python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-offset/021-offset0.01 --frontend \
    XVIO --name "XVIO IMU offset +0.01" --overrides cam1_time_offset=0.01

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-imu-offset/ --output_folder $1/$DATE-xvio-imu-offset/results

fi


if [ $EXPLORE_XVIO_MSCKF_BASELINE -gt 0 ]
then
  echo
  echo "Performing frame based XVIO MSCKF baseline exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-msckf-baseline/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-msckf-baseline/001-msckf-baseline-5 --frontend \
     XVIO --name "XVIO MSCKF baseline 5" --overrides msckf_baseline=5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-msckf-baseline/002-msckf-baseline-10 --frontend \
     XVIO --name "XVIO MSCKF baseline 10" --overrides msckf_baseline=10

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-msckf-baseline/003-msckf-baseline-15 --frontend \
     XVIO --name "XVIO MSCKF baseline 15" --overrides msckf_baseline=15

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-msckf-baseline/004-msckf-baseline-20 --frontend \
     XVIO --name "XVIO MSCKF baseline 20" --overrides msckf_baseline=20

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-msckf-baseline/005-msckf-baseline-25 --frontend \
     XVIO --name "XVIO MSCKF baseline 25" --overrides msckf_baseline=25

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-msckf-baseline/006-msckf-baseline-30 --frontend \
     XVIO --name "XVIO MSCKF baseline 30" --overrides msckf_baseline=30

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-msckf-baseline/007-msckf-baseline-35 --frontend \
     XVIO --name "XVIO MSCKF baseline 35" --overrides msckf_baseline=35

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-msckf-baseline/008-msckf-baseline-40 --frontend \
     XVIO --name "XVIO MSCKF baseline 40" --overrides msckf_baseline=40

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-msckf-baseline/009-msckf-baseline-45 --frontend \
     XVIO --name "XVIO MSCKF baseline 45" --overrides msckf_baseline=45

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-msckf-baseline/ --output_folder $1/$DATE-xvio-msckf-baseline/results

fi


if [ $EXPLORE_XVIO_RHO_0 -gt 0 ]
then
  echo
  echo "Performing frame based XVIO rho_0 exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/001-rho-0.3 --frontend \
     XVIO --name "XVIO rho_0=0.3" --overrides rho_0=0.3 sigma_rho_0=0.15

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/002-rho-0.5 --frontend \
     XVIO --name "XVIO rho_0=0.5" --overrides rho_0=0.5 sigma_rho_0=0.25

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/003-rho-0.7 --frontend \
     XVIO --name "XVIO rho_0=0.7" --overrides rho_0=0.7 sigma_rho_0=0.35

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/004-rho-0.9 --frontend \
     XVIO --name "XVIO rho_0=0.9" --overrides rho_0=0.9 sigma_rho_0=0.45

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/005-rho-1.1 --frontend \
     XVIO --name "XVIO rho_0=1.1" --overrides rho_0=1.1 sigma_rho_0=0.55

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/006-rho-1.3 --frontend \
     XVIO --name "XVIO rho_0=1.3" --overrides rho_0=1.3 sigma_rho_0=0.65

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/007-rho-1.5 --frontend \
     XVIO --name "XVIO rho_0=1.5" --overrides rho_0=1.5 sigma_rho_0=0.75

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/008-rho-1.7 --frontend \
     XVIO --name "XVIO rho_0=1.7" --overrides rho_0=1.7 sigma_rho_0=0.85

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/009-rho-1.9 --frontend \
     XVIO --name "XVIO rho_0=1.9" --overrides rho_0=1.9 sigma_rho_0=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/010-rho-2.1 --frontend \
     XVIO --name "XVIO rho_0=2.1" --overrides rho_0=2.1 sigma_rho_0=1.05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/011-rho-2.3 --frontend \
     XVIO --name "XVIO rho_0=2.3" --overrides rho_0=2.3 sigma_rho_0=1.15

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-rho-0/012-rho-2.5 --frontend \
     XVIO --name "XVIO rho_0=2.5" --overrides rho_0=2.5 sigma_rho_0=1.25

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-rho-0/ --output_folder $1/$DATE-xvio-rho-0/results

fi


if [ $EXPLORE_XVIO_FAST_DETECTION_DELTA -gt 0 ]
then
  echo
  echo "Performing frame based XVIO fast_detection_delta exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/001-fast-dd-5 --frontend \
     XVIO --name "XVIO fast_dd=5" --overrides fast_detection_delta=5


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/002-fast-dd-7 --frontend \
     XVIO --name "XVIO fast_dd=7" --overrides fast_detection_delta=7


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/003-fast-dd-10 --frontend \
     XVIO --name "XVIO fast_dd=10" --overrides fast_detection_delta=10


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/004-fast-dd-12 --frontend \
     XVIO --name "XVIO fast_dd=12" --overrides fast_detection_delta=12


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/005-fast-dd-13 --frontend \
     XVIO --name "XVIO fast_dd=13" --overrides fast_detection_delta=13


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/006-fast-dd-15 --frontend \
     XVIO --name "XVIO fast_dd=15" --overrides fast_detection_delta=15

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/003-fast-dd-25 --frontend \
    #  XVIO --name "XVIO fast_dd=25" --overrides fast_detection_delta=25

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/004-fast-dd-30 --frontend \
    #  XVIO --name "XVIO fast_dd=30" --overrides fast_detection_delta=30

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/005-fast-dd-35 --frontend \
    #  XVIO --name "XVIO fast_dd=35" --overrides fast_detection_delta=35

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/006-fast-dd-40 --frontend \
    #  XVIO --name "XVIO fast_dd=40" --overrides fast_detection_delta=40

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/007-fast-dd-45 --frontend \
    #  XVIO --name "XVIO fast_dd=45" --overrides fast_detection_delta=45

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-fast-detection-delta/008-fast-dd-50 --frontend \
    #  XVIO --name "XVIO fast_dd=50" --overrides fast_detection_delta=50

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-fast-detection-delta/ --output_folder
  $1/$DATE-xvio-fast-detection-delta/results

fi



if [ $EXPLORE_XVIO_NON_MAX_SUPP -gt 0 ]
then
  echo
  echo "Performing frame based XVIO non_max_supp exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-non-max-supp/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-non-max-supp/001-non-max-supp-false --frontend \
     XVIO --name "XVIO non_max_supp=0" --overrides non_max_supp=False

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-non-max-supp/002-non-max-supp-true --frontend \
     XVIO --name "XVIO non_max_supp=1" --overrides non_max_supp=True

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-non-max-supp/ --output_folder $1/$DATE-xvio-non-max-supp/results

fi


if [ $EXPLORE_XVIO_N_FEAT_MIN -gt 0 ]
then
  echo
  echo "Performing frame based XVIO n_feat_min exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-feat-min/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-feat-min/001-n-feat-min-50 --frontend \
     XVIO --name "XVIO n_feat_min=50" --overrides n_feat_min=50

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-feat-min/002-n-feat-min-200 --frontend \
     XVIO --name "XVIO n_feat_min=200" --overrides n_feat_min=200

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-feat-min/003-n-feat-min-350 --frontend \
     XVIO --name "XVIO n_feat_min=350" --overrides n_feat_min=350

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-feat-min/004-n-feat-min-500 --frontend \
     XVIO --name "XVIO n_feat_min=500" --overrides n_feat_min=500

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-n-feat-min/ --output_folder $1/$DATE-xvio-n-feat-min/results

fi



if [ $EXPLORE_XVIO_OUTLIER_METHOD -gt 0 ]
then
  echo
  echo "Performing frame based XVIO outlier_method exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/001-ransac-px-0.05 --frontend \
     XVIO --name "XVIO RANSAC px=0.05" --overrides outlier_method=8 outlier_param1=0.05 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/002-ransac-px-0.1 --frontend \
     XVIO --name "XVIO RANSAC px=0.1" --overrides outlier_method=8 outlier_param1=0.1 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/003-ransac-px-0.15 --frontend \
     XVIO --name "XVIO RANSAC px=0.15" --overrides outlier_method=8 outlier_param1=0.15 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/004-ransac-px-0.2 --frontend \
     XVIO --name "XVIO RANSAC px=0.2" --overrides outlier_method=8 outlier_param1=0.2 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/005-ransac-px-0.25 --frontend \
     XVIO --name "XVIO RANSAC px=0.25" --overrides outlier_method=8 outlier_param1=0.25 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/006-ransac-px-0.35 --frontend \
     XVIO --name "XVIO RANSAC px=0.35" --overrides outlier_method=8 outlier_param1=0.35 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/007-ransac-px-0.4 --frontend \
     XVIO --name "XVIO RANSAC px=0.4" --overrides outlier_method=8 outlier_param1=0.4 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/008-ransac-px-0.45 --frontend \
     XVIO --name "XVIO RANSAC px=0.45" --overrides outlier_method=8 outlier_param1=0.45 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/009-ransac-px-0.5 --frontend \
     XVIO --name "XVIO RANSAC px=0.5" --overrides outlier_method=8 outlier_param1=0.5 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/010-ransac-px-0.55 --frontend \
     XVIO --name "XVIO RANSAC px=0.55" --overrides outlier_method=8 outlier_param1=0.55 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/011-ransac-px-0.6 --frontend \
     XVIO --name "XVIO RANSAC px=0.6" --overrides outlier_method=8 outlier_param1=0.6 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/012-ransac-px-0.65 --frontend \
     XVIO --name "XVIO RANSAC px=0.65" --overrides outlier_method=8 outlier_param1=0.65 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/013-ransac-px-0.7 --frontend \
     XVIO --name "XVIO RANSAC px=0.7" --overrides outlier_method=8 outlier_param1=0.7 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/014-ransac-px-0.75 --frontend \
     XVIO --name "XVIO RANSAC px=0.75" --overrides outlier_method=8 outlier_param1=0.75 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/015-ransac-px-0.8 --frontend \
     XVIO --name "XVIO RANSAC px=0.8" --overrides outlier_method=8 outlier_param1=0.8 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/016-ransac-px-0.85 --frontend \
     XVIO --name "XVIO RANSAC px=0.85" --overrides outlier_method=8 outlier_param1=0.85 outlier_param2=0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/017-ransac-px-0.05-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.05 p=0.95" --overrides outlier_method=8 outlier_param1=0.05 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/018-ransac-px-0.1-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.1 p=0.95" --overrides outlier_method=8 outlier_param1=0.1 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/019-ransac-px-0.15-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.15 p=0.95" --overrides outlier_method=8 outlier_param1=0.15 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/020-ransac-px-0.2-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.2 p=0.95" --overrides outlier_method=8 outlier_param1=0.2 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/021-ransac-px-0.25-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.25 p=0.95" --overrides outlier_method=8 outlier_param1=0.25 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/022-ransac-px-0.35-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.35 p=0.95" --overrides outlier_method=8 outlier_param1=0.35 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/023-ransac-px-0.4-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.4 p=0.95" --overrides outlier_method=8 outlier_param1=0.4 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/024-ransac-px-0.45-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.45 p=0.95" --overrides outlier_method=8 outlier_param1=0.45 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/025-ransac-px-0.5-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.5 p=0.95" --overrides outlier_method=8 outlier_param1=0.5 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/026-ransac-px-0.55-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.55 p=0.95" --overrides outlier_method=8 outlier_param1=0.55 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/027-ransac-px-0.6-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.6 p=0.95" --overrides outlier_method=8 outlier_param1=0.6 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/028-ransac-px-0.65-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.65 p=0.95" --overrides outlier_method=8 outlier_param1=0.65 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/029-ransac-px-0.7-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.7 p=0.95" --overrides outlier_method=8 outlier_param1=0.7 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/030-ransac-px-0.75-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.75 p=0.95" --overrides outlier_method=8 outlier_param1=0.75 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/031-ransac-px-0.8-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.8 p=0.95" --overrides outlier_method=8 outlier_param1=0.8 outlier_param2=0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/032-ransac-px-0.85-p-0.95 --frontend \
     XVIO --name "XVIO RANSAC px=0.85 p=0.95" --overrides outlier_method=8 outlier_param1=0.85 outlier_param2=0.95
    

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/001-ransac-px-0.575-p-0.98 --frontend \
    #  XVIO --name "XVIO RANSAC px=0.575 p=0.98" --overrides outlier_method=8 outlier_param1=0.575 outlier_param2=0.98

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/002-ransac-px-0.585-p-0.98 --frontend \
    #  XVIO --name "XVIO RANSAC px=0.585 p=0.98" --overrides outlier_method=8 outlier_param1=0.585 outlier_param2=0.98

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/003-ransac-px-0.625-p-0.98 --frontend \
    #  XVIO --name "XVIO RANSAC px=0.625 p=0.98" --overrides outlier_method=8 outlier_param1=0.625 outlier_param2=0.98

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/004-ransac-px-0.635-p-0.98 --frontend \
    #  XVIO --name "XVIO RANSAC px=0.635 p=0.98" --overrides outlier_method=8 outlier_param1=0.635 outlier_param2=0.98

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/005-ransac-p-0.975 --frontend \
    #  XVIO --name "XVIO RANSAC px=0.6 p=0.975" --overrides outlier_method=8 outlier_param1=0.6 outlier_param2=0.975

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/006-ransac-p-0.978 --frontend \
    #  XVIO --name "XVIO RANSAC px=0.6 p=0.978" --overrides outlier_method=8 outlier_param1=0.6 outlier_param2=0.978

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/007-ransac-p-0.982 --frontend \
    #  XVIO --name "XVIO RANSAC px=0.6 p=0.982" --overrides outlier_method=8 outlier_param1=0.6 outlier_param2=0.982

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/008-ransac-p-0.985 --frontend \
    #  XVIO --name "XVIO RANSAC px=0.6 p=0.985" --overrides outlier_method=8 outlier_param1=0.6 outlier_param2=0.985

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/009-ransac-p-0.987 --frontend \
    #  XVIO --name "XVIO RANSAC px=0.6 p=0.987" --overrides outlier_method=8 outlier_param1=0.6 outlier_param2=0.987

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/016-lmeds-p-0.8 --frontend \
    #  XVIO --name "XVIO LMEDS p=0.8" --overrides outlier_method=4 outlier_param1=0.3 outlier_param2=0.8

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/017-lmeds-p-0.9 --frontend \
    #  XVIO --name "XVIO LMEDS p=0.9" --overrides outlier_method=4 outlier_param1=0.3 outlier_param2=0.9

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/018-lmeds-p-0.95 --frontend \
    #  XVIO --name "XVIO LMEDS p=0.95" --overrides outlier_method=4 outlier_param1=0.3 outlier_param2=0.95

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/019-lmeds-p-0.99 --frontend \
    #  XVIO --name "XVIO LMEDS p=0.99" --overrides outlier_method=4 outlier_param1=0.3 outlier_param2=0.99

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-outlier-method/020-lmeds-p-0.999 --frontend \
    #  XVIO --name "XVIO LMEDS p=0.999" --overrides outlier_method=4 outlier_param1=0.3 outlier_param2=0.999

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-outlier-method/ --output_folder $1/$DATE-xvio-outlier-method/results

fi


if [ $EXPLORE_XVIO_TILING -gt 0 ]
then
  echo
  echo "Performing frame based XVIO tiling exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/001-w-1-h-1 --frontend \
     XVIO --name "XVIO tiles w=1 h=1" --overrides n_tiles_h=1 n_tiles_w=1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/002-w-2-h-1 --frontend \
     XVIO --name "XVIO tiles w=2 h=1" --overrides n_tiles_h=1 n_tiles_w=2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/003-w-2-h-2 --frontend \
     XVIO --name "XVIO tiles w=2 h=2" --overrides n_tiles_h=2 n_tiles_w=2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/004-w-3-h-1 --frontend \
     XVIO --name "XVIO tiles w=3 h=1" --overrides n_tiles_h=1 n_tiles_w=3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/005-w-3-h-2 --frontend \
     XVIO --name "XVIO tiles w=3 h=2" --overrides n_tiles_h=2 n_tiles_w=3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/006-w-3-h-3 --frontend \
     XVIO --name "XVIO tiles w=3 h=3" --overrides n_tiles_h=3 n_tiles_w=3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/007-w-4-h-2 --frontend \
     XVIO --name "XVIO tiles w=4 h=2" --overrides n_tiles_h=2 n_tiles_w=4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/008-w-4-h-3 --frontend \
     XVIO --name "XVIO tiles w=4 h=3" --overrides n_tiles_h=3 n_tiles_w=4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/009-w-4-h-4 --frontend \
     XVIO --name "XVIO tiles w=4 h=4" --overrides n_tiles_h=4 n_tiles_w=4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/010-w-5-h-3 --frontend \
     XVIO --name "XVIO tiles w=5 h=3" --overrides n_tiles_h=3 n_tiles_w=5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/011-w-5-h-4 --frontend \
     XVIO --name "XVIO tiles w=5 h=4" --overrides n_tiles_h=4 n_tiles_w=5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-tiling/012-w-5-h-5 --frontend \
     XVIO --name "XVIO tiles w=5 h=5" --overrides n_tiles_h=5 n_tiles_w=5

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-tiling/ --output_folder $1/$DATE-xvio-tiling/results

fi


if [ $EXPLORE_XVIO_N_POSES_MAX -gt 0 ]
then
  echo
  echo "Performing frame based XVIO n_poses_max exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-poses-max/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-poses-max/001-n-poses-5 --frontend \
     XVIO --name "XVIO n_poses=5" --overrides n_poses_max=5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-poses-max/002-n-poses-10 --frontend \
     XVIO --name "XVIO n_poses=10" --overrides n_poses_max=10

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-poses-max/003-n-poses-15 --frontend \
     XVIO --name "XVIO n_poses=15" --overrides n_poses_max=15

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-poses-max/004-n-poses-20 --frontend \
     XVIO --name "XVIO n_poses=20" --overrides n_poses_max=20

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-poses-max/005-n-poses-25 --frontend \
     XVIO --name "XVIO n_poses=25" --overrides n_poses_max=25

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-n-poses-max/ --output_folder $1/$DATE-xvio-n-poses-max/results

fi


if [ $EXPLORE_XVIO_N_SLAM_FEATURES_MAX -gt 0 ]
then
  echo
  echo "Performing frame based XVIO n_slam_features_max exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-slam-features-max/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-slam-features-max/001-n-slam-10 --frontend \
     XVIO --name "XVIO n_slam=10" --overrides n_slam_features_max=10

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-slam-features-max/002-n-slam-15 --frontend \
     XVIO --name "XVIO n_slam=15" --overrides n_slam_features_max=15

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-slam-features-max/003-n-slam-20 --frontend \
     XVIO --name "XVIO n_slam=20" --overrides n_slam_features_max=20

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-slam-features-max/004-n-slam-25 --frontend \
     XVIO --name "XVIO n_slam=25" --overrides n_slam_features_max=25

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-slam-features-max/005-n-slam-30 --frontend \
     XVIO --name "XVIO n_slam=30" --overrides n_slam_features_max=30

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-slam-features-max/006-n-slam-35 --frontend \
     XVIO --name "XVIO n_slam=35" --overrides n_slam_features_max=35

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-slam-features-max/007-n-slam-40 --frontend \
     XVIO --name "XVIO n_slam=40" --overrides n_slam_features_max=40

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-n-slam-features-max/008-n-slam-45 --frontend \
     XVIO --name "XVIO n_slam=45" --overrides n_slam_features_max=45

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-n-slam-features-max/ --output_folder
  $1/$DATE-xvio-n-slam-features-max/results

fi


if [ $EXPLORE_XVIO_SIGMA_IMG -gt 0 ]
then
  echo
  echo "Performing frame based XVIO sigma_img exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-sigma-img/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-sigma-img/001-sigma-1-f --frontend \
     XVIO --name "XVIO sigma_img=1/f" --overrides sigma_img=0.004850286885389534

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-sigma-img/002-sigma-2.5-f --frontend \
     XVIO --name "XVIO sigma_img=2.5/f" --overrides sigma_img=0.012125717213473835

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-sigma-img/003-sigma-4-f --frontend \
     XVIO --name "XVIO sigma_img=4/f" --overrides sigma_img=0.019401147541558136

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-sigma-img/004-sigma-5-f --frontend \
     XVIO --name "XVIO sigma_img=5/f" --overrides sigma_img=0.02425143442694767
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-sigma-img/005-sigma-6-f --frontend \
     XVIO --name "XVIO sigma_img=6/f" --overrides sigma_img=0.029101721312337205
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-sigma-img/006-sigma-7.5-f --frontend \
     XVIO --name "XVIO sigma_img=7.5/f" --overrides sigma_img=0.036377151640421504
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-sigma-img/007-sigma-8.86-f --frontend \
     XVIO --name "XVIO sigma_img=8.86/f" --overrides sigma_img=0.04297354180455127

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-sigma-img/008-sigma-10-f --frontend \
     XVIO --name "XVIO sigma_img=10/f" --overrides sigma_img=0.04850286885389534

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-sigma-img/ --output_folder $1/$DATE-xvio-sigma-img/results

fi
 

if [ $EXPLORE_XVIO_IMU_NOISE -gt 0 ]
then
  echo
  echo "Performing frame based XVIO IMU noise exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/001-imu-noise-a-opt-w-opt --frontend \
     XVIO --name "XVIO IMU noise a opt, w opt" --overrides n_a=0.004316 n_ba=0.0004316 n_w=0.00013 n_bw=0.000013

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/002-imu-noise-a-0.003-w-opt --frontend \
     XVIO --name "XVIO IMU noise a 0.003, w opt" --overrides n_a=0.003 n_ba=0.0003 n_w=0.00013 n_bw=0.000013

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/003-imu-noise-a-0.004-w-opt --frontend \
     XVIO --name "XVIO IMU noise a 0.004, w opt" --overrides n_a=0.004 n_ba=0.0004 n_w=0.00013 n_bw=0.000013

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/004-imu-noise-a-0.005-w-opt --frontend \
     XVIO --name "XVIO IMU noise a 0.005, w opt" --overrides n_a=0.005 n_ba=0.0005 n_w=0.00013 n_bw=0.000013

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/005-imu-noise-a-opt-w-0.0001 --frontend \
     XVIO --name "XVIO IMU noise a opt, w 0.0001" --overrides n_a=0.004316 n_ba=0.0004316 n_w=0.0001 n_bw=0.00001

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/006-imu-noise-a-opt-w-0.0002 --frontend \
     XVIO --name "XVIO IMU noise a opt, w 0.0002" --overrides n_a=0.004316 n_ba=0.0004316 n_w=0.0002 n_bw=0.00002

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/007-imu-noise-a-opt-w-0.0003 --frontend \
     XVIO --name "XVIO IMU noise a opt, w 0.0003" --overrides n_a=0.004316 n_ba=0.0004316 n_w=0.0003 n_bw=0.00003

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/008-imu-noise-a-0.004-w-0.0002 --frontend \
     XVIO --name "XVIO IMU noise a 0.004, w 0.0002" --overrides n_a=0.004 n_ba=0.0004 n_w=0.0002 n_bw=0.00002

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/009-imu-noise-a-0.0035-w-0.0002 --frontend \
     XVIO --name "XVIO IMU noise a 0.0035, w 0.0002" --overrides n_a=0.0035 n_ba=0.00035 n_w=0.0002 n_bw=0.00002

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/010-imu-noise-a-0.00375-w-0.0002 --frontend \
     XVIO --name "XVIO IMU noise a 0.00375, w 0.0002" --overrides n_a=0.00375 n_ba=0.000375 n_w=0.0002 n_bw=0.00002

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/011-imu-noise-a-0.00425-w-0.0002 --frontend \
     XVIO --name "XVIO IMU noise a 0.00425, w 0.0002" --overrides n_a=0.00425 n_ba=0.000425 n_w=0.0002 n_bw=0.00002

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/012-imu-noise-a-0.0045-w-0.0002 --frontend \
     XVIO --name "XVIO IMU noise a 0.0045, w 0.0002" --overrides n_a=0.0045 n_ba=0.00045 n_w=0.0002 n_bw=0.00002

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/013-imu-noise-a-0.004-w-0.00015 --frontend \
     XVIO --name "XVIO IMU noise a 0.004, w 0.00015" --overrides n_a=0.004 n_ba=0.0004 n_w=0.00015 n_bw=0.000015

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/014-imu-noise-a-0.004-w-0.000175 --frontend \
     XVIO --name "XVIO IMU noise a 0.004, w 0.000175" --overrides n_a=0.004 n_ba=0.0004 n_w=0.000175 n_bw=0.0000175

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/015-imu-noise-a-0.004-w-0.000225 --frontend \
     XVIO --name "XVIO IMU noise a 0.004, w 0.000225" --overrides n_a=0.004 n_ba=0.0004 n_w=0.000225 n_bw=0.0000225
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/016-imu-noise-a-0.004-w-0.00025 --frontend \
     XVIO --name "XVIO IMU noise a 0.004, w 0.00025" --overrides n_a=0.004 n_ba=0.0004 n_w=0.00025 n_bw=0.000025
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/016-imu-noise-a-0.004-w-0.00025 --frontend \
     XVIO --name "XVIO IMU noise a 0.00425, w 0.00025" --overrides n_a=0.00425 n_ba=0.000425 n_w=0.00025 n_bw=0.000025

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/001-imu-noise-a-0.01 --frontend \
     XVIO --name "XVIO IMU noise a 0.01" --overrides n_a=8.3e-05 n_ba=8.3e-06

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/002-imu-noise-a-0.02 --frontend \
     XVIO --name "XVIO IMU noise a 0.02" --overrides n_a=0.000166 n_ba=1.66e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/003-imu-noise-a-0.03 --frontend \
     XVIO --name "XVIO IMU noise a 0.03" --overrides n_a=0.000249 n_ba=2.49e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/004-imu-noise-a-0.05 --frontend \
     XVIO --name "XVIO IMU noise a 0.05" --overrides n_a=0.000415 n_ba=4.1500000000000006e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/005-imu-noise-a-0.1 --frontend \
     XVIO --name "XVIO IMU noise a 0.1" --overrides n_a=0.00083 n_ba=8.300000000000001e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/006-imu-noise-a-0.17 --frontend \
     XVIO --name "XVIO IMU noise a 0.17" --overrides n_a=0.0014110000000000001 n_ba=0.00014110000000000001

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/007-imu-noise-a-0.3 --frontend \
     XVIO --name "XVIO IMU noise a 0.3" --overrides n_a=0.00249 n_ba=0.000249

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/008-imu-noise-a-0.52 --frontend \
     XVIO --name "XVIO IMU noise a 0.52" --overrides n_a=0.004316 n_ba=0.00043160000000000003

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/009-imu-noise-a-0.92 --frontend \
     XVIO --name "XVIO IMU noise a 0.92" --overrides n_a=0.007636 n_ba=0.0007636

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/010-imu-noise-a-1.62 --frontend \
     XVIO --name "XVIO IMU noise a 1.62" --overrides n_a=0.013446000000000001 n_ba=0.0013446

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/011-imu-noise-a-2.84 --frontend \
     XVIO --name "XVIO IMU noise a 2.84" --overrides n_a=0.023572 n_ba=0.0023572

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/012-imu-noise-a-5.0 --frontend \
     XVIO --name "XVIO IMU noise a 5.0" --overrides n_a=0.0415 n_ba=0.00415

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/013-imu-noise-w-0.01 --frontend \
     XVIO --name "XVIO IMU noise w 0.01" --overrides n_w=1.3e-05 n_bw=1.2999999999999998e-06

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/014-imu-noise-w-0.02 --frontend \
     XVIO --name "XVIO IMU noise w 0.02" --overrides n_w=2.6e-05 n_bw=2.5999999999999997e-06

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/015-imu-noise-w-0.03 --frontend \
     XVIO --name "XVIO IMU noise w 0.03" --overrides n_w=3.9e-05 n_bw=3.9e-06

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/016-imu-noise-w-0.05 --frontend \
     XVIO --name "XVIO IMU noise w 0.05" --overrides n_w=6.5e-05 n_bw=6.5e-06

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/017-imu-noise-w-0.1 --frontend \
     XVIO --name "XVIO IMU noise w 0.1" --overrides n_w=0.00013 n_bw=1.3e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/018-imu-noise-w-0.17 --frontend \
     XVIO --name "XVIO IMU noise w 0.17" --overrides n_w=0.000221 n_bw=2.21e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/019-imu-noise-w-0.3 --frontend \
     XVIO --name "XVIO IMU noise w 0.3" --overrides n_w=0.00039 n_bw=3.899999999999999e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/020-imu-noise-w-0.52 --frontend \
     XVIO --name "XVIO IMU noise w 0.52" --overrides n_w=0.000676 n_bw=6.759999999999999e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/021-imu-noise-w-0.92 --frontend \
     XVIO --name "XVIO IMU noise w 0.92" --overrides n_w=0.001196 n_bw=0.0001196

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/022-imu-noise-w-1.62 --frontend \
     XVIO --name "XVIO IMU noise w 1.62" --overrides n_w=0.0021060000000000002 n_bw=0.0002106

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/023-imu-noise-w-2.84 --frontend \
     XVIO --name "XVIO IMU noise w 2.84" --overrides n_w=0.0036919999999999995 n_bw=0.0003692

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/024-imu-noise-w-5.0 --frontend \
     XVIO --name "XVIO IMU noise w 5.0" --overrides n_w=0.0065 n_bw=0.00065

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/025-imu-noise-a-w-0.01 --frontend \
     XVIO --name "XVIO IMU noise a, w 0.01" --overrides n_a=8.3e-05 n_ba=8.3e-06 n_w=1.3e-05 n_bw=1.2999999999999998e-06

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/026-imu-noise-a-w-0.02 --frontend \
     XVIO --name "XVIO IMU noise a, w 0.02" --overrides n_a=0.000166 n_ba=1.66e-05 n_w=2.6e-05 n_bw=2.5999999999999997e-06

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/027-imu-noise-a-w-0.03 --frontend \
     XVIO --name "XVIO IMU noise a, w 0.03" --overrides n_a=0.000249 n_ba=2.49e-05 n_w=3.9e-05 n_bw=3.9e-06

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/028-imu-noise-a-w-0.05 --frontend \
     XVIO --name "XVIO IMU noise a, w 0.05" --overrides n_a=0.000415 n_ba=4.1500000000000006e-05 n_w=6.5e-05 n_bw=6.5e-06

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/029-imu-noise-a-w-0.1 --frontend \
     XVIO --name "XVIO IMU noise a, w 0.1" --overrides n_a=0.00083 n_ba=8.300000000000001e-05 n_w=0.00013 n_bw=1.3e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/030-imu-noise-a-w-0.17 --frontend \
     XVIO --name "XVIO IMU noise a, w 0.17" --overrides n_a=0.0014110000000000001 n_ba=0.00014110000000000001 n_w=0.000221 n_bw=2.21e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/031-imu-noise-a-w-0.3 --frontend \
     XVIO --name "XVIO IMU noise a, w 0.3" --overrides n_a=0.00249 n_ba=0.000249 n_w=0.00039 n_bw=3.899999999999999e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/032-imu-noise-a-w-0.52 --frontend \
     XVIO --name "XVIO IMU noise a, w 0.52" --overrides n_a=0.004316 n_ba=0.00043160000000000003 n_w=0.000676 n_bw=6.759999999999999e-05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/033-imu-noise-a-w-0.92 --frontend \
     XVIO --name "XVIO IMU noise a, w 0.92" --overrides n_a=0.007636 n_ba=0.0007636 n_w=0.001196 n_bw=0.0001196

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/034-imu-noise-a-w-1.62 --frontend \
     XVIO --name "XVIO IMU noise a, w 1.62" --overrides n_a=0.013446000000000001 n_ba=0.0013446 n_w=0.0021060000000000002 n_bw=0.0002106

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/035-imu-noise-a-w-2.84 --frontend \
     XVIO --name "XVIO IMU noise a, w 2.84" --overrides n_a=0.023572 n_ba=0.0023572 n_w=0.0036919999999999995 n_bw=0.0003692

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-imu-noise/036-imu-noise-a-w-5.0 --frontend \
     XVIO --name "XVIO IMU noise a, w 5.0" --overrides n_a=0.0415 n_ba=0.00415 n_w=0.0065 n_bw=0.00065

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-imu-noise/ --output_folder $1/$DATE-xvio-imu-noise/results

fi



if [ $EXPLORE_XVIO_ACC_SPIKE -gt 0 ]
then
  echo
  echo "Performing frame based XVIO accelerometer spike exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-acc-spike/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-acc-spike/001-imu-acc-spike-30 --frontend \
     XVIO --name "XVIO IMU spike 30" --overrides a_m_max=30

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-acc-spike/002-imu-acc-spike-40 --frontend \
     XVIO --name "XVIO IMU spike 40" --overrides a_m_max=40

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-acc-spike/003-imu-acc-spike-50 --frontend \
     XVIO --name "XVIO IMU spike 50" --overrides a_m_max=50

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-acc-spike/004-imu-acc-spike-60 --frontend \
     XVIO --name "XVIO IMU spike 60" --overrides a_m_max=60

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-acc-spike/005-imu-acc-spike-70 --frontend \
     XVIO --name "XVIO IMU spike 70" --overrides a_m_max=70

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-acc-spike/006-imu-acc-spike-80 --frontend \
     XVIO --name "XVIO IMU spike 80" --overrides a_m_max=80

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-acc-spike/ --output_folder $1/$DATE-xvio-acc-spike/results

fi


if [ $EXPLORE_XVIO_INITIAL_SIGMA_P -gt 0 ]
then
  echo
  echo "Performing frame based XVIO initial standard deviation sigma_dp exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

# sigma_dp: [0.0, 0.0, 0.0] # [m]


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/001-sigma-dp-0.0 --frontend \
     XVIO --name "XVIO sigma_dp 0.0" --overrides sigma_dp=[0.0,0.0,0.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/002-sigma-dp-0.001 --frontend \
     XVIO --name "XVIO sigma_dp 0.001" --overrides sigma_dp=[0.001,0.001,0.001]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/003-sigma-dp-0.01 --frontend \
     XVIO --name "XVIO sigma_dp 0.01" --overrides sigma_dp=[0.01,0.01,0.01]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/004-sigma-dp-0.1 --frontend \
     XVIO --name "XVIO sigma_dp 0.1" --overrides sigma_dp=[0.1,0.1,0.1]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/005-sigma-dp-0.2 --frontend \
     XVIO --name "XVIO sigma_dp 0.2" --overrides sigma_dp=[0.2,0.2,0.2]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/006-sigma-dp-0.5 --frontend \
     XVIO --name "XVIO sigma_dp 0.5" --overrides sigma_dp=[0.5,0.5,0.5]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/007-sigma-dp-0.7 --frontend \
     XVIO --name "XVIO sigma_dp 0.7" --overrides sigma_dp=[0.7,0.7,0.7]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/008-sigma-dp-1.0 --frontend \
     XVIO --name "XVIO sigma_dp 1.0" --overrides sigma_dp=[1.0,1.0,1.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/009-sigma-dp-1.5 --frontend \
     XVIO --name "XVIO sigma_dp 1.5" --overrides sigma_dp=[1.5,1.5,1.5]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dp/010-sigma-dp-2 --frontend \
     XVIO --name "XVIO sigma_dp 2" --overrides sigma_dp=[2,2,2]

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-initial-stds-dp/ --output_folder $1/$DATE-xvio-initial-stds-dp/results

fi



if [ $EXPLORE_XVIO_INITIAL_SIGMA_V -gt 0 ]
then
  echo
  echo "Performing frame based XVIO initial standard deviation sigma_dv exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

# sigma_dv: [0.05, 0.05, 0.05] # [m/s]


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/011-sigma-dv-0.0 --frontend \
     XVIO --name "XVIO sigma_dv 0.0" --overrides sigma_dv=[0.0,0.0,0.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/012-sigma-dv-0.0001 --frontend \
     XVIO --name "XVIO sigma_dv 0.0001" --overrides sigma_dv=[0.0001,0.0001,0.0001]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/013-sigma-dv-0.001 --frontend \
     XVIO --name "XVIO sigma_dv 0.001" --overrides sigma_dv=[0.001,0.001,0.001]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/014-sigma-dv-0.01 --frontend \
     XVIO --name "XVIO sigma_dv 0.01" --overrides sigma_dv=[0.01,0.01,0.01]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/015-sigma-dv-0.03 --frontend \
     XVIO --name "XVIO sigma_dv 0.03" --overrides sigma_dv=[0.03,0.03,0.03]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/016-sigma-dv-0.05 --frontend \
     XVIO --name "XVIO sigma_dv 0.05" --overrides sigma_dv=[0.05,0.05,0.05]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/017-sigma-dv-0.1 --frontend \
     XVIO --name "XVIO sigma_dv 0.1" --overrides sigma_dv=[0.1,0.1,0.1]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/018-sigma-dv-0.2 --frontend \
     XVIO --name "XVIO sigma_dv 0.2" --overrides sigma_dv=[0.2,0.2,0.2]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/019-sigma-dv-0.5 --frontend \
     XVIO --name "XVIO sigma_dv 0.5" --overrides sigma_dv=[0.5,0.5,0.5]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dv/020-sigma-dv-1.0 --frontend \
     XVIO --name "XVIO sigma_dv 1.0" --overrides sigma_dv=[1.0,1.0,1.0]

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-initial-stds-dv/ --output_folder $1/$DATE-xvio-initial-stds-dv/results

fi



if [ $EXPLORE_XVIO_INITIAL_SIGMA_THETA -gt 0 ]
then
  echo
  echo "Performing frame based XVIO initial standard deviation sigma_dtheta exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

# sigma_dtheta: [3.0, 3.0, 3.0] # [deg]


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/021-sigma-dtheta-0.0 --frontend \
     XVIO --name "XVIO sigma_dtheta 0.0" --overrides sigma_dtheta=[0.0,0.0,0.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/022-sigma-dtheta-0.01 --frontend \
     XVIO --name "XVIO sigma_dtheta 0.01" --overrides sigma_dtheta=[0.01,0.01,0.01]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/023-sigma-dtheta-0.1 --frontend \
     XVIO --name "XVIO sigma_dtheta 0.1" --overrides sigma_dtheta=[0.1,0.1,0.1]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/024-sigma-dtheta-0.3 --frontend \
     XVIO --name "XVIO sigma_dtheta 0.3" --overrides sigma_dtheta=[0.3,0.3,0.3]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/025-sigma-dtheta-0.8 --frontend \
     XVIO --name "XVIO sigma_dtheta 0.8" --overrides sigma_dtheta=[0.8,0.8,0.8]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/026-sigma-dtheta-1.5 --frontend \
     XVIO --name "XVIO sigma_dtheta 1.5" --overrides sigma_dtheta=[1.5,1.5,1.5]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/027-sigma-dtheta-3.0 --frontend \
     XVIO --name "XVIO sigma_dtheta 3.0" --overrides sigma_dtheta=[3.0,3.0,3.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/028-sigma-dtheta-6.0 --frontend \
     XVIO --name "XVIO sigma_dtheta 6.0" --overrides sigma_dtheta=[6.0,6.0,6.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/029-sigma-dtheta-15.0 --frontend \
     XVIO --name "XVIO sigma_dtheta 15.0" --overrides sigma_dtheta=[15.0,15.0,15.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dtheta/030-sigma-dtheta-30.0 --frontend \
     XVIO --name "XVIO sigma_dtheta 30.0" --overrides sigma_dtheta=[30.0,30.0,30.0]

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-initial-stds-dtheta/ --output_folder
  $1/$DATE-xvio-initial-stds-dtheta/results

fi



if [ $EXPLORE_XVIO_INITIAL_SIGMA_BW -gt 0 ]
then
  echo
  echo "Performing frame based XVIO initial standard deviation sigma_dbw exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

# sigma_dbw: [6.0, 6.0, 6.0] # [deg/s]


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dbw/031-sigma-dbw-0.0 --frontend \
     XVIO --name "XVIO sigma_dbw 0.0" --overrides sigma_dbw=[0.0,0.0,0.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dbw/032-sigma-dbw-0.01 --frontend \
     XVIO --name "XVIO sigma_dbw 0.01" --overrides sigma_dbw=[0.01,0.01,0.01]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dbw/033-sigma-dbw-0.1 --frontend \
     XVIO --name "XVIO sigma_dbw 0.1" --overrides sigma_dbw=[0.1,0.1,0.1]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dbw/034-sigma-dbw-0.5 --frontend \
     XVIO --name "XVIO sigma_dbw 0.5" --overrides sigma_dbw=[0.5,0.5,0.5]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dbw/035-sigma-dbw-1.8 --frontend \
     XVIO --name "XVIO sigma_dbw 1.8" --overrides sigma_dbw=[1.8,1.8,1.8]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dbw/036-sigma-dbw-4.5 --frontend \
     XVIO --name "XVIO sigma_dbw 4.5" --overrides sigma_dbw=[4.5,4.5,4.5]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dbw/037-sigma-dbw-8.0 --frontend \
     XVIO --name "XVIO sigma_dbw 8.0" --overrides sigma_dbw=[8.0,8.0,8.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dbw/038-sigma-dbw-12.0 --frontend \
     XVIO --name "XVIO sigma_dbw 12.0" --overrides sigma_dbw=[12.0,12.0,12.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dbw/039-sigma-dbw-30.0 --frontend \
     XVIO --name "XVIO sigma_dbw 30.0" --overrides sigma_dbw=[30.0,30.0,30.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dbw/040-sigma-dbw-45.0 --frontend \
     XVIO --name "XVIO sigma_dbw 45.0" --overrides sigma_dbw=[45.0,45.0,45.0]

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-initial-stds-dbw/ --output_folder $1/$DATE-xvio-initial-stds-dbw/results

fi



if [ $EXPLORE_XVIO_INITIAL_SIGMA_BA -gt 0 ]
then
  echo
  echo "Performing frame based XVIO initial standard deviation sigma_dba exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

# sigma_dba: [0.3, 0.3, 0.3] # [m/s^2]


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/041-sigma-dba-0.0 --frontend \
     XVIO --name "XVIO sigma_dba 0.0" --overrides sigma_dba=[0.0,0.0,0.0]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/042-sigma-dba-0.0001 --frontend \
     XVIO --name "XVIO sigma_dba 0.0001" --overrides sigma_dba=[0.0001,0.0001,0.0001]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/043-sigma-dba-0.001 --frontend \
     XVIO --name "XVIO sigma_dba 0.001" --overrides sigma_dba=[0.001,0.001,0.001]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/044-sigma-dba-0.01 --frontend \
     XVIO --name "XVIO sigma_dba 0.01" --overrides sigma_dba=[0.01,0.01,0.01]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/045-sigma-dba-0.05 --frontend \
     XVIO --name "XVIO sigma_dba 0.05" --overrides sigma_dba=[0.05,0.05,0.05]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/046-sigma-dba-0.2 --frontend \
     XVIO --name "XVIO sigma_dba 0.2" --overrides sigma_dba=[0.2,0.2,0.2]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/047-sigma-dba-0.45 --frontend \
     XVIO --name "XVIO sigma_dba 0.45" --overrides sigma_dba=[0.45,0.45,0.45]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/048-sigma-dba-0.6 --frontend \
     XVIO --name "XVIO sigma_dba 0.6" --overrides sigma_dba=[0.6,0.6,0.6]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/049-sigma-dba-1.2 --frontend \
     XVIO --name "XVIO sigma_dba 1.2" --overrides sigma_dba=[1.2,1.2,1.2]

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-xvio-initial-stds-dba/050-sigma-dba-3.0 --frontend \
     XVIO --name "XVIO sigma_dba 3.0" --overrides sigma_dba=[3.0,3.0,3.0]

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-xvio-initial-stds-dba/ --output_folder $1/$DATE-xvio-initial-stds-dba/results

fi


if [ $EXPLORE_EKLT_PATCH_SIZE -gt 0 ]
then
  echo
  echo "Performing EKLT patch size exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-size/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-patch-size/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-size/001-p-15 \
      --frontend EKLT --name "EKLT p=15" --overrides eklt_patch_size=15 eklt_detection_min_distance=7

    cleanup $1/$DATE-eklt-patch-size/001-p-15

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-size/002-p-17 \
    #   --frontend EKLT --name "EKLT p=17" --overrides eklt_patch_size=17 eklt_detection_min_distance=8

    # cleanup $1/$DATE-eklt-patch-size/002-p-17

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-size/003-p-19 \
      --frontend EKLT --name "EKLT p=19" --overrides eklt_patch_size=19 eklt_detection_min_distance=9

    cleanup $1/$DATE-eklt-patch-size/003-p-19

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-size/004-p-21 \
      --frontend EKLT --name "EKLT p=21" --overrides eklt_patch_size=21 eklt_detection_min_distance=10

    cleanup $1/$DATE-eklt-patch-size/004-p-21

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-size/005-p-23 \
    #   --frontend EKLT --name "EKLT p=23" --overrides eklt_patch_size=21 eklt_detection_min_distance=11

    # cleanup $1/$DATE-eklt-patch-size/005-p-23

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-size/006-p-25 \
      --frontend EKLT --name "EKLT p=25" --overrides eklt_patch_size=25 eklt_detection_min_distance=12

    cleanup $1/$DATE-eklt-patch-size/006-p-25

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-size/007-p-27 \
    #   --frontend EKLT --name "EKLT p=27" --overrides eklt_patch_size=27 eklt_detection_min_distance=13

    cleanup $1/$DATE-eklt-patch-size/007-p-27

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-size/008-p-31 \
      --frontend EKLT --name "EKLT p=31" --overrides eklt_patch_size=31 eklt_detection_min_distance=14

    cleanup $1/$DATE-eklt-patch-size/008-p-31

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-size/009-p-35 \
      --frontend EKLT --name "EKLT p=35" --overrides eklt_patch_size=35 eklt_detection_min_distance=14

    cleanup $1/$DATE-eklt-patch-size/009-p-35

    
  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-patch-size/ --output_folder $1/$DATE-eklt-patch-size/results

fi


if [ $EXPLORE_EKLT_IMU_OFFSET -gt 0 ]
then
  echo
  echo "Performing EKLT IMU offset exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-imu-offset/000-xvio-baseline

    # 0.0027626

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/001-t-0.005 \
      --frontend EKLT --name "EKLT IMU offset -0.005" --overrides cam1_time_offset=-0.005

    cleanup $1/$DATE-eklt-imu-offset/001-t-0.005

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/002-t-0.003 \
      --frontend EKLT --name "EKLT IMU offset -0.003" --overrides cam1_time_offset=-0.003

    cleanup $1/$DATE-eklt-imu-offset/002-t-0.003


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/003-t-0.001 \
      --frontend EKLT --name "EKLT IMU offset -0.001" --overrides cam1_time_offset=-0.001

    cleanup $1/$DATE-eklt-imu-offset/003-t-0.001


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/004-t-0.0005 \
      --frontend EKLT --name "EKLT IMU offset -0.0005" --overrides cam1_time_offset=-0.0005

    cleanup $1/$DATE-eklt-imu-offset/004-t-0.0005


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/005-t0.0005 \
      --frontend EKLT --name "EKLT IMU offset 0.0005" --overrides cam1_time_offset=0.0005

    cleanup $1/$DATE-eklt-imu-offset/005-t0.0005

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/006-t0.001 \
      --frontend EKLT --name "EKLT IMU offset 0.001" --overrides cam1_time_offset=0.001

    cleanup $1/$DATE-eklt-imu-offset/006-t0.001

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/007-t0.0025 \
      --frontend EKLT --name "EKLT IMU offset 0.0025" --overrides cam1_time_offset=0.0025

    cleanup $1/$DATE-eklt-imu-offset/007-t0.0025

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/008-t0.00275 \
      --frontend EKLT --name "EKLT IMU offset 0.00275" --overrides cam1_time_offset=0.00275

    cleanup $1/$DATE-eklt-imu-offset/008-t0.00275

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/009-t0.0028 \
      --frontend EKLT --name "EKLT IMU offset 0.0028" --overrides cam1_time_offset=0.0028

    cleanup $1/$DATE-eklt-imu-offset/009-t0.0028

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/010-t0.0029 \
      --frontend EKLT --name "EKLT IMU offset 0.0029" --overrides cam1_time_offset=0.0029

    cleanup $1/$DATE-eklt-imu-offset/010-t0.0029

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/011-t0.003 \
      --frontend EKLT --name "EKLT IMU offset 0.003" --overrides cam1_time_offset=0.003

    cleanup $1/$DATE-eklt-imu-offset/011-t0.003

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-imu-offset/012-t0.005 \
      --frontend EKLT --name "EKLT IMU offset 0.005" --overrides cam1_time_offset=0.005

    cleanup $1/$DATE-eklt-imu-offset/012-t0.005

    
  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-imu-offset/ --output_folder $1/$DATE-eklt-imu-offset/results

fi


if [ $EXPLORE_EKLT_OUTLIER_REMOVAL -gt 0 ]
then
  echo
  echo "Performing EKLT outlier removal exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-outlier-removal/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/001-eklt-remove-outliers-on \
      --frontend EKLT --name "EKLT outlier removal ON" --overrides eklt_enable_outlier_removal=true

    cleanup $1/$DATE-eklt-outlier-removal/001-eklt-remove-outliers-on

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/002-eklt-remove-outliers-off \
      --frontend EKLT --name "EKLT outlier removal OFF" --overrides eklt_enable_outlier_removal=false

    cleanup $1/$DATE-eklt-outlier-removal/002-eklt-remove-outliers-off


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/003-eklt-outliers-px1.3-p0.8 \
      --frontend EKLT --name "EKLT outlier px=1.3 p=0.8" --overrides eklt_enable_outlier_removal=true outlier_param1=1.3 outlier_param2=0.8

    cleanup $1/$DATE-eklt-outlier-removal/003-eklt-outliers-px1.3-p0.8


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/004-eklt-outliers-px1.3-p0.9 \
      --frontend EKLT --name "EKLT outlier px=1.3 p=0.9" --overrides eklt_enable_outlier_removal=true outlier_param1=1.3 outlier_param2=0.9

    cleanup $1/$DATE-eklt-outlier-removal/004-eklt-outliers-px1.3-p0.9


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/005-eklt-outliers-px1.3-p0.95 \
      --frontend EKLT --name "EKLT outlier px=1.3 p=0.95" --overrides eklt_enable_outlier_removal=true outlier_param1=1.3 outlier_param2=0.95

    cleanup $1/$DATE-eklt-outlier-removal/005-eklt-outliers-px1.3-p0.95


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/006-eklt-outliers-px1.3-p0.99 \
      --frontend EKLT --name "EKLT outlier px=1.3 p=0.99" --overrides eklt_enable_outlier_removal=true outlier_param1=1.3 outlier_param2=0.99

    cleanup $1/$DATE-eklt-outlier-removal/006-eklt-outliers-px1.3-p0.99


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/007-eklt-outliers-px1.6-p0.8 \
      --frontend EKLT --name "EKLT outlier px=1.6 p=0.8" --overrides eklt_enable_outlier_removal=true outlier_param1=1.6 outlier_param2=0.8

    cleanup $1/$DATE-eklt-outlier-removal/007-eklt-outliers-px1.6-p0.8


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/008-eklt-outliers-px1.6-p0.9 \
      --frontend EKLT --name "EKLT outlier px=1.6 p=0.9" --overrides eklt_enable_outlier_removal=true outlier_param1=1.6 outlier_param2=0.9

    cleanup $1/$DATE-eklt-outlier-removal/008-eklt-outliers-px1.6-p0.9


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/009-eklt-outliers-px1.6-p0.95 \
      --frontend EKLT --name "EKLT outlier px=1.6 p=0.95" --overrides eklt_enable_outlier_removal=true outlier_param1=1.6 outlier_param2=0.95

    cleanup $1/$DATE-eklt-outlier-removal/009-eklt-outliers-px1.6-p0.95


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-outlier-removal/010-eklt-outliers-px1.6-p0.99 \
      --frontend EKLT --name "EKLT outlier px=1.6 p=0.99" --overrides eklt_enable_outlier_removal=true outlier_param1=1.6 outlier_param2=0.99

    cleanup $1/$DATE-eklt-outlier-removal/010-eklt-outliers-px1.6-p0.99


  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-outlier-removal/ --output_folder $1/$DATE-eklt-outlier-removal/results

fi

if [ $EXPLORE_EKLT_TRACKING_QUALITY -gt 0 ]
then
  echo
  echo "Performing EKLT tracking quality exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-tracking-quality/000-xvio-baseline


     # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/001-eklt-tracking-q-0.0001 \
     #   --frontend EKLT --name "EKLT tracking-q=0.0001" --overrides eklt_tracking_quality=0.0001

     # cleanup $1/$DATE-eklt-tracking-quality/001-eklt-tracking-q-0.0001

     # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/002-eklt-tracking-q-0.001 \
     #   --frontend EKLT --name "EKLT tracking-q=0.001" --overrides eklt_tracking_quality=0.001

     # cleanup $1/$DATE-eklt-tracking-quality/002-eklt-tracking-q-0.001

     # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/003-eklt-tracking-q-0.005 \
     #   --frontend EKLT --name "EKLT tracking-q=0.005" --overrides eklt_tracking_quality=0.005

     # cleanup $1/$DATE-eklt-tracking-quality/003-eklt-tracking-q-0.005

     # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/004-eklt-tracking-q-0.01 \
     #   --frontend EKLT --name "EKLT tracking-q=0.01" --overrides eklt_tracking_quality=0.01

     # cleanup $1/$DATE-eklt-tracking-quality/004-eklt-tracking-q-0.01

     # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/005-eklt-tracking-q-0.1 \
     #   --frontend EKLT --name "EKLT tracking-q=0.1" --overrides eklt_tracking_quality=0.1

     # cleanup $1/$DATE-eklt-tracking-quality/005-eklt-tracking-q-0.1

     # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/006-eklt-tracking-q-0.15 \
     #   --frontend EKLT --name "EKLT tracking-q=0.15" --overrides eklt_tracking_quality=0.15

     # cleanup $1/$DATE-eklt-tracking-quality/006-eklt-tracking-q-0.15

     # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/007-eklt-tracking-q-0.2 \
     #   --frontend EKLT --name "EKLT tracking-q=0.2" --overrides eklt_tracking_quality=0.2

     # cleanup $1/$DATE-eklt-tracking-quality/007-eklt-tracking-q-0.2

     # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/008-eklt-tracking-q-0.25 \
     #   --frontend EKLT --name "EKLT tracking-q=0.25" --overrides eklt_tracking_quality=0.25

     # cleanup $1/$DATE-eklt-tracking-quality/008-eklt-tracking-q-0.25

     # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/009-eklt-tracking-q-0.3 \
     #   --frontend EKLT --name "EKLT tracking-q=0.3" --overrides eklt_tracking_quality=0.3

     # cleanup $1/$DATE-eklt-tracking-quality/009-eklt-tracking-q-0.3

      python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/010-eklt-tracking-q-0.4 \
        --frontend EKLT --name "EKLT tracking-q=0.4" --overrides eklt_tracking_quality=0.4

      cleanup $1/$DATE-eklt-tracking-quality/010-eklt-tracking-q-0.4

      python evaluate.py --configuration $CONFIGURATION --output_folder
      $1/$DATE-eklt-tracking-quality/010-eklt-tracking-q-0.45 \
        --frontend EKLT --name "EKLT tracking-q=0.45" --overrides eklt_tracking_quality=0.45

      cleanup $1/$DATE-eklt-tracking-quality/010-eklt-tracking-q-0.45

     python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/011-eklt-tracking-q-0.5 \
       --frontend EKLT --name "EKLT tracking-q=0.5" --overrides eklt_tracking_quality=0.5

     cleanup $1/$DATE-eklt-tracking-quality/011-eklt-tracking-q-0.5

     python evaluate.py --configuration $CONFIGURATION --output_folder
     $1/$DATE-eklt-tracking-quality/011-eklt-tracking-q-0.55 \
       --frontend EKLT --name "EKLT tracking-q=0.55" --overrides eklt_tracking_quality=0.55

     cleanup $1/$DATE-eklt-tracking-quality/011-eklt-tracking-q-0.55

     python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/012-eklt-tracking-q-0.6 \
       --frontend EKLT --name "EKLT tracking-q=0.6" --overrides eklt_tracking_quality=0.6

     cleanup $1/$DATE-eklt-tracking-quality/012-eklt-tracking-q-0.6

     python evaluate.py --configuration $CONFIGURATION --output_folder
     $1/$DATE-eklt-tracking-quality/012-eklt-tracking-q-0.65 \
       --frontend EKLT --name "EKLT tracking-q=0.65" --overrides eklt_tracking_quality=0.65

     cleanup $1/$DATE-eklt-tracking-quality/012-eklt-tracking-q-0.65

     python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/013-eklt-tracking-q-0.7 \
       --frontend EKLT --name "EKLT tracking-q=0.7" --overrides eklt_tracking_quality=0.7

     cleanup $1/$DATE-eklt-tracking-quality/013-eklt-tracking-q-0.7

#     python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/014-eklt-tracking-q-0.8 \
#       --frontend EKLT --name "EKLT tracking-q=0.8" --overrides eklt_tracking_quality=0.8
#
#     cleanup $1/$DATE-eklt-tracking-quality/014-eklt-tracking-q-0.8
#
#     python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/015-eklt-tracking-q-0.9 \
#       --frontend EKLT --name "EKLT tracking-q=0.9" --overrides eklt_tracking_quality=0.9
#
#     cleanup $1/$DATE-eklt-tracking-quality/015-eklt-tracking-q-0.9

#    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/001-eklt-tracking-q-0.16 \
#      --frontend EKLT --name "EKLT tracking-q=0.16" --overrides eklt_tracking_quality=0.16
#
#    cleanup $1/$DATE-eklt-tracking-quality/001-eklt-tracking-q-0.16
#
#    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/002-eklt-tracking-q-0.18 \
#      --frontend EKLT --name "EKLT tracking-q=0.18" --overrides eklt_tracking_quality=0.18
#
#    cleanup $1/$DATE-eklt-tracking-quality/002-eklt-tracking-q-0.18
#
#    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/003-eklt-tracking-q-0.19 \
#      --frontend EKLT --name "EKLT tracking-q=0.19" --overrides eklt_tracking_quality=0.19
#
#    cleanup $1/$DATE-eklt-tracking-quality/003-eklt-tracking-q-0.19
#
#    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/004-eklt-tracking-q-0.20 \
#      --frontend EKLT --name "EKLT tracking-q=0.20" --overrides eklt_tracking_quality=0.20
#
#    cleanup $1/$DATE-eklt-tracking-quality/004-eklt-tracking-q-0.20
#
#    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/005-eklt-tracking-q-0.21 \
#      --frontend EKLT --name "EKLT tracking-q=0.21" --overrides eklt_tracking_quality=0.21
#
#    cleanup $1/$DATE-eklt-tracking-quality/005-eklt-tracking-q-0.21
#
#    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/006-eklt-tracking-q-0.22 \
#      --frontend EKLT --name "EKLT tracking-q=0.22" --overrides eklt_tracking_quality=0.22
#
#    cleanup $1/$DATE-eklt-tracking-quality/006-eklt-tracking-q-0.22
#
#    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-tracking-quality/007-eklt-tracking-q-0.24 \
#      --frontend EKLT --name "EKLT tracking-q=0.24" --overrides eklt_tracking_quality=0.24
#
#    cleanup $1/$DATE-eklt-tracking-quality/007-eklt-tracking-q-0.24

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-tracking-quality/ --output_folder $1/$DATE-eklt-tracking-quality/results

fi


if [ $EXPLORE_EKLT_UPDATE_STRATEGY_N_MSEC -gt 0 ]
then
  echo
  echo "Performing EKLT UPDATE_STRATEGY_N_MSEC exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/000-xvio-baseline \
    #   --frontend XVIO --name "XVIO baseline"

    # cleanup $1/$DATE-eklt-update-strategy-msec/000-xvio-baseline

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/001-eklt-update-1-msec \
    #   --frontend EKLT --name "EKLT update every 1msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=1

    # cleanup $1/$DATE-eklt-update-strategy-msec/001-eklt-update-1-msec

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/002-eklt-update-2-msec \
    #   --frontend EKLT --name "EKLT update every 2msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=2

    # cleanup $1/$DATE-eklt-update-strategy-msec/002-eklt-update-2-msec

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/003-eklt-update-3-msec \
    #   --frontend EKLT --name "EKLT update every 3msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=3

    # cleanup $1/$DATE-eklt-update-strategy-msec/003-eklt-update-3-msec

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/004-eklt-update-5-msec \
    #   --frontend EKLT --name "EKLT update every 5msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=5

    # cleanup $1/$DATE-eklt-update-strategy-msec/004-eklt-update-5-msec

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/005-eklt-update-7-msec \
    #   --frontend EKLT --name "EKLT update every 7msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=7

    # cleanup $1/$DATE-eklt-update-strategy-msec/005-eklt-update-7-msec

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/006-eklt-update-9-msec \
    #   --frontend EKLT --name "EKLT update every 9msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=9

    # cleanup $1/$DATE-eklt-update-strategy-msec/006-eklt-update-9-msec

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/007-eklt-update-12-msec \
    #   --frontend EKLT --name "EKLT update every 12msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=12

    # cleanup $1/$DATE-eklt-update-strategy-msec/007-eklt-update-12-msec

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/008-eklt-update-40-msec \
    #   --frontend EKLT --name "EKLT update every 40msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=40

    # cleanup $1/$DATE-eklt-update-strategy-msec/008-eklt-update-40-msec





    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/001-eklt-update-10-msec \
      --frontend EKLT --name "EKLT update every 10msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-update-strategy-msec/001-eklt-update-10-msec

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/002-eklt-update-20-msec \
      --frontend EKLT --name "EKLT update every 20msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=20

    cleanup $1/$DATE-eklt-update-strategy-msec/002-eklt-update-20-msec

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/003-eklt-update-30-msec \
      --frontend EKLT --name "EKLT update every 30msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=30

    cleanup $1/$DATE-eklt-update-strategy-msec/003-eklt-update-30-msec

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-msec/004-eklt-update-40-msec \
      --frontend EKLT --name "EKLT update every 40msec" --overrides eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=40

    cleanup $1/$DATE-eklt-update-strategy-msec/004-eklt-update-40-msec




  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-update-strategy-msec/ --output_folder
  $1/$DATE-eklt-update-strategy-msec/results

fi


if [ $EXPLORE_EKLT_UPDATE_STRATEGY_N_EVENTS -gt 0 ]
then
  echo
  echo "Performing EKLT UPDATE_STRATEGY_N_EVENTS exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/000-xvio-baseline \
    #   --frontend XVIO --name "XVIO baseline"

    # cleanup $1/$DATE-eklt-update-strategy-n-events/000-xvio-baseline

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/001-eklt-update-500-events \
    #   --frontend EKLT --name "EKLT update every 500 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=500

    # cleanup $1/$DATE-eklt-update-strategy-n-events/001-eklt-update-500-events

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/002-eklt-update-1000-events \
    #   --frontend EKLT --name "EKLT update every 1000 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=1000

    # cleanup $1/$DATE-eklt-update-strategy-n-events/002-eklt-update-1000-events

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/003-eklt-update-2000-events \
    #   --frontend EKLT --name "EKLT update every 2000 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=2000

    # cleanup $1/$DATE-eklt-update-strategy-n-events/003-eklt-update-2000-events

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/004-eklt-update-3600-events \
    #   --frontend EKLT --name "EKLT update every 3600 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=3600

    # cleanup $1/$DATE-eklt-update-strategy-n-events/004-eklt-update-3600-events

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/005-eklt-update-4800-events \
    #   --frontend EKLT --name "EKLT update every 4800 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=4800

    # cleanup $1/$DATE-eklt-update-strategy-n-events/005-eklt-update-4800-events

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/006-eklt-update-7200-events \
    #   --frontend EKLT --name "EKLT update every 7200 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=7200

    # cleanup $1/$DATE-eklt-update-strategy-n-events/006-eklt-update-7200-events

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/007-eklt-update-9200-events \
    #   --frontend EKLT --name "EKLT update every 9200 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=9200

    # cleanup $1/$DATE-eklt-update-strategy-n-events/007-eklt-update-9200-events


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/001-eklt-update-5000-events \
      --frontend EKLT --name "EKLT update every 5000 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=5000

    cleanup $1/$DATE-eklt-update-strategy-n-events/001-eklt-update-5000-events

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/002-eklt-update-10000-events \
      --frontend EKLT --name "EKLT update every 10000 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=10000

    cleanup $1/$DATE-eklt-update-strategy-n-events/002-eklt-update-10000-events

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/003-eklt-update-15000-events \
      --frontend EKLT --name "EKLT update every 15000 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=15000

    cleanup $1/$DATE-eklt-update-strategy-n-events/003-eklt-update-15000-events

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/004-eklt-update-20000-events \
      --frontend EKLT --name "EKLT update every 20000 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=20000

    cleanup $1/$DATE-eklt-update-strategy-n-events/004-eklt-update-20000-events

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-update-strategy-n-events/005-eklt-update-25000-events \
      --frontend EKLT --name "EKLT update every 25000 events" --overrides eklt_ekf_update_strategy=every-n-events eklt_ekf_update_every_n=25000

    cleanup $1/$DATE-eklt-update-strategy-n-events/005-eklt-update-25000-events

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-update-strategy-n-events/ --output_folder
  $1/$DATE-eklt-update-strategy-n-events/results

fi



if [ $EXPLORE_EKLT_INTERPOLATION_TIMESTAMP -gt 0 ]
then
  echo
  echo "Performing EKLT INTERPOLATION_TIMESTAMP exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpolation-ts/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-interpolation-ts/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpolation-ts/001-eklt-baseline \
      --frontend EKLT --name "EKLT baseline"

    cleanup $1/$DATE-eklt-interpolation-ts/001-eklt-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpolation-ts/002-eklt-interpolation-t-avg \
      --frontend EKLT --name "EKLT ekf update ts AVG" --overrides eklt_ekf_update_timestamp=patches-average

    cleanup $1/$DATE-eklt-interpolation-ts/002-eklt-interpolation-t-avg

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpolation-ts/003-eklt-interpolation-t-max \
      --frontend EKLT --name "EKLT ekf update ts MAX" --overrides eklt_ekf_update_timestamp=patches-maximum

    cleanup $1/$DATE-eklt-interpolation-ts/003-eklt-interpolation-t-max

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpolation-ts/004-eklt-interpolation-latest-ev \
      --frontend EKLT --name "EKLT ekf update ts latest event" --overrides eklt_ekf_update_timestamp=latest-event-ts

    cleanup $1/$DATE-eklt-interpolation-ts/004-eklt-interpolation-latest-ev

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-interpolation-ts/ --output_folder $1/$DATE-eklt-interpolation-ts/results

fi





if [ $EXPLORE_EKLT_FEATURE_INTERPOLATION -gt 0 ]
then
  echo
  echo "Performing EKLT FEATURE_INTERPOLATION exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-feature-interpolation/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-feature-interpolation/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-feature-interpolation/001-eklt-nearest-neighbor \
      --frontend EKLT --name "EKLT 10ms feat interpol NN" --overrides eklt_ekf_feature_interpolation=nearest-neighbor eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-feature-interpolation/001-eklt-nearest-neighbor


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-feature-interpolation/002-eklt-relative-limit \
      --frontend EKLT --name "EKLT 10ms linear-relative-limit 1.0" --overrides eklt_ekf_feature_interpolation=linear-relative-limit eklt_ekf_feature_extrapolation_limit=1.0 eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-feature-interpolation/002-eklt-relative-limit

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-feature-interpolation/003-eklt-absolute-limit \
      --frontend EKLT --name "EKLT 10ms linear-absolute-limit 5ms" --overrides eklt_ekf_feature_interpolation=linear-absolute-limit eklt_ekf_feature_extrapolation_limit=5 eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-feature-interpolation/003-eklt-absolute-limit


  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-feature-interpolation/ --output_folder
  $1/$DATE-eklt-feature-interpolation/results

fi




if [ $EXPLORE_EKLT_FEATURE_INTERPOLATION_RELATIVE_LIMIT -gt 0 ]
then
  echo
  echo "Performing EKLT FEATURE_INTERPOLATION_RELATIVE_LIMIT exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpol-rel-limit/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-interpol-rel-limit/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpol-rel-limit/001-eklt-rl-0.5 \
      --frontend EKLT --name "EKLT 10ms linear-relative-limit 0.5" --overrides eklt_ekf_feature_interpolation=linear-relative-limit eklt_ekf_feature_extrapolation_limit=0.5 eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-interpol-rel-limit/001-eklt-rl-0.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpol-rel-limit/002-eklt-rl-1.5 \
      --frontend EKLT --name "EKLT 10ms linear-relative-limit 1.5" --overrides eklt_ekf_feature_interpolation=linear-relative-limit eklt_ekf_feature_extrapolation_limit=1.5 eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-interpol-rel-limit/002-eklt-rl-1.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpol-rel-limit/003-eklt-rl-5.0 \
      --frontend EKLT --name "EKLT 10ms linear-relative-limit 5.0" --overrides eklt_ekf_feature_interpolation=linear-relative-limit eklt_ekf_feature_extrapolation_limit=5.0 eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-interpol-rel-limit/003-eklt-rl-5.0

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-interpol-rel-limit/ --output_folder
  $1/$DATE-eklt-interpol-rel-limit/results

fi



if [ $EXPLORE_EKLT_FEATURE_INTERPOLATION_ABSOLUTE_LIMIT -gt 0 ]
then
  echo
  echo "Performing EKLT FEATURE_INTERPOLATION_ABSOLUTE_LIMIT exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpol-abs-limit/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-interpol-abs-limit/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpol-abs-limit/001-eklt-al-1 \
      --frontend EKLT --name "EKLT 10ms linear-absolute-limit 1ms" --overrides eklt_ekf_feature_interpolation=linear-absolute-limit eklt_ekf_feature_extrapolation_limit=0.001 eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-interpol-abs-limit/001-eklt-al-1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpol-abs-limit/002-eklt-al-1.5 \
      --frontend EKLT --name "EKLT 10ms linear-absolute-limit 5ms" --overrides eklt_ekf_feature_interpolation=linear-absolute-limit eklt_ekf_feature_extrapolation_limit=0.005 eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-interpol-abs-limit/002-eklt-al-1.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpol-abs-limit/003-eklt-al-5.0 \
      --frontend EKLT --name "EKLT 10ms linear-absolute-limit 10ms" --overrides eklt_ekf_feature_interpolation=linear-absolute-limit eklt_ekf_feature_extrapolation_limit=0.010 eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-interpol-abs-limit/003-eklt-al-5.0

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-interpol-abs-limit/004-eklt-al-10.0 \
      --frontend EKLT --name "EKLT 10ms linear-absolute-limit 30ms" --overrides eklt_ekf_feature_interpolation=linear-absolute-limit eklt_ekf_feature_extrapolation_limit=0.030 eklt_ekf_update_strategy=every-n-msec-with-events eklt_ekf_update_every_n=10

    cleanup $1/$DATE-eklt-interpol-abs-limit/004-eklt-al-10.0

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-interpol-abs-limit/ --output_folder
  $1/$DATE-eklt-interpol-abs-limit/results

fi



if [ $EXPLORE_EKLT_LINLOG_SCALE -gt 0 ]
then
  echo
  echo "Performing lin-log scale exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-linlog-scale/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-linlog-scale/001-eklt-default-log-scale \
      --frontend EKLT --name "EKLT default log scale" --overrides eklt_use_linlog_scale=false

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-linlog-scale/002-eklt-novel-linlog-scale \
      --frontend EKLT --name "EKLT linlog scale" --overrides eklt_use_linlog_scale=true

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-linlog-scale/ --output_folder $1/$DATE-eklt-linlog-scale/results

fi


if [ $EXPLORE_EKLT_PATCH_TIMESTAMP_ASSIGNMENT -gt 0 ]
then
  echo
  echo "Performing EKLT patch timestamp assignment exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-ts-assignment/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-ts-assignment/001-eklt-latest-event-pmax \
      --frontend EKLT --name "EKLT pmax ts latest-event" --overrides eklt_patch_timestamp_assignment=latest-event eklt_ekf_update_timestamp=patches-maximum

    cleanup $1/$DATE-eklt-patch-ts-assignment/001-eklt-latest-event-pmax
    
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-ts-assignment/002-eklt-accumulated-events-center-pmax \
      --frontend EKLT --name "EKLT pmax ts events-center" --overrides eklt_patch_timestamp_assignment=accumulated-events-center eklt_ekf_update_timestamp=patches-maximum

    cleanup $1/$DATE-eklt-patch-ts-assignment/002-eklt-accumulated-events-center-pmax

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-ts-assignment/003-eklt-latest-event-pavg \
      --frontend EKLT --name "EKLT pavg ts latest-event" --overrides eklt_patch_timestamp_assignment=latest-event eklt_ekf_update_timestamp=patches-average

    cleanup $1/$DATE-eklt-patch-ts-assignment/003-eklt-latest-event-pavg
    
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-patch-ts-assignment/004-eklt-accumulated-events-center-pavg \
      --frontend EKLT --name "EKLT pavg ts events-center" --overrides eklt_patch_timestamp_assignment=accumulated-events-center eklt_ekf_update_timestamp=patches-average

    cleanup $1/$DATE-eklt-patch-ts-assignment/004-eklt-accumulated-events-center-pavg

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-patch-ts-assignment/ --output_folder
  $1/$DATE-eklt-patch-ts-assignment/results

fi



if [ $EXPLORE_EKLT_SIGMA_IMG -gt 0 ]
then
  echo
  echo "Performing EKLT sigma_img exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-sigma-img/000-baseline --frontend \
     XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-sigma-img/000-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-sigma-img/001-sigma-img-1-f --frontend \
     EKLT --name "EKLT sigma_img=1/f" --overrides sigma_img=0.005022794282273277

    cleanup $1/$DATE-eklt-sigma-img/001-sigma-img-1-f

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-sigma-img/002-sigma-img-2.5-f --frontend \
     EKLT --name "EKLT sigma_img=2.5/f" --overrides sigma_img=0.012556985705683194

    cleanup $1/$DATE-eklt-sigma-img/002-sigma-img-2.5-f

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-sigma-img/003-sigma-img-4-f --frontend \
     EKLT --name "EKLT sigma_img=4/f" --overrides sigma_img=0.02009117712909311

    cleanup $1/$DATE-eklt-sigma-img/003-sigma-img-4-f

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-sigma-img/004-sigma-img-5-f --frontend \
     EKLT --name "EKLT sigma_img=5/f" --overrides sigma_img=0.025113971411366388

    cleanup $1/$DATE-eklt-sigma-img/004-sigma-img-5-f
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-sigma-img/005-sigma-img-6-f --frontend \
     EKLT --name "EKLT sigma_img=6/f" --overrides sigma_img=0.030136765693639666

    cleanup $1/$DATE-eklt-sigma-img/005-sigma-img-6-f
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-sigma-img/006-sigma-img-7.5-f --frontend \
     EKLT --name "EKLT sigma_img=7.5/f" --overrides sigma_img=0.03767095711704958

    cleanup $1/$DATE-eklt-sigma-img/006-sigma-img-7.5-f
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-sigma-img/007-sigma-img-8.86-f --frontend \
     EKLT --name "EKLT sigma_img=8.86/f" --overrides sigma_img=0.044501957340941235

    cleanup $1/$DATE-eklt-sigma-img/007-sigma-img-8.86-f
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-sigma-img/008-sigma-img-10-f --frontend \
     EKLT --name "EKLT sigma_img=10/f" --overrides sigma_img=0.050227942822732775

    cleanup $1/$DATE-eklt-sigma-img/008-sigma-img-10-f

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-sigma-img/ --output_folder $1/$DATE-eklt-sigma-img/results

fi


if [ $EXPLORE_EKLT_HARRIS_K -gt 0 ]
then
  echo
  echo "Performing EKLT HARRIS K exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-k/000-xvio-baseline --frontend \
     XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-harris-k/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-k/001-harris-k-0 --frontend \
     EKLT --name "EKLT harris_k=0" --overrides eklt_harris_k=0

    cleanup $1/$DATE-eklt-harris-k/001-harris-k-0

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-k/002-harris-k-0.01 --frontend \
     EKLT --name "EKLT harris_k=0.01" --overrides eklt_harris_k=0.01

    cleanup $1/$DATE-eklt-harris-k/002-harris-k-0.01

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-k/003-harris-k-0.025 --frontend \
     EKLT --name "EKLT harris_k=0.025" --overrides eklt_harris_k=0.025

    cleanup $1/$DATE-eklt-harris-k/003-harris-k-0.025

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-k/004-harris-k-0.04 --frontend \
     EKLT --name "EKLT harris_k=0.04" --overrides eklt_harris_k=0.04

    cleanup $1/$DATE-eklt-harris-k/004-harris-k-0.04
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-k/005-harris-k-0.06 --frontend \
     EKLT --name "EKLT harris_k=0.06" --overrides eklt_harris_k=0.06

    cleanup $1/$DATE-eklt-harris-k/005-harris-k-0.06
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-k/006-harris-k-0.08 --frontend \
     EKLT --name "EKLT harris_k=0.08" --overrides eklt_harris_k=0.08

    cleanup $1/$DATE-eklt-harris-k/006-harris-k-0.08
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-k/007-harris-k-0.1 --frontend \
     EKLT --name "EKLT harris_k=0.1" --overrides eklt_harris_k=0.1

    cleanup $1/$DATE-eklt-harris-k/007-harris-k-0.1
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-k/008-harris-k-0.2 --frontend \
     EKLT --name "EKLT harris_k=0.2" --overrides eklt_harris_k=0.2

    cleanup $1/$DATE-eklt-harris-k/008-harris-k-0.2

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-harris-k/ --output_folder $1/$DATE-eklt-harris-k/results

fi


if [ $EXPLORE_EKLT_HARRIS_QL -gt 0 ]
then
  echo
  echo "Performing EKLT HARRIS QL exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-ql/000-xvio-baseline --frontend \
     XVIO --name "XVIO baseline"

    cleanup $1/$DATE-eklt-harris-ql/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-ql/002-harris-ql-0.05 --frontend \
     EKLT --name "EKLT harris_ql=0.05" --overrides eklt_harris_quality_level=0.05

    cleanup $1/$DATE-eklt-harris-ql/002-harris-ql-0.05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-ql/003-harris-ql-0.1 --frontend \
     EKLT --name "EKLT harris_ql=0.1" --overrides eklt_harris_quality_level=0.1

    cleanup $1/$DATE-eklt-harris-ql/003-harris-ql-0.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-ql/004-harris-ql-0.15 --frontend \
     EKLT --name "EKLT harris_ql=0.15" --overrides eklt_harris_quality_level=0.15

    cleanup $1/$DATE-eklt-harris-ql/004-harris-ql-0.15
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-ql/005-harris-ql-0.2 --frontend \
     EKLT --name "EKLT harris_ql=0.2" --overrides eklt_harris_quality_level=0.2

    cleanup $1/$DATE-eklt-harris-ql/005-harris-ql-0.2
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-ql/006-harris-ql-0.25 --frontend \
     EKLT --name "EKLT harris_ql=0.25" --overrides eklt_harris_quality_level=0.25

    cleanup $1/$DATE-eklt-harris-ql/006-harris-ql-0.25
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-ql/007-harris-ql-0.3 --frontend \
     EKLT --name "EKLT harris_ql=0.3" --overrides eklt_harris_quality_level=0.3

    cleanup $1/$DATE-eklt-harris-ql/007-harris-ql-0.3
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-eklt-harris-ql/008-harris-ql-0.35 --frontend \
     EKLT --name "EKLT harris_ql=0.35" --overrides eklt_harris_quality_level=0.35

    cleanup $1/$DATE-eklt-harris-ql/008-harris-ql-0.35

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-eklt-harris-ql/ --output_folder $1/$DATE-eklt-harris-ql/results

fi



if [ $EXPLORE_HASTE_OUTLIER_METHOD -gt 0 ]
then
  echo
  echo "Performing HASTE outlier_method exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/000-baseline --frontend \
     HASTE --name "HASTE baseline"

    cleanup $1/$DATE-haste-outlier-method/000-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/000-haste-remove-outliers-off \
      --frontend HASTE --name "HASTE outlier removal OFF" --overrides haste_enable_outlier_removal=false

    cleanup $1/$DATE-haste-outlier-method/000-haste-remove-outliers-off

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/001-ransac-px-1.2 --frontend \
     HASTE --name "HASTE RANSAC px=1.2" --overrides haste_outlier_method=8 haste_outlier_param1=1.2 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/001-ransac-px-1.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/002-ransac-px-1.3 --frontend \
     HASTE --name "HASTE RANSAC px=1.3" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/002-ransac-px-1.3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/003-ransac-px-1.4 --frontend \
     HASTE --name "HASTE RANSAC px=1.4" --overrides haste_outlier_method=8 haste_outlier_param1=1.4 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/003-ransac-px-1.4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/004-ransac-px-1.5 --frontend \
     HASTE --name "HASTE RANSAC px=1.5" --overrides haste_outlier_method=8 haste_outlier_param1=1.5 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/004-ransac-px-1.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/005-ransac-px-1.6 --frontend \
     HASTE --name "HASTE RANSAC px=1.6" --overrides haste_outlier_method=8 haste_outlier_param1=1.6 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/005-ransac-px-1.6

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/006-ransac-px-1.7 --frontend \
     HASTE --name "HASTE RANSAC px=1.7" --overrides haste_outlier_method=8 haste_outlier_param1=1.7 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/006-ransac-px-1.7

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/007-ransac-px-1.8 --frontend \
     HASTE --name "HASTE RANSAC px=1.8" --overrides haste_outlier_method=8 haste_outlier_param1=1.8 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/007-ransac-px-1.8

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/008-ransac-px-1.9 --frontend \
     HASTE --name "HASTE RANSAC px=1.9" --overrides haste_outlier_method=8 haste_outlier_param1=1.9 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/008-ransac-px-1.9

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/009-ransac-px-2.0 --frontend \
     HASTE --name "HASTE RANSAC px=2.0" --overrides haste_outlier_method=8 haste_outlier_param1=2.0 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/009-ransac-px-2.0

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/010-ransac-px-1.0 --frontend \
     HASTE --name "HASTE RANSAC px=1.0" --overrides haste_outlier_method=8 haste_outlier_param1=1.0 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/010-ransac-px-1.0

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/011-ransac-px-1.1 --frontend \
     HASTE --name "HASTE RANSAC px=1.1" --overrides haste_outlier_method=8 haste_outlier_param1=1.1 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/011-ransac-px-1.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/012-ransac-px-1.2 --frontend \
     HASTE --name "HASTE RANSAC px=1.2" --overrides haste_outlier_method=8 haste_outlier_param1=1.2 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/012-ransac-px-1.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/013-ransac-px-1.3 --frontend \
     HASTE --name "HASTE RANSAC px=1.3" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/013-ransac-px-1.3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/014-ransac-p-0.8 --frontend \
     HASTE --name "HASTE RANSAC px=1.3 p=0.8" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.8

    cleanup $1/$DATE-haste-outlier-method/014-ransac-p-0.8

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/015-ransac-p-0.85 --frontend \
     HASTE --name "HASTE RANSAC px=1.3 p=0.85" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.85

    cleanup $1/$DATE-haste-outlier-method/015-ransac-p-0.85

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/016-ransac-p-0.9 --frontend \
     HASTE --name "HASTE RANSAC px=1.3 p=0.9" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.9

    cleanup $1/$DATE-haste-outlier-method/016-ransac-p-0.9

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/017-ransac-p-0.95 --frontend \
     HASTE --name "HASTE RANSAC px=1.3 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method/017-ransac-p-0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/017-ransac-p-0.98 --frontend \
     HASTE --name "HASTE RANSAC px=1.3 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method/017-ransac-p-0.98

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method/017-ransac-p-0.99 --frontend \
     HASTE --name "HASTE RANSAC px=1.3 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method/017-ransac-p-0.99
    
  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-outlier-method/ --output_folder $1/$DATE-haste-outlier-method/results

fi


if [ $EXPLORE_HASTE_OUTLIER_METHOD_95 -gt 0 ]
then
  echo
  echo "Performing HASTE outlier_method exploration 0.95"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/000-xvio-baseline --frontend \
     XVIO --name "XVIO baseline"

    cleanup $1/$DATE-haste-outlier-method-95/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/001-haste-baseline --frontend \
     HASTE --name "HASTE baseline"

    cleanup $1/$DATE-haste-outlier-method-95/001-haste-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/002-ransax-px-0.1 --frontend \
     HASTE --name "HASTE RANSAC px=0.1 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.1 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/002-ransax-px-0.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/003-ransax-px-0.2 --frontend \
     HASTE --name "HASTE RANSAC px=0.2 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.2 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/003-ransax-px-0.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/004-ransax-px-0.3 --frontend \
     HASTE --name "HASTE RANSAC px=0.3 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.3 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/004-ransax-px-0.3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/005-ransax-px-0.4 --frontend \
     HASTE --name "HASTE RANSAC px=0.4 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.4 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/005-ransax-px-0.4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/006-ransax-px-0.5 --frontend \
     HASTE --name "HASTE RANSAC px=0.5 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.5 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/006-ransax-px-0.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/007-ransax-px-0.6 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/007-ransax-px-0.6

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/008-ransax-px-0.7 --frontend \
     HASTE --name "HASTE RANSAC px=0.7 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.7 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/008-ransax-px-0.7

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/009-ransax-px-0.8 --frontend \
     HASTE --name "HASTE RANSAC px=0.8 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.8 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/009-ransax-px-0.8

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/010-ransax-px-0.9 --frontend \
     HASTE --name "HASTE RANSAC px=0.9 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.9 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/010-ransax-px-0.9

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/011-ransax-px-1.0 --frontend \
     HASTE --name "HASTE RANSAC px=1.0 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=1.0 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/011-ransax-px-1.0

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/012-ransax-px-1.1 --frontend \
     HASTE --name "HASTE RANSAC px=1.1 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=1.1 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/012-ransax-px-1.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/013-ransax-px-1.2 --frontend \
     HASTE --name "HASTE RANSAC px=1.2 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=1.2 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/013-ransax-px-1.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/014-ransax-px-1.3 --frontend \
     HASTE --name "HASTE RANSAC px=1.3 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/014-ransax-px-1.3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/015-ransax-px-1.4 --frontend \
     HASTE --name "HASTE RANSAC px=1.4 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=1.4 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/015-ransax-px-1.4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/016-ransax-px-1.5 --frontend \
     HASTE --name "HASTE RANSAC px=1.5 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=1.5 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/016-ransax-px-1.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/017-ransax-px-1.6 --frontend \
     HASTE --name "HASTE RANSAC px=1.6 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=1.6 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/017-ransax-px-1.6

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/018-ransax-px-1.7 --frontend \
     HASTE --name "HASTE RANSAC px=1.7 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=1.7 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/018-ransax-px-1.7

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-95/019-ransax-px-1.8 --frontend \
     HASTE --name "HASTE RANSAC px=1.8 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=1.8 haste_outlier_param2=0.95

    cleanup $1/$DATE-haste-outlier-method-95/019-ransax-px-1.8
    
  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-outlier-method-95/ --output_folder
  $1/$DATE-haste-outlier-method-95/results

fi



if [ $EXPLORE_HASTE_OUTLIER_METHOD_98 -gt 0 ]
then
  echo
  echo "Performing HASTE outlier_method exploration 0.98"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/000-xvio-baseline --frontend \
     XVIO --name "XVIO baseline"

    cleanup $1/$DATE-haste-outlier-method-98/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/001-haste-baseline --frontend \
     HASTE --name "HASTE baseline"

    cleanup $1/$DATE-haste-outlier-method-98/001-haste-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/002-ransax-px-0.1 --frontend \
     HASTE --name "HASTE RANSAC px=0.1 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=0.1 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/002-ransax-px-0.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/003-ransax-px-0.2 --frontend \
     HASTE --name "HASTE RANSAC px=0.2 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=0.2 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/003-ransax-px-0.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/004-ransax-px-0.3 --frontend \
     HASTE --name "HASTE RANSAC px=0.3 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=0.3 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/004-ransax-px-0.3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/005-ransax-px-0.4 --frontend \
     HASTE --name "HASTE RANSAC px=0.4 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=0.4 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/005-ransax-px-0.4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/006-ransax-px-0.5 --frontend \
     HASTE --name "HASTE RANSAC px=0.5 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=0.5 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/006-ransax-px-0.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/007-ransax-px-0.6 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/007-ransax-px-0.6

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/008-ransax-px-0.7 --frontend \
     HASTE --name "HASTE RANSAC px=0.7 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=0.7 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/008-ransax-px-0.7

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/009-ransax-px-0.8 --frontend \
     HASTE --name "HASTE RANSAC px=0.8 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=0.8 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/009-ransax-px-0.8

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/010-ransax-px-0.9 --frontend \
     HASTE --name "HASTE RANSAC px=0.9 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=0.9 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/010-ransax-px-0.9

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/011-ransax-px-1.0 --frontend \
     HASTE --name "HASTE RANSAC px=1.0 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=1.0 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/011-ransax-px-1.0

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/012-ransax-px-1.1 --frontend \
     HASTE --name "HASTE RANSAC px=1.1 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=1.1 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/012-ransax-px-1.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/013-ransax-px-1.2 --frontend \
     HASTE --name "HASTE RANSAC px=1.2 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=1.2 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/013-ransax-px-1.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/014-ransax-px-1.3 --frontend \
     HASTE --name "HASTE RANSAC px=1.3 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/014-ransax-px-1.3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/015-ransax-px-1.4 --frontend \
     HASTE --name "HASTE RANSAC px=1.4 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=1.4 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/015-ransax-px-1.4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/016-ransax-px-1.5 --frontend \
     HASTE --name "HASTE RANSAC px=1.5 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=1.5 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/016-ransax-px-1.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/017-ransax-px-1.6 --frontend \
     HASTE --name "HASTE RANSAC px=1.6 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=1.6 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/017-ransax-px-1.6

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/018-ransax-px-1.7 --frontend \
     HASTE --name "HASTE RANSAC px=1.7 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=1.7 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/018-ransax-px-1.7

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-98/019-ransax-px-1.8 --frontend \
     HASTE --name "HASTE RANSAC px=1.8 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=1.8 haste_outlier_param2=0.98

    cleanup $1/$DATE-haste-outlier-method-98/019-ransax-px-1.8
    
  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-outlier-method-98/ --output_folder
  $1/$DATE-haste-outlier-method-98/results

fi



if [ $EXPLORE_HASTE_OUTLIER_METHOD_99 -gt 0 ]
then
  echo
  echo "Performing HASTE outlier_method exploration 0.99"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/000-xvio-baseline --frontend \
     XVIO --name "XVIO baseline"

    cleanup $1/$DATE-haste-outlier-method-99/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/001-haste-baseline --frontend \
     HASTE --name "HASTE baseline"

    cleanup $1/$DATE-haste-outlier-method-99/001-haste-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/002-ransax-px-0.1 --frontend \
     HASTE --name "HASTE RANSAC px=0.1 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=0.1 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/002-ransax-px-0.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/003-ransax-px-0.2 --frontend \
     HASTE --name "HASTE RANSAC px=0.2 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=0.2 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/003-ransax-px-0.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/004-ransax-px-0.3 --frontend \
     HASTE --name "HASTE RANSAC px=0.3 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=0.3 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/004-ransax-px-0.3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/005-ransax-px-0.4 --frontend \
     HASTE --name "HASTE RANSAC px=0.4 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=0.4 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/005-ransax-px-0.4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/006-ransax-px-0.5 --frontend \
     HASTE --name "HASTE RANSAC px=0.5 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=0.5 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/006-ransax-px-0.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/007-ransax-px-0.6 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/007-ransax-px-0.6

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/008-ransax-px-0.7 --frontend \
     HASTE --name "HASTE RANSAC px=0.7 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=0.7 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/008-ransax-px-0.7

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/009-ransax-px-0.8 --frontend \
     HASTE --name "HASTE RANSAC px=0.8 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=0.8 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/009-ransax-px-0.8

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/010-ransax-px-0.9 --frontend \
     HASTE --name "HASTE RANSAC px=0.9 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=0.9 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/010-ransax-px-0.9

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/011-ransax-px-1.0 --frontend \
     HASTE --name "HASTE RANSAC px=1.0 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=1.0 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/011-ransax-px-1.0

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/012-ransax-px-1.1 --frontend \
     HASTE --name "HASTE RANSAC px=1.1 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=1.1 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/012-ransax-px-1.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/013-ransax-px-1.2 --frontend \
     HASTE --name "HASTE RANSAC px=1.2 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=1.2 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/013-ransax-px-1.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/014-ransax-px-1.3 --frontend \
     HASTE --name "HASTE RANSAC px=1.3 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/014-ransax-px-1.3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/015-ransax-px-1.4 --frontend \
     HASTE --name "HASTE RANSAC px=1.4 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=1.4 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/015-ransax-px-1.4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/016-ransax-px-1.5 --frontend \
     HASTE --name "HASTE RANSAC px=1.5 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=1.5 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/016-ransax-px-1.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/017-ransax-px-1.6 --frontend \
     HASTE --name "HASTE RANSAC px=1.6 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=1.6 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/017-ransax-px-1.6

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/018-ransax-px-1.7 --frontend \
     HASTE --name "HASTE RANSAC px=1.7 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=1.7 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/018-ransax-px-1.7

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-99/019-ransax-px-1.8 --frontend \
     HASTE --name "HASTE RANSAC px=1.8 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=1.8 haste_outlier_param2=0.99

    cleanup $1/$DATE-haste-outlier-method-99/019-ransax-px-1.8
    
  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-outlier-method-99/ --output_folder
  $1/$DATE-haste-outlier-method-99/results

fi



if [ $EXPLORE_HASTE_OUTLIER_METHOD_EVRY_MSG -gt 0 ]
then
  echo
  echo "Performing HASTE outlier_method EVRY_MSG exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/000-baseline --frontend \
     HASTE --name "HASTE baseline"

    cleanup $1/$DATE-haste-outlier-method-evry-msg/000-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/000-haste-remove-outliers-off \
      --frontend HASTE --name "HASTE outlier removal OFF" --overrides haste_enable_outlier_removal=false haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/000-haste-remove-outliers-off

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/001-ransac-px-0.1 --frontend \
     HASTE --name "HASTE RANSAC px=0.1" --overrides haste_outlier_method=8 haste_outlier_param1=0.1 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/001-ransac-px-0.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/002-ransac-px-0.2 --frontend \
     HASTE --name "HASTE RANSAC px=0.2" --overrides haste_outlier_method=8 haste_outlier_param1=0.2 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/002-ransac-px-0.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/003-ransac-px-0.3 --frontend \
     HASTE --name "HASTE RANSAC px=0.3" --overrides haste_outlier_method=8 haste_outlier_param1=0.3 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/003-ransac-px-0.3

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/004-ransac-px-0.4 --frontend \
     HASTE --name "HASTE RANSAC px=0.4" --overrides haste_outlier_method=8 haste_outlier_param1=0.4 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/004-ransac-px-0.4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/005-ransac-px-0.5 --frontend \
     HASTE --name "HASTE RANSAC px=0.5" --overrides haste_outlier_method=8 haste_outlier_param1=0.5 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/005-ransac-px-0.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/006-ransac-px-0.6 --frontend \
     HASTE --name "HASTE RANSAC px=0.6" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/006-ransac-px-0.6

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/007-ransac-px-0.7 --frontend \
     HASTE --name "HASTE RANSAC px=0.7" --overrides haste_outlier_method=8 haste_outlier_param1=0.7 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/007-ransac-px-0.7

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/008-ransac-px-0.8 --frontend \
     HASTE --name "HASTE RANSAC px=0.8" --overrides haste_outlier_method=8 haste_outlier_param1=0.8 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/008-ransac-px-0.8

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/009-ransac-px-0.9 --frontend \
     HASTE --name "HASTE RANSAC px=0.9" --overrides haste_outlier_method=8 haste_outlier_param1=0.9 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/009-ransac-px-0.9

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/010-ransac-px-1.0 --frontend \
     HASTE --name "HASTE RANSAC px=1.0" --overrides haste_outlier_method=8 haste_outlier_param1=1.0 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/010-ransac-px-1.0

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/011-ransac-px-1.1 --frontend \
     HASTE --name "HASTE RANSAC px=1.1" --overrides haste_outlier_method=8 haste_outlier_param1=1.1 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/011-ransac-px-1.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/012-ransac-px-1.2 --frontend \
     HASTE --name "HASTE RANSAC px=1.2" --overrides haste_outlier_method=8 haste_outlier_param1=1.2 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/012-ransac-px-1.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/013-ransac-px-1.3 --frontend \
     HASTE --name "HASTE RANSAC px=1.3" --overrides haste_outlier_method=8 haste_outlier_param1=1.3 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/013-ransac-px-0.85

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/014-ransac-p-0.8 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.8" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.8 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/014-ransac-p-0.8

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/015-ransac-p-0.85 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.85" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.85 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/015-ransac-p-0.85

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/016-ransac-p-0.9 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.9" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.9 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/016-ransac-p-0.9

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-outlier-method-evry-msg/017-ransac-p-0.95 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.95 haste_ekf_update_strategy=every-ros-event-message

    cleanup $1/$DATE-haste-outlier-method-evry-msg/017-ransac-p-0.95
    
  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-outlier-method-evry-msg/ --output_folder
  $1/$DATE-haste-outlier-method-evry-msg/results

fi





if [ $EXPLORE_HASTE_DIFF_HASTE_OUTLIER_METHOD -gt 0 ]
then
  echo
  echo "Performing HASTE DIFF haste_outlier_method exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/000-xvio-baseline --frontend \
     XVIO --name "XVIO baseline"

    cleanup $1/$DATE-haste-diff-outlier-method/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/001-ransac-px-0.2 --frontend \
     HASTE --name "HASTE RANSAC px=0.2" --overrides haste_outlier_method=8 haste_outlier_param1=0.2 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/001-ransac-px-0.2

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/002-ransac-px-0.25 --frontend \
     HASTE --name "HASTE RANSAC px=0.25" --overrides haste_outlier_method=8 haste_outlier_param1=0.25 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/002-ransac-px-0.25

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/003-ransac-px-0.35 --frontend \
     HASTE --name "HASTE RANSAC px=0.35" --overrides haste_outlier_method=8 haste_outlier_param1=0.35 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/003-ransac-px-0.35

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/004-ransac-px-0.4 --frontend \
     HASTE --name "HASTE RANSAC px=0.4" --overrides haste_outlier_method=8 haste_outlier_param1=0.4 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/004-ransac-px-0.4

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/005-ransac-px-0.45 --frontend \
     HASTE --name "HASTE RANSAC px=0.45" --overrides haste_outlier_method=8 haste_outlier_param1=0.45 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/005-ransac-px-0.45

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/006-ransac-px-0.5 --frontend \
     HASTE --name "HASTE RANSAC px=0.5" --overrides haste_outlier_method=8 haste_outlier_param1=0.5 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/006-ransac-px-0.5

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/007-ransac-px-0.55 --frontend \
     HASTE --name "HASTE RANSAC px=0.55" --overrides haste_outlier_method=8 haste_outlier_param1=0.55 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/007-ransac-px-0.55

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/008-ransac-px-0.6 --frontend \
     HASTE --name "HASTE RANSAC px=0.6" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/008-ransac-px-0.6

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/009-ransac-px-0.65 --frontend \
     HASTE --name "HASTE RANSAC px=0.65" --overrides haste_outlier_method=8 haste_outlier_param1=0.65 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/009-ransac-px-0.65

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/010-ransac-px-0.7 --frontend \
     HASTE --name "HASTE RANSAC px=0.7" --overrides haste_outlier_method=8 haste_outlier_param1=0.7 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/010-ransac-px-0.7

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/011-ransac-px-0.75 --frontend \
     HASTE --name "HASTE RANSAC px=0.75" --overrides haste_outlier_method=8 haste_outlier_param1=0.75 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/011-ransac-px-0.75

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/012-ransac-px-0.8 --frontend \
     HASTE --name "HASTE RANSAC px=0.8" --overrides haste_outlier_method=8 haste_outlier_param1=0.8 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/012-ransac-px-0.8

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/013-ransac-px-0.85 --frontend \
     HASTE --name "HASTE RANSAC px=0.85" --overrides haste_outlier_method=8 haste_outlier_param1=0.85 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/013-ransac-px-0.85

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/014-ransac-p-0.95 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.95" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.95 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/014-ransac-p-0.95

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/015-ransac-p-0.98 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.98" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.98 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/015-ransac-p-0.98

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/016-ransac-p-0.99 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.99" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.99 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/016-ransac-p-0.99

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-diff-outlier-method/017-ransac-p-0.999 --frontend \
     HASTE --name "HASTE RANSAC px=0.6 p=0.999" --overrides haste_outlier_method=8 haste_outlier_param1=0.6 haste_outlier_param2=0.999 haste_tracker_type=haste-difference

    cleanup $1/$DATE-haste-diff-outlier-method/017-ransac-p-0.999
    
  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-diff-outlier-method/ --output_folder
  $1/$DATE-haste-diff-outlier-method/results

fi


if [ $EXPLORE_HASTE_TYPE -gt 0 ]
then
  echo
  echo "Performing HASTE type exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-tracker-type/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-tracker-type/001-type-correlation \
      --frontend HASTE --name "HASTE (correlation)" --overrides haste_tracker_type=correlation

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-tracker-type/002-type-haste-correlation \
      --frontend HASTE --name "HASTE (haste-correlation)" --overrides haste_tracker_type=haste-correlation

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-tracker-type/003-type-haste-correlation-star \
      --frontend HASTE --name "HASTE (haste-correlation-star)" --overrides haste_tracker_type=haste-correlation-star

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-tracker-type/004-type-haste-difference \
      --frontend HASTE --name "HASTE (haste-difference)" --overrides haste_tracker_type=haste-difference

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-tracker-type/005-type-haste-difference-star \
      --frontend HASTE --name "HASTE (haste-difference-star)" --overrides haste_tracker_type=haste-difference-star

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-tracker-type/ --output_folder $1/$DATE-haste-tracker-type/results

fi


if [ $EXPLORE_HASTE_INTERPOLATION_TIMESTAMP -gt 0 ]
then
  echo
  echo "Performing HASTE INTERPOLATION_TIMESTAMP exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-interpolation-ts/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-haste-interpolation-ts/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-interpolation-ts/001-haste-baseline \
      --frontend HASTE --name "HASTE baseline"

    cleanup $1/$DATE-haste-interpolation-ts/001-haste-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-interpolation-ts/002-haste-interpolation-t-avg \
      --frontend HASTE --name "HASTE ekf update ts AVG" --overrides haste_ekf_update_timestamp=patches-average

    cleanup $1/$DATE-haste-interpolation-ts/002-haste-interpolation-t-avg

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-interpolation-ts/003-haste-interpolation-t-max \
      --frontend HASTE --name "HASTE ekf update ts MAX" --overrides haste_ekf_update_timestamp=patches-maximum

    cleanup $1/$DATE-haste-interpolation-ts/003-haste-interpolation-t-max

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-interpolation-ts/004-haste-interpolation-latest-ev \
      --frontend HASTE --name "HASTE ekf update ts latest event" --overrides haste_ekf_update_timestamp=latest-event-ts

    cleanup $1/$DATE-haste-interpolation-ts/004-haste-interpolation-latest-ev

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-interpolation-ts/ --output_folder
  $1/$DATE-haste-interpolation-ts/results

fi



if [ $EXPLORE_HASTE_FEATURE_INTERPOLATION -gt 0 ]
then
  echo
  echo "Performing HASTE FEATURE_INTERPOLATION exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-feature-interpolation/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-haste-feature-interpolation/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-feature-interpolation/001-haste-nearest-neighbor \
      --frontend HASTE --name "HASTE 10ms feat interpol NN" --overrides haste_ekf_feature_interpolation=nearest-neighbor haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=10

    cleanup $1/$DATE-haste-feature-interpolation/001-haste-nearest-neighbor


    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-feature-interpolation/002-haste-relative-limit \
      --frontend HASTE --name "HASTE 10ms linear-relative-limit 1.0" --overrides haste_ekf_feature_interpolation=linear-relative-limit haste_ekf_feature_extrapolation_limit=1.0 haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=10

    cleanup $1/$DATE-haste-feature-interpolation/002-haste-relative-limit

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-feature-interpolation/003-haste-absolute-limit \
      --frontend HASTE --name "HASTE 10ms linear-absolute-limit 5ms" --overrides haste_ekf_feature_interpolation=linear-absolute-limit haste_ekf_feature_extrapolation_limit=5 haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=10

    cleanup $1/$DATE-haste-feature-interpolation/003-haste-absolute-limit


  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-feature-interpolation/ --output_folder
  $1/$DATE-haste-feature-interpolation/results

fi




if [ $EXPLORE_HASTE_UPDATE_STRATEGY_N_MSEC -gt 0 ]
then
  echo
  echo "Performing HASTE UPDATE_STRATEGY_N_MSEC exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-msec/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-haste-update-strategy-msec/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-msec/001-haste-update-1-msec \
      --frontend HASTE --name "HASTE update every 1msec" --overrides haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=1

    cleanup $1/$DATE-haste-update-strategy-msec/001-haste-update-1-msec

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-msec/002-haste-update-2-msec \
      --frontend HASTE --name "HASTE update every 2msec" --overrides haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=2

    cleanup $1/$DATE-haste-update-strategy-msec/002-haste-update-2-msec

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-msec/003-haste-update-3-msec \
      --frontend HASTE --name "HASTE update every 3msec" --overrides haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=3

    cleanup $1/$DATE-haste-update-strategy-msec/003-haste-update-3-msec

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-msec/004-haste-update-5-msec \
      --frontend HASTE --name "HASTE update every 5msec" --overrides haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=5

    cleanup $1/$DATE-haste-update-strategy-msec/004-haste-update-5-msec

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-msec/005-haste-update-7-msec \
      --frontend HASTE --name "HASTE update every 7msec" --overrides haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=7

    cleanup $1/$DATE-haste-update-strategy-msec/005-haste-update-7-msec

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-msec/006-haste-update-9-msec \
      --frontend HASTE --name "HASTE update every 9msec" --overrides haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=9

    cleanup $1/$DATE-haste-update-strategy-msec/006-haste-update-9-msec

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-msec/007-haste-update-12-msec \
      --frontend HASTE --name "HASTE update every 12msec" --overrides haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=12

    cleanup $1/$DATE-haste-update-strategy-msec/007-haste-update-12-msec

    # python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-msec/008-haste-update-40-msec \
    #   --frontend HASTE --name "HASTE update every 40msec" --overrides haste_ekf_update_strategy=every-n-msec-with-events haste_ekf_update_every_n=40

    # cleanup $1/$DATE-haste-update-strategy-msec/008-haste-update-40-msec

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-update-strategy-msec/ --output_folder
  $1/$DATE-haste-update-strategy-msec/results

fi





if [ $EXPLORE_HASTE_UPDATE_STRATEGY_N_EVENTS -gt 0 ]
then
  echo
  echo "Performing HASTE UPDATE_STRATEGY_N_EVENTS exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-n-events/000-xvio-baseline \
      --frontend XVIO --name "XVIO baseline"

    cleanup $1/$DATE-haste-update-strategy-n-events/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-n-events/001-haste-update-500-events \
      --frontend HASTE --name "HASTE update every 500 events" --overrides haste_ekf_update_strategy=every-n-events haste_ekf_update_every_n=500

    cleanup $1/$DATE-haste-update-strategy-n-events/001-haste-update-500-events

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-n-events/002-haste-update-1000-events \
      --frontend HASTE --name "HASTE update every 1000 events" --overrides haste_ekf_update_strategy=every-n-events haste_ekf_update_every_n=1000

    cleanup $1/$DATE-haste-update-strategy-n-events/002-haste-update-1000-events

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-n-events/003-haste-update-2000-events \
      --frontend HASTE --name "HASTE update every 2000 events" --overrides haste_ekf_update_strategy=every-n-events haste_ekf_update_every_n=2000

    cleanup $1/$DATE-haste-update-strategy-n-events/003-haste-update-2000-events

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-n-events/004-haste-update-3600-events \
      --frontend HASTE --name "HASTE update every 3600 events" --overrides haste_ekf_update_strategy=every-n-events haste_ekf_update_every_n=3600

    cleanup $1/$DATE-haste-update-strategy-n-events/004-haste-update-3600-events

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-n-events/005-haste-update-4800-events \
      --frontend HASTE --name "HASTE update every 4800 events" --overrides haste_ekf_update_strategy=every-n-events haste_ekf_update_every_n=4800

    cleanup $1/$DATE-haste-update-strategy-n-events/005-haste-update-4800-events

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-n-events/006-haste-update-7200-events \
      --frontend HASTE --name "HASTE update every 7200 events" --overrides haste_ekf_update_strategy=every-n-events haste_ekf_update_every_n=7200

    cleanup $1/$DATE-haste-update-strategy-n-events/006-haste-update-7200-events

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-update-strategy-n-events/007-haste-update-9200-events \
      --frontend HASTE --name "HASTE update every 9200 events" --overrides haste_ekf_update_strategy=every-n-events haste_ekf_update_every_n=9200

    cleanup $1/$DATE-haste-update-strategy-n-events/007-haste-update-9200-events

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-update-strategy-n-events/ --output_folder
  $1/$DATE-haste-update-strategy-n-events/results

fi



if [ $EXPLORE_HASTE_HARRIS_K -gt 0 ]
then
  echo
  echo "Performing HASTE HARRIS K exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-k/000-xvio-baseline --frontend \
     XVIO --name "XVIO baseline"

    cleanup $1/$DATE-haste-harris-k/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-k/001-harris-k-0 --frontend \
     HASTE --name "HASTE harris_k=0" --overrides haste_harris_k=0

    cleanup $1/$DATE-haste-harris-k/001-harris-k-0

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-k/002-harris-k-0.01 --frontend \
     HASTE --name "HASTE harris_k=0.01" --overrides haste_harris_k=0.01

    cleanup $1/$DATE-haste-harris-k/002-harris-k-0.01

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-k/003-harris-k-0.025 --frontend \
     HASTE --name "HASTE harris_k=0.025" --overrides haste_harris_k=0.025

    cleanup $1/$DATE-haste-harris-k/003-harris-k-0.025

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-k/004-harris-k-0.04 --frontend \
     HASTE --name "HASTE harris_k=0.04" --overrides haste_harris_k=0.04

    cleanup $1/$DATE-haste-harris-k/004-harris-k-0.04
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-k/005-harris-k-0.06 --frontend \
     HASTE --name "HASTE harris_k=0.06" --overrides haste_harris_k=0.06

    cleanup $1/$DATE-haste-harris-k/005-harris-k-0.06
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-k/006-harris-k-0.08 --frontend \
     HASTE --name "HASTE harris_k=0.08" --overrides haste_harris_k=0.08

    cleanup $1/$DATE-haste-harris-k/006-harris-k-0.08
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-k/007-harris-k-0.1 --frontend \
     HASTE --name "HASTE harris_k=0.1" --overrides haste_harris_k=0.1

    cleanup $1/$DATE-haste-harris-k/007-harris-k-0.1
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-k/008-harris-k-0.2 --frontend \
     HASTE --name "HASTE harris_k=0.2" --overrides haste_harris_k=0.2

    cleanup $1/$DATE-haste-harris-k/008-harris-k-0.2

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-harris-k/ --output_folder $1/$DATE-haste-harris-k/results

fi


if [ $EXPLORE_HASTE_HARRIS_QL -gt 0 ]
then
  echo
  echo "Performing HASTE HARRIS QL exploration"
  echo


  if [ $COMPARISONS_ONLY -lt 1 ]
  then

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-ql/000-xvio-baseline --frontend \
     XVIO --name "XVIO baseline"

    cleanup $1/$DATE-haste-harris-ql/000-xvio-baseline

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-ql/001-harris-ql-0 --frontend \
     HASTE --name "HASTE harris_ql=0" --overrides haste_harris_quality_level=0

    cleanup $1/$DATE-haste-harris-ql/001-harris-ql-0

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-ql/002-harris-ql-0.05 --frontend \
     HASTE --name "HASTE harris_ql=0.05" --overrides haste_harris_quality_level=0.05

    cleanup $1/$DATE-haste-harris-ql/002-harris-ql-0.05

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-ql/003-harris-ql-0.1 --frontend \
     HASTE --name "HASTE harris_ql=0.1" --overrides haste_harris_quality_level=0.1

    cleanup $1/$DATE-haste-harris-ql/003-harris-ql-0.1

    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-ql/004-harris-ql-0.15 --frontend \
     HASTE --name "HASTE harris_ql=0.15" --overrides haste_harris_quality_level=0.15

    cleanup $1/$DATE-haste-harris-ql/004-harris-ql-0.15
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-ql/005-harris-ql-0.2 --frontend \
     HASTE --name "HASTE harris_ql=0.2" --overrides haste_harris_quality_level=0.2

    cleanup $1/$DATE-haste-harris-ql/005-harris-ql-0.2
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-ql/006-harris-ql-0.25 --frontend \
     HASTE --name "HASTE harris_ql=0.25" --overrides haste_harris_quality_level=0.25

    cleanup $1/$DATE-haste-harris-ql/006-harris-ql-0.25
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-ql/007-harris-ql-0.3 --frontend \
     HASTE --name "HASTE harris_ql=0.3" --overrides haste_harris_quality_level=0.3

    cleanup $1/$DATE-haste-harris-ql/007-harris-ql-0.3
     
    python evaluate.py --configuration $CONFIGURATION --output_folder $1/$DATE-haste-harris-ql/008-harris-ql-0.35 --frontend \
     HASTE --name "HASTE harris_ql=0.35" --overrides haste_harris_quality_level=0.35

    cleanup $1/$DATE-haste-harris-ql/008-harris-ql-0.35

  fi

  python $COMPARISON_SCRIPT --input_folder $1/$DATE-haste-harris-ql/ --output_folder $1/$DATE-haste-harris-ql/results

fi
