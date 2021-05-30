#!/usr/bin/env bash

BASE_DIR=$(pwd -P)

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
SCRIPT_PATH="$( cd -- "$(dirname "$SCRIPT_PATH/../test/evaluate.py")" >/dev/null 2>&1 ; pwd -P )"

DO_XVIO_EXPLORATION_PATCH_SIZE=1

DATE=$(date '+%Y-%m-%d')

echo "We are in '$BASE_DIR'"
echo "Assuming location of evaluation script to be '$SCRIPT_PATH'"
echo "Today is $DATE"

echo "Taking first argument as results folder: '$1'"

cd "$SCRIPT_PATH" || exit

trap 'cd "$BASE_DIR"' EXIT

if [ $DO_XVIO_EXPLORATION_PATCH_SIZE -gt 0 ]
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


