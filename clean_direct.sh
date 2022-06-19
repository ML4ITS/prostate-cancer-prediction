#!/usr/bin/env bash
# shellcheck disable=SC2164
rm -r __pycache__
cd Baseline
find results -type f -delete
rm -r __pycache__
cd ../Models_1
find lstm -type f -delete
rm -r lightning_logs
rm -r logs
rm -r __pycache__
cd ../Models_2
find lstm -type f -delete
find cnn1d -type f -delete
find cnn1d_heads -type f -delete
find mlp -type f -delete
rm -r lightning_logs
rm -r logs
rm -r __pycache__
find boxplot -type f -delete
cd ..
$SHELL