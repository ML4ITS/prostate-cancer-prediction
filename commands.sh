#!/usr/bin/env bash
# shellcheck disable=SC2164
cd Baseline
python main.py ../dataset/cancer.csv ../dataset/nocancer.csv
cd ../Models_1
python main.py ../dataset/cancer.csv ../dataset/nocancer.csv
cd ../Models_2
python main.py ../dataset/cancer.csv ../dataset/nocancer.csv
cd ..
$SHELL