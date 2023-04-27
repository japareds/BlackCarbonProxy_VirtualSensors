#!/bin/bash
color0='\033[0m'
color1='\033[1;37m'
color1='\033[1;33m'

echo -e "${color}----------------------"
echo "*** BC concentration proxy***"
echo -e "----------------------${color0}"

#Script paramters
device='RefSt'
dataformat='data_frame'
filenames="dataSet_timeagg60min.csv" #"PR_Data2018.xlsx PR_Data2019.xlsx

predictors='BC NO2 NO O3 PM25 PM1 PM10 N T RH'
MLalgorithm='SVR'

python BC_proxy.py --device $device --dataformat $dataformat --DataSetName $filenames --predictors $predictors --algorithm $MLalgorithm --train --bfs
