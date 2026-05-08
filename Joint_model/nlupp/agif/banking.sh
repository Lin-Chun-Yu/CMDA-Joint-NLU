#!/bin/bash
#bash train.sh
output=$(python3 script.py)
IFS=$'\n' read -rd '' -a lines <<< "$output" #將輸出按行分割並處理

percents=(1 2) #1 2 5 10
datasets=(banking) #banking hotels

for line in "${lines[@]}"; do
    for dataset in "${datasets[@]}"; do # 資料集
        for percent in "${percents[@]}"; do #百分比
            run=0
            while [ $run -le 9 ]; do #fold0 ~ 9
                echo "Dataset: $dataset, Percent: $percent, Run: $run"
                log_dir="./log/${dataset}/${percent}"  #存檔資料夾
                save_dir="./save/${dataset}" #存相關資料
                dataset_dir="./data/${dataset}${run}" #資料集位置
                aug_dir="./data_aug/${dataset}${run}/${percent}/${line}" #擴增資料集位置
                python train.py -g -bs=4 -ne=100 -dd=$dataset_dir -lod=$log_dir -sd=$save_dir -ln=$line -afp=$aug_dir
                run=`expr $run + 1`
            done
        done
    done
done
#無資料擴增
#python train.py -g -bs=4 -ne=100 -dd=./data/banking0 -lod=./log/banking/1 -sd=./save/banking
#資料擴增
#python train.py -g -bs=4 -ne=100 -dd=./data/banking0 -lod=./log/banking/1 -sd=./save/banking -ln=sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt -afp=./data_aug/banking0/1/sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt


