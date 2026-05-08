#!/bin/bash
#bash run.sh
output=$(python3 script.py)
IFS=$'\n' read -rd '' -a lines <<< "$output" #將輸出按行分割並處理

percents=(1 2) #1 2 5 10
dataset=MixATIS_TP

for line in "${lines[@]}"; do
    for percent in "${percents[@]}"; do #擴增百分比
        ws=1
        while [ $ws -le 5 ]; do #window size = 1~5
            run=0
            while [ $run -le 2 ]; do #跑N次取平均
                echo "Dataset: $dataset, Percent: $percent, Window_size: $ws, Run: $run"
                log_dir="./log/${dataset}/ws${ws}/${percent}" #存檔資料夾
                save_dir="./save/${dataset}${run}" #存相關資料
                dataset_dir="./data/${dataset}" #資料集位置
                aug_dir="./data_aug/${dataset}/${percent}/${line}" #擴增資料集位置
                python train.py -g -bs=16 -ne=100 -dd=$dataset_dir -lod=$log_dir -sd=$save_dir -ws=$ws -ln=$line -afp=$aug_dir
                run=`expr $run + 1`
            done
            ws=`expr $ws + 1`
        done    
    done
done

#無資料擴增
#python train.py -g -bs=16 -ne=100 -dd=./data/MixATIS_TP -lod=./log/MixATIS_TP -sd=./save/MixATIS_TP -ws=3
#資料擴增
#python train.py -g -bs=16 -ne=100 -dd=./data/MixATIS_TP -lod=./log/MixATIS_TP/1 -sd=./save/MixATIS_TP -ws=3 -ln=sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt -afp=./data_aug/MixATIS_TP/1/sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt
