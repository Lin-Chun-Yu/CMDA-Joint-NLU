# CMDA-Joint-NLU
1. CM Model Execution
MixATIS (Directory: joint_model/mix/cmda)
python train.py -g -bs=16 -ne=100 -dd=./data/MixATIS -lod=./log/MixATIS -sd=./save/MixATIS -ws=3
python train.py --gpu --batch_size=16 --num_epoch=100 --data_dir=./data/MixATIS --log_dir=./log/MixATIS --save_dir=./save/MixATIS --window_size=3

MixSNIPS (Directory: joint_model/mix/cmda)
python train.py -g -bs=64 -ne=50 -dd=./data/MixSNIPS -lod=./log/MixSNIPS -sd=./save/MixSNIPS -ws=3
python train.py --gpu --batch_size=64 --num_epoch=50 --data_dir=./data/MixSNIPS --log_dir=./log/MixSNIPS --save_dir=./save/MixSNIPS --window_size=3

Hotels (Directory: joint_model/nlupp/cmda)
python train.py -g -bs=4 -ne=100 -dd=./data/hotels0 -lod=./log/hotels -sd=./save/hotels -ws=3
python train.py --gpu --batch_size=4 --num_epoch=100 --data_dir=./data/hotels0 --log_dir=./log/hotels --save_dir=./save/hotels --window_size=3

Banking (Directory: joint_model/nlupp/cmda)
python train.py -g -bs=4 -ne=100 -dd=./data/banking0 -lod=./log/banking -sd=./save/banking -ws=3
python train.py --gpu --batch_size=4 --num_epoch=100 --data_dir=./data/banking0 --log_dir=./log/banking --save_dir=./save/banking --window_size=3

#############################################################################
2. CMDA Model Execution (With Data Augmentation)
MixATIS:
python train.py -g -bs=16 -ne=100 -dd=./data/MixATIS -lod=./log/MixATIS/1 -sd=./save/MixATIS -ws=3 -ln=sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt -afp=./data_aug/MixATIS/1/sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt

python train.py --gpu --batch_size=32 --num_epoch=1 --data_dir=./data/MixATIS --log_dir=./log/MixATIS/1 --save_dir=./save/MixATIS --window_size=3 --log_name=sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt --aug_file_path=./data_aug/MixATIS/1/sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt


MixSNIPS:
python train.py -g -bs=64 -ne=50 -dd=./data/MixSNIPS -lod=./log/MixSNIPS/1 -sd=./save/MixSNIPS -ws=3 -ln=sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt -afp=./data_aug/MixSNIPS/1/sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt
python train.py --gpu --batch_size=64 --num_epoch=50 --data_dir=./data/MixSNIPS --log_dir=./log/MixSNIPS/1 --save_dir=./save/MixSNIPS --window_size=3 --log_name=sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt --aug_file_path=./data_aug/MixSNIPS/1/sr0.1ri0.1rs0.1rd0.1cr0.1ro0.1sl0.4.txt

#############################################################################
3. Automation Scripts
Automated shell scripts are available in joint_model/mix/cmda and joint_model/nlupp/cmda:
Generate all augmentation parameter combinations: bash data_aug.sh
bash data_aug.sh

Run MixATIS pipeline
bash mixatis.sh

Run MixSNIPS pipeline
bash mixsnips.sh

Run Hotels pipeline
bash hotels.sh

Run Banking pipeline
bash banking.sh
##############################################################################
4. Utility Functions
Intent Statistics: Count multi-intents in Training, Validation, and Test sets.
python intent_num.py 

Data Partitioning: Split Hotels and Banking datasets into 10 folds.
python make_data.py

Results Averaging: Calculate average performance from logs.
python avg.py  

################################################################################
5. Virtual Environment Commands (Python 3.11.7)

Create environment
py -3.11 -m venv miranda_aurora

Activate environment
.\miranda_aurora\Scripts\activate

Deactivate environment
deactivate

Remove environment
Remove-Item -Recurse -Force miranda_aurora
