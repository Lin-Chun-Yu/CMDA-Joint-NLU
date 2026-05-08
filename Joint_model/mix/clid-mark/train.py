# -*- coding: utf-8 -*-#
#查看gpu指令watch nvidia-smi
import os, json, random
import numpy as np
import torch
from models.module import ModelManager  #自定義
from utils.loader import DatasetManager #自定義
from utils.process import Processor     #自定義
from utils.config import *              #自定義

if __name__ == "__main__":

    #創建保存資料夾：代碼檢查是否存在一个指定的資料夾（args.save_dir），如果不存在，則使用系统命令創建該資料夾。
    #這是用于儲存模型和訓練参数的資料夾。
    if not os.path.exists(args.save_dir):
        os.system("mkdir -p " + args.save_dir)

    #保存參數到JSON檔案：程式碼將參數對象args中的參數保存為一個JSON文件（param.json），
    #這個文件位於save資料夾中。這樣做可以記錄訓練和評估時所使用的參數配置。
    log_path = os.path.join(args.save_dir, "param.json")
    with open(log_path, "w", encoding="utf8") as fw:
        fw.write(json.dumps(args.__dict__, indent=True))

    #固定隨機種子：代碼固定了隨機種子，以确保訓練和評估的结果在不同運行之间是可重現的。
    #這包括了Python的隨机種子和PyTorch庫在GPU和CPU上的隨機種子。
    '''
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Fix the random seed of Pytorch when using GPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Fix the random seed of Pytorch when using CPU.
    torch.manual_seed(args.random_state)
    torch.random.manual_seed(args.random_state)
    '''

    #實例化數據集對象：使用DatasetManager類實例化了一个數據集對象（dataset），
    #並進行了一些數據的快速建構和匯總。
    dataset = DatasetManager(args)
    dataset.quick_build()#讀取資料集，並做一些處理。
    #exit()
    dataset.show_summary()

    #實例化模型对象：使用ModelManager類實例化了一個網路模型對象（model），
    #並提供了輸入參數，包括詞彙表大小和意圖/槽位的數量。
    #print(len(dataset.intent_alphabet))
    #exit()
    model = ModelManager(
        args, len(dataset.word_alphabet),
        len(dataset.slot_alphabet),
        len(dataset.intent_alphabet),
        dataset.tp_index
    )
    model.show_summary()

    #訓練和評估模型:
    #使用Processor類實例化了一个處理器對象（process），並通過其train方法来訓練模型。在訓練過程中，最佳模型在每個epochs都会被保存
    #使用Processor類的validate方法評估了訓練後的模型，並印出了在測試資料集上的性能指標。
    process = Processor(dataset, model, args)
    best_epoch = process.train()#這裡跑完就會得到最好的結果
    results = Processor.validate(
        os.path.join(args.save_dir, "model/model.pkl"),
        dataset,
        args.batch_size, len(dataset.intent_alphabet), args=args)
    print('\nAccepted performance: ' + str(results) + " at test dataset;\n")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, args.log_name), 'a') as fw:
        fw.write(str(best_epoch))
        for result in results:
            fw.write(',' + str(result))
        fw.write("\n")
