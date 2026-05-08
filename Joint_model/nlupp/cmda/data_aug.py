import os
import random
import argparse

# 將隨機種子設定為0
random.seed(0)
from data_augmentation.method import *

parser = argparse.ArgumentParser()
parser.add_argument('--percent_of_dataset', '-pod', type=float, default=0.02)#資料集擴增百分比

parser.add_argument('--alpha_synonym_replacement', '-alpha_sr', type=float, default=0.1)
parser.add_argument('--alpha_random_insertion', '-alpha_ri', type=float, default=0.1)
parser.add_argument('--alpha_random_swap', '-alpha_rs', type=float, default=0.1)
parser.add_argument('--alpha_random_deletion', '-alpha_rd', type=float, default=0.1)
args = parser.parse_args()

methods = ["sr","ri","rs","rd","cr", "ro","sl"]

#先讀入
if __name__ == "__main__":
    all_combinations = generate_combinations()#產生所有組合

    BANKING, HOTELS = ("banking", "hotels")   # Domains
    for domain in [HOTELS]:
        for fold in range(10):#10個fold
            for combination in all_combinations:#全部組合
                dataset = domain + str(fold)  
                file_path = os.path.join("data", dataset, 'train.txt')#讀檔位置    

                texts, slots, intents = read_file(file_path)

                augment_num = round(args.percent_of_dataset * len(texts))#要擴增的句子數    

                print(dataset, "原本:", len(texts), "句, 擴增:", augment_num)

                aug_texts, aug_slots, aug_intents = [], [], []
                augment_count = 0 #計算擴增數量。
                data_name=''
                for i, method in enumerate(methods):

                    if i == len(methods) - 1: # 判斷是否為最後一個迴圈
                        method_num = augment_num - augment_count
                        if method_num<0:
                            method_num=0
                    else:
                        method_num = round(len(texts) * (combination[i] * args.percent_of_dataset))
                        if augment_count + method_num > augment_num:#超過，就不擴增。
                            method_num = augment_num - augment_count
                        else:
                            augment_count += method_num                           

                    if method_num > 0:
                        if method == 'sr': #同義詞替換
                            new_text, new_slots, new_intents = synonym_replacement_augment(texts, slots, intents, method_num, alpha_sr=args.alpha_synonym_replacement)
                        elif method == 'ri': #隨機插入
                            new_text, new_slots, new_intents = random_insertion_augment(texts, slots, intents, method_num, alpha_ri=args.alpha_random_insertion)
                        elif method == 'rs': #隨機交換
                            new_text, new_slots, new_intents = random_swap_augment(texts, slots, intents, method_num, alpha_rs=args.alpha_random_swap)
                        elif method == 'rd': #隨機刪除
                            new_text, new_slots, new_intents = random_deletion_augment(texts, slots, intents, method_num, alpha_rd=args.alpha_random_deletion)
                        elif method == 'cr': #裁剪
                            new_text, new_slots, new_intents = rotate_augment(texts, slots, intents, method_num)
                        elif method == 'ro': #旋轉
                            new_text, new_slots, new_intents = crop_augment(texts, slots, intents, method_num)
                        elif method == 'sl': #槽位替換
                            new_text, new_slots, new_intents = slot_sub_augment(texts, slots, intents, method_num)
                    
                        aug_texts.extend(new_text)
                        aug_slots.extend(new_slots)
                        aug_intents.extend(new_intents)

                    print(method, "增強:", method_num, "句")
                    data_name = data_name + method+f'{str(combination[i])}'        
                print("總增強:", len(aug_texts), "句")

                percent_str = str(int(args.percent_of_dataset * 100))
                save_path = os.path.join("data_aug", dataset, percent_str, f'{data_name}.txt')#存檔位置   
                write_file(os.path.join(save_path), aug_texts, aug_slots, aug_intents)

#python data_aug.py -dataset=hotels -method=sr --n_aug=1 -thr=0.1 