#將nlupp分成10個檔案
import os
import shutil
import json
import os
import re

def process_data(input_data, output_data):
    data = {}
    with open(os.path.join(input_data)) as f:
        data = json.load(f)

    for sen in data:#跑每一個fold的句子。
        words = sen['text'].lower()#大寫轉小寫
        words = words.split(" ")

        intents = []
        if 'intents' in sen:# 檢查是否有意圖標籤，把沒有意圖標籤的句子拿掉
            intents.append('#'.join(sen['intents']))

            if 'slots' in sen:  # 檢查是否有槽標籤
                #print(words)
                slots = sen['slots']  
                tags = ['O'] * len(words)# 初始化標籤列表
                for slot_name, slot_info in slots.items():# 將標籤與槽位相對應
                    start, end = slot_info['span']
                    text = slot_info['text']
                    #print(text)
                    #print(slot_name, start, end)
                    #input()
                    count = 0
                    for i, word in enumerate(words):
                        if count == start:
                            tags[i] = 'B-' + slot_name
                        elif count > start and count < end:
                            tags[i] = 'I-' + slot_name
                        count += len(word) + 1  # 更新字符數量，包含空格
                #print(len(tags), tags)
                #input()
            else:
                tags = ['O'] * len(words)# 初始化標籤列表


            #print(words)
            #print(tags)
            new_words, new_tags = [], []
            for word, tag in zip(words, tags):#這裡是為了檢察標點符號
                #print(len(word))
                #input()
                punctuations = ['?', ',', '.', '!']
                for punctuation in punctuations:
                    if punctuation in word:
                        if len(word) != 1 and word != 'a.m.' and word != 'p.m.':
                            word = word.replace(punctuation, '')
                            new_words.append(word)
                            new_tags.append(tag)
                            new_words.append(punctuation)
                            new_tags.append('O')
                            break  # 找到標點符號後停止搜索其他標點符號
                else:  # 如果沒有找到任何標點符號
                    new_words.append(word)
                    new_tags.append(tag)
            #print(new_words)
            #input(new_tags)

            # 將文本分詞，並建構帶有標籤的文本
            tagged_text = '\n'.join([f"{word} {tag}" for word, tag in zip(new_words, new_tags)])

            # 寫檔
            with open(output_data, 'a') as file:# train 、 dev 、 test 。
                file.write(tagged_text)
                file.write("\n")

                if intents:# 寫入意圖標籤
                    file.write("#".join(intents))  # 使用 # 符號連接多个意圖標籤
                    file.write("\n\n")  # 在意图标签之后放置两个换行符，用於分隔下一个句子的標籤
                else:
                    file.write("\n") 

def create_folds(domain, input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')], key=lambda x: int(x.split('fold')[1].split('.json')[0]))

    for i in range(len(files)):
        # 創建新的資料夾結構
        fold_folder = os.path.join(output_folder, f'{domain}{i}')
        print(fold_folder)
        if not os.path.exists(fold_folder):
            os.makedirs(fold_folder)

        dev_folder = os.path.join(fold_folder, 'dev.txt')
        test_folder = os.path.join(fold_folder, 'test.txt')
        train_folder = os.path.join(fold_folder, 'train.txt')

        for j, file in enumerate(files):
            src_file = os.path.join(input_folder, file)

            if i == j:                              #驗證集
                process_data(src_file, dev_folder)
                #shutil.copy(src_file, os.path.join(dev_folder, file))
            elif (i + 1) % len(files) == j:          #測試集           
                process_data(src_file, test_folder)
                #shutil.copy(src_file, os.path.join(test_folder, file))
            else:                                    #訓練集               
                process_data(src_file, train_folder)
                #shutil.copy(src_file, os.path.join(train_folder, file))

BANKING, HOTELS = ("banking", "hotels")   # Domains
for domain in [BANKING, HOTELS]:
    # 輸入資料夾和輸出資料夾
    output_folder = 'data/'    
    input_folder = os.path.join(output_folder, domain)

    create_folds(domain, input_folder, output_folder)


