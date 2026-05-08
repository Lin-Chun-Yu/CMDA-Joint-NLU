import os
import random
from data_augmentation.eda import *
from data_augmentation.SP import conllud
from data_augmentation.SP import augmenter
from data_augmentation.slot_sub import *
import itertools
# 設定隨機種子
random.seed(0)

loi = [u"nsubj", u"dobj", u"iobj", u"obj", u"obl"]
pl = u"root"
multilabs = [u"case", u"fixed", u"flat", u"cop", u"compound"]

def generate_combinations(total=1, parts=7, step=0.1):#生成所有組合
    possible_values = [round(i * step, 1) for i in range(1, int(total / step))]

    # 生成所有可能的组合
    all_combinations = itertools.product(possible_values, repeat=parts)

    # 过滤出总和为1的组合
    valid_combinations = [comb for comb in all_combinations if round(sum(comb), 1) == total]

    # 去重并转换为列表
    unique_combinations = list(set(valid_combinations))
    return unique_combinations

def write_file(file_path, texts, slots, intents):#寫檔
    print('存檔位置:', file_path)   
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as file:# train 、 dev 、 test 。
        for i in range(len(texts)):
            tagged_text = '\n'.join([f"{word} {slot[0]} {slot[1]}" for word, slot in zip(texts[i], slots[i])])
            file.write(tagged_text)
            file.write("\n")

            if intents:# 寫入意圖標籤
                file.write("#".join(intents[i]))  # 使用 # 符號連接多个意圖標籤
                file.write("\n\n")  # 在意图标签之后放置两个换行符，用於分隔下一个句子的標籤
            else:
                file.write("\n") 

def read_tp_file(file_path):#讀檔
    texts, slots, tp_intents, intents = [], [], [], []
    text, slot, tp_intent = [], [], []

    with open(file_path, 'r', encoding="utf8") as fr:
        for line in fr.readlines():     #讀取每一行
            items = line.strip().split()#使用空格分開。

            if len(items) == 1: # 找到意圖標籤 ex: atis_abbreviation#atis_airport#atis_city
                texts.append(text)
                slots.append(slot)
                tp_intents.append(tp_intent)
                #print(items[0])
                #input()
                if "/" not in items[0]:#好像就只是放意圖列表用的
                    intents.append(items)
                else:
                    new = items[0].split("/")
                    intents.append([new[1]])
                text, slot ,tp_intent = [], [], []# 清空 buffer lists，因為要重新裝文字和槽標籤
            elif len(items) == 2: # 文字 和 槽位標籤
                text.append(items[0].strip())
                slot.append(items[1].strip())
                tp_intent.append('')                    
            elif len(items) == 3:#文字 和 槽位標籤 和 TP意圖標籤 ex:['and', 'O', 'TP']
                text.append(items[0].strip())
                slot.append(items[1].strip())
                tp_intent.append(items[2].strip())
            #input()
    return texts, slots, intents, tp_intents

def rotate_augment(texts, slots, intents, method_num):
    fields, fields_order= conllud.convert_to_conllu(texts)#使用 conllud.convert_to_conllu 函數將文本轉換為 CoNLL-U 格式。
    ud_reader = conllud.conllUD(fields)#使用 conllud.conllUD 讀取 CoNLL-U 格式的文本，並獲取句子列表。
    ud_sents = ud_reader.sents

    new_text, new_slot, new_intent = [], [], []
    for i, s in enumerate(ud_sents):#遍歷每個句子，根據方法選擇旋轉或裁剪增強，並處理相應的槽位。
        rotator = augmenter.rotator(s, aloi=loi, pl=pl, multilabs=multilabs)
        aug_sents, aug_slot = rotator.rotate(fields_order[i], texts[i], slots[i])
        if len(aug_sents)>0:
            for j, sentences in enumerate(aug_sents):
                new_text.append(aug_sents[j])
                new_slot.append(aug_slot[j])
                new_intent.append(intents[i])

    # 隨機選取 method_num 個句子、槽位和意圖
    combined = list(zip(new_text, new_slot, new_intent))
    random.shuffle(combined)
    selected = combined[:method_num]
    selected_texts, selected_slots, selected_intents = zip(*selected) if selected else ([], [], [])

    return list(selected_texts), list(selected_slots), list(selected_intents)

def crop_augment(texts, slots, intents, method_num):
    fields, fields_order= conllud.convert_to_conllu(texts)#使用 conllud.convert_to_conllu 函數將文本轉換為 CoNLL-U 格式。
    ud_reader = conllud.conllUD(fields)#使用 conllud.conllUD 讀取 CoNLL-U 格式的文本，並獲取句子列表。
    ud_sents = ud_reader.sents

    new_text, new_slot, new_intent = [], [], []
    for i, s in enumerate(ud_sents):#遍歷每個句子，根據方法選擇旋轉或裁剪增強，並處理相應的槽位。
        cropper = augmenter.cropper(s, aloi=loi, pl=pl, multilabs=multilabs)
        aug_sents, aug_slot = cropper.crop(fields_order[i], texts[i], slots[i])
        if len(aug_sents)>0:
            for j, sentences in enumerate(aug_sents):
                new_text.append(aug_sents[j])
                new_slot.append(aug_slot[j])
                new_intent.append(intents[i])

    # 隨機選取 method_num 個句子、槽位和意圖
    combined = list(zip(new_text, new_slot, new_intent))
    random.shuffle(combined)
    selected = combined[:method_num]
    selected_texts, selected_slots, selected_intents = zip(*selected) if selected else ([], [], [])

    return list(selected_texts), list(selected_slots), list(selected_intents)

def slot_sub_augment(texts, slots, slot_tmp, intents, method_num):
    dictionary = slot_dictionary(texts, slot_tmp)
    new_text, new_slot, new_intent = [], [], []

    # 將 text, slot, intent 打包在一起並過濾掉只有 O 槽位標籤的句子
    combined = [(text, slot, intent) for text, slot, intent in zip(texts, slots, intents) if any(slot_lab != 'O' for slot_lab in slot)]
    random.shuffle(combined)
    selected = combined[:method_num]

    for text, slot, intent in selected:
        slot_label = {slot_lab[2:] for slot_lab in slot if slot_lab != 'O'}

        replace_text, replace_slot = perform_slot_substitution(text, slot, slot_label, dictionary)
        new_text.append(replace_text)
        new_slot.append(replace_slot)
        new_intent.append(intent)

    return new_text, new_slot, new_intent

def perform_slot_substitution(text, slot, slot_label, dictionary):
    sub_labels = random.choices(list(slot_label), k=1)#隨機選一個槽值進行替換
    replace_text, replace_slot = list(text), list(slot)
    for label in sub_labels:
        slot_values = list(dictionary.get(label, []))
        if slot_values:
            chosen_value = random.choice(slot_values).split()
            start_index = replace_slot.index('B-' + label)
            end_index = start_index + 1
            while end_index < len(replace_slot) and replace_slot[end_index].startswith('I-'):
                end_index += 1
            replace_text = replace_text[:start_index] + chosen_value + replace_text[end_index:]
            replace_slot = replace_slot[:start_index] + ['B-' + label] + ['I-' + label] * (len(chosen_value) - 1) + replace_slot[end_index:]
    return replace_text, replace_slot