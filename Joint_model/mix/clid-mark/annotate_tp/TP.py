#這裡是用來將資料集將連接詞標註TP標籤。
#先執行annotate.py
#再執行TP.py
import os
import glob

#標註連接詞為TP
def annotate_tp(data):
    annotated_data = []
    for line in data:
        tokens = line.split()
        if len(tokens) == 2:
            word, tag = tokens
            annotated_data.append((word, tag))
        elif len(tokens) == 1 and "#" in tokens[0]:#如果是多意圖對話。
            annotated_data.append((tokens[0] + "#TP",))  # adding TP tag
        elif len(tokens) == 1:#如果是單一意圖對話。
            annotated_data.append((tokens[0],))
    return annotated_data



#讀取資料
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

#存取資料
def save_data(file_path, annotate_tp, annotated_data):
    i=0
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in annotated_data:
            annotate_tp[i]=annotate_tp[i].strip()
            if len(item) == 2:
                word, tag = item
                if annotate_tp[i] =="TP":
                    file.write(f"{word} {tag} {annotate_tp[i]}\n")
                else:
                    file.write(f"{word} {tag}\n")
                i+=1
            elif len(item) == 1:
                # If only one value, write it without a tag and add an extra newline character
                file.write(f"{item[0]}\n\n")
                i+=2
            else:
                print(f"Skipping invalid item: {item}")


# 路徑設置
file_path = "/home/cyl22/work/joint_model/mix/clid-mark/annotate_tp/MixSNIPS_clean/train.txt"
annotate_tp_path = "/home/cyl22/work/joint_model/mix/clid-mark/annotate_tp/MixSNIPS_clean_tp.txt"
output_path = "/home/cyl22/work/joint_model/mix/clid-mark/annotate_tp/train.txt"

annotate_tp_data = []

data = load_data(file_path)
annotate_tp_data = load_data(annotate_tp_path)
annotated_data = annotate_tp(data)#標註多意圖TP用得

save_data(output_path, annotate_tp_data, annotated_data)

