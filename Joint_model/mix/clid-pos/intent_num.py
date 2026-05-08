#計算 MixATIS 和 MixSNIPS 資料集的多意圖數量
import os
def __read_file(file_path):
    texts, slots, intents = [], [], []
    text, slot = [], []

    with open(file_path, 'r', encoding="utf8") as fr:
        for line in fr.readlines():
            items = line.strip().split()

            if len(items) == 1:
                texts.append(text)
                slots.append(slot)
                if "/" not in items[0]:
                    intents.append(items)
                else:
                    new = items[0].split("/")
                    intents.append([new[1]])

                # clear buffer lists.
                text, slot = [], []

            elif len(items) == 2:
                text.append(items[0].strip())
                slot.append(items[1].strip())

    return texts, slots, intents

from collections import Counter
def intent_number(intents):
    intent_counts = [len(sentence[0].split('#')) for sentence in intents]

    intent_distribution = Counter(intent_counts)

    for intent_label_count, sentence_count in intent_distribution.items():
        print(f"意圖標籤數 {intent_label_count} 有 {sentence_count} 個語句")

    # 計算所有語句數的總和
    total_sentences = sum(intent_distribution.values())
    print(f"總語句數: {total_sentences}")
    
    return intents


def read_files_in_folders(folder_path):
    # 確保指定的路徑存在並是一個資料夾
    if not os.path.isdir(folder_path):
        print("指定的路徑不是一個資料夾。")
        return
    
    # 遍歷資料夾中的每個檔案和子資料夾
    for root, dirs, files in os.walk(folder_path):
        print(f"在 {root} 中的檔案:")
        # 列印每個檔案的名稱
        for file in files:
            print(f"- {root}/{file}")
            texts, slots, intents = __read_file(os.path.join(root, file))
            intent_number(intents)

MixATIS, MixSNIPS = ("MixATIS", "MixSNIPS")  # Domains
for domain in [MixATIS, MixSNIPS]:
    dataset = os.path.join("data", domain)
    read_files_in_folders(dataset)