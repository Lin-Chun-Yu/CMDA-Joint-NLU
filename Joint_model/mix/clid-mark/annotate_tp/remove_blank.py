#snips資料集有多餘的空白，這裡會將句子的用單空格去將每一個自隔開。
#python remove_blank.py
import os
import re

def process_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".in"):
                file_path = os.path.join(root, file)
                process_file(file_path)

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    processed_lines = []
    for line in lines:
        # 使用正则表达式替换每行中多个连续空格为单个空格
        processed_line = re.sub(r' +', ' ', line)
        processed_lines.append(processed_line)
    
    processed_content = ''.join(processed_lines)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(processed_content)

# 将下面的 'your_directory_path' 替换为你的目录路径
directory_path = '/home/cyl22/work/clid/annotate_tp/SNIPS'
process_directory(directory_path)
