#計算效能的平均
import os

def calculate_average(file_path):
    # 初始化計數器和總和
    total_sum = [0, 0, 0, 0]
    count = 0
    # 打開檔案並逐行讀取
    with open(file_path, 'r') as file:
        for line in file:
            count+=1
            # 將數據解析為浮點數
            data = line.strip().split(',')
            for i, val in enumerate(data[1:]):
                total_sum[i] += float(val)

    print(f"{file_path} 除以{count}取平均")
    for i, sum in enumerate(total_sum):
        total_sum[i] = sum / count
    
    return total_sum

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
            #print(f"- {root}/{file}")
            total_sum = calculate_average(os.path.join(root, file))
            print("slot(f1)：", total_sum[0], "intent(f1):", total_sum[1], " intent(acc):", total_sum[2], "overall(acc):", total_sum[3])

# 指定要遍歷的目錄路徑
MixATIS, MixSNIPS, = ("MixATIS", "MixSNIPS")  # Domains
for domain in [MixATIS, MixSNIPS]:
    read_files_in_folders(domain)
