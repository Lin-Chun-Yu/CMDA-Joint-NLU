#獲取w1+w2+w3+w4+w5+w6+w7=1的參數組合的檔案名稱
from data_augmentation.method import *

methods = ["sr", "ri", "rs", "rd", "cr", "ro", "sl"]

def main():
    all_combinations = generate_combinations()#產生所有組合

    for combination in all_combinations:
        data_name=''
        for i, method in enumerate(methods):
            data_name = data_name + method+f'{str(combination[i])}'  
        
        save_data = f'{data_name}.txt'#存檔位置   
        print(save_data)

if __name__ == "__main__":
    main()
