def slot_dictionary(texts, slots):#建立查找槽值的字典
    
    dic = {}
    for i in range(len(texts)):
        current_slot = None
        for j, slot in enumerate(slots[i]):
            #print(slot)
            if slot.startswith('B-'):  # 如果是槽值的開始
                current_slot = slot[2:]  # 取出槽標籤
                if current_slot not in dic:#沒有出現在字典裡的話
                    dic[current_slot] = set()
                current_value = texts[i][j]  # 初始化槽值字串

                if j == len(slots[i])-1:#如果是最後一個槽標籤
                    dic[current_slot].add(current_value)
                    current_slot = None
            elif slot.startswith('I-'):  # 如果是槽值的中間部分
                current_slot = slot[2:]
                if current_slot is not None:
                    current_value += " " + texts[i][j]  # 將槽值連接成字串
                
                if j == len(slots[i])-1:#如果是最後一個槽標籤
                    dic[current_slot].add(current_value)
                    current_slot = None
            else:  # 如果是'O'
                if current_slot is not None:
                    dic[current_slot].add(current_value)  # 將槽值添加到字典
                    current_slot = None

    return dic