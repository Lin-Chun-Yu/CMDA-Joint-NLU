#原eda作者實作的eda
import random
from random import shuffle
random.seed(0)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet 

def synonym_replacement(words, n):#改成閾值的方式來決定同義詞替換。
	new_words = words.copy()
	random_word_list = []
	for word in words:
		if not(word in stop_words):
			random_word_list.append(word)    
	random_word_list=list(dict.fromkeys(random_word_list))
	random.shuffle(random_word_list)#將字給打亂

	num_replaced = 0  # 計數已替換單詞的數量
	for random_word in random_word_list:  # 遍歷可替換單詞列表
		synonyms = get_synonyms(random_word)  # 獲取該單詞的同義詞列表
		if len(synonyms) >= 1:  # 如果同義詞列表不為空
			synonym = get_matching_synonym(random_word, synonyms)  # 從同義詞列表中選擇一個匹配的同義詞
			if synonym:  # 如果找到匹配的同義詞
				new_words = [synonym if word == random_word else word for word in new_words]  # 用同義詞替換原始單詞
				num_replaced += 1  # 已替換單詞的計數加1
		if num_replaced >= n:  # 如果已替換單詞的數量達到n
			break  # 結束替換過程

	return new_words  # 返回替換後的新單詞列表

from collections import OrderedDict
def get_synonyms(word):#獲得同義詞
	synonyms = []
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()#將同義詞給取出來
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			if len(synonym.split()) == 1:  # 只保留單詞數量為1的同義詞
				synonyms.append((synonym, syn.pos())) # 保存同義詞和詞性
	synonyms = list(OrderedDict.fromkeys(synonyms))	#去除重複的同義詞
	synonyms = [syn for syn in synonyms if syn[0] != word]#從同義詞列表中移除與原詞相同的詞
	
	return synonyms

# 定義匹配詞性的同義詞替換函數
def get_matching_synonym(word, synonyms):
    word_pos = None
    for syn in wordnet.synsets(word):#需要原詞的詞性
        word_pos = syn.pos()
        break  

    for synonym, syn_pos in synonyms:
        if word_pos == syn_pos:
            return synonym
    return None

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, slots, p):

	if len(words) == 1:#只有一個單詞就不刪
		return words, slots

	new_words, new_slots = [], []
	for i, word in enumerate(words):
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)
			new_slots.append(slots[i])

	if len(new_words) == 0:	#如果刪除了所有的單詞，隨機返回一個單詞
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]], [slots[rand_int]]

	return new_words, new_slots

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, slots, n):
	new_words = words.copy()
	new_slots = slots.copy()

	num_words = len(new_words)
	if num_words < 2:# 不足兩個詞無法進行交換
		return new_words, new_slots  
		
	for _ in range(n):
		idx1, idx2 = random.sample(range(num_words), 2)
		# 交換單詞和槽位
		new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
		new_slots[idx1], new_slots[idx2] = new_slots[idx2], new_slots[idx1]

	return new_words, new_slots

########################################################################
# Random addition
# 隨機插入會去除停用詞、使用閾值來決定是否要將同義詞插入、同義詞也會考慮詞性。
########################################################################

def random_insertion(words, slots, n):
	new_words = words.copy()
	new_slots = slots.copy()    
	random_word_slot_list = []
	for word, slot in zip(words, slots):
		if word not in stop_words:
			random_word_slot_list.append((word, slot))
    
	random_word_slot_list = list(dict.fromkeys(random_word_slot_list))
	random.shuffle(random_word_slot_list)  # 將字打亂

	num_replaced = 0#要插入同義詞的數量
	for random_word, random_slot in random_word_slot_list:
		synonyms = get_synonyms(random_word)#找同義詞
		if len(synonyms) >= 1: #如果有找到同義詞，就隨機插入
			synonym = get_matching_synonym(random_word, synonyms)
			if synonym:		
				rand_index = random.randint(0, len(new_words) - 1)#隨機選一個位置
				random_synonym = synonym
				new_words.insert(rand_index, random_synonym)
				new_slots.insert(rand_index, random_slot)
				num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	return new_words, new_slots

########################################################################
# main data augmentation function
########################################################################
def synonym_replacement_augment(texts, slots, intents, num_aug, alpha_sr):#同義詞替換
    new_text, new_slot, new_intent = [], [], []
    for _ in range(num_aug):

        index = random.randint(0, len(texts) - 1) #隨機選一個句子
        num_words = len(texts[index])

        n_sr = int(alpha_sr * num_words)
        random_sr = random.randint(1, max(1, n_sr))# 根據 n_sr 隨機選擇一個數字，決定增強次數

        a_words = synonym_replacement(texts[index], random_sr)
        new_text.append(a_words)
        new_slot.append(slots[index])
        new_intent.append(intents[index])
    
    return new_text, new_slot, new_intent

def random_insertion_augment(texts, slots, intents, num_aug, alpha_ri):#隨機插入
    new_text, new_slot, new_intent = [], [], []
    for _ in range(num_aug):
        index = random.randint(0, len(texts) - 1) #隨機選一個句子
        num_words = len(texts[index])

        n_sr = int(alpha_ri * num_words)
        random_ri = random.randint(1, max(1, n_sr))# 根據 n_sr 隨機選擇一個數字，決定增強次數

        a_words, a_slots = random_insertion(texts[index], slots[index], random_ri)
        new_text.append(a_words)
        new_slot.append(a_slots)
        new_intent.append(intents[index])
    
    return new_text, new_slot, new_intent

def random_swap_augment(texts, slots, intents, num_aug, alpha_rs):#隨機交換
    new_text, new_slot, new_intent = [], [], []
    for _ in range(num_aug):	

        index = random.randint(0, len(texts) - 1) #隨機選一個句子
        num_words = len(texts[index])

        n_sr = int(alpha_rs * num_words)
        random_rs = random.randint(1, max(1, n_sr))# 根據 n_sr 隨機選擇一個數字，決定增強次數

        a_words, a_slots = random_swap(texts[index], slots[index], random_rs)
        new_text.append(a_words)
        new_slot.append(a_slots)
        new_intent.append(intents[index])
    
    return new_text, new_slot, new_intent

def random_deletion_augment(texts, slots, intents, num_aug, alpha_rd):#隨機刪除
    new_text, new_slot, new_intent = [], [], []
    for _ in range(num_aug):

        index = random.randint(0, len(texts) - 1) #隨機選一個句子

        a_words, a_slots = random_deletion(texts[index], slots[index], alpha_rd)
        new_text.append(a_words)
        new_slot.append(a_slots)
        new_intent.append(intents[index])
    
    return new_text, new_slot, new_intent