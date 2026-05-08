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

def synonym_replacement(words, slots, n):
	new_words = words.copy()
	new_slots = slots.copy()  # 創建 slots 的副本
	#random_word_list = list(set([word for word in words if word not in stop_words]))#使用set會將自給不斷的隨機交換，無法固定。
	random_word_list = []
	for word in words:
		if not(word in stop_words):
			random_word_list.append(word)    
	random_word_list=list(dict.fromkeys(random_word_list))
	random.shuffle(random_word_list)#將字給打亂

	num_replaced = 0#要取代字的數量
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			for i, word in enumerate(words):
				if word == random_word:
					new_slots[i] = slots[new_words.index(synonym)]
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break
	
	'''
	final_words = []
	final_slots = []
	for word, slot in zip(new_words, new_slots):
		final_words.append(word)
		final_slots.append(slot)
		if ' ' in word:
			sub_words = word.split(' ')
			sub_slots = [slot] * len(sub_words)
			final_words.extend(sub_words)
			final_slots.extend(sub_slots)
	'''
	return new_words, new_slots

from collections import OrderedDict
def get_synonyms(word):#獲得同義詞
	synonyms = list()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()#將同義詞給取出來
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			if len(synonym.split()) == 1:  # 只保留單詞數量為1的同義詞
				synonyms.append(synonym)
	synonyms = list(OrderedDict.fromkeys(synonyms))	#去除重複的同義詞
	if word in synonyms:#去掉原來的詞，如果有在裡面的話。
		synonyms.remove(word)
	
	return synonyms

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, slots, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words, slots

	#randomly delete words with probability p
	new_words, new_slots = [], []
	for i, word in enumerate(words):
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)
			new_slots.append(slots[i])

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
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
	for _ in range(n):
		new_words, new_slots = swap_word(new_words, new_slots)
	return new_words, new_slots

def swap_word(new_words, new_slots):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words, new_slots
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	new_slots[random_idx_1], new_slots[random_idx_2] = new_slots[random_idx_2], new_slots[random_idx_1]#進行交換
	return new_words, new_slots

########################################################################
# Random addition
# Randomly add n words into the sentence
########################################################################

def random_addition(words, slots, n):
	new_words = words.copy()
	new_slots = slots.copy()    
	for _ in range(n):
		new_words, new_slots = add_word(new_words, new_slots)
	return new_words, new_slots

def add_word(new_words, new_slots):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		rand_index = random.randint(0, len(new_words)-1)
		random_word = new_words[rand_index]
		random_slot = new_slots[rand_index]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return new_words, new_slots
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)
	new_slots.insert(random_idx, random_slot)
	return new_words, new_slots

########################################################################
# main data augmentation function
########################################################################

def eda_4(words, slots, intents, num_aug = 1, alpha_sr = 0.1, alpha_ri = 0.1, alpha_rs = 0.1, alpha_rd = 0.1):
	
	augmented_sentences, new_slot, new_intent = [],[],[]
	num_words = len(words)
	
	num_new_per_technique = int(num_aug/4)+1
	n_sr = max(1, int(alpha_sr*num_words))
	n_ri = max(1, int(alpha_ri*num_words))
	n_rs = max(1, int(alpha_rs*num_words))

	#同義詞替換
	for _ in range(num_new_per_technique):
		a_words, a_slots = synonym_replacement(words, slots, n_sr)
		#print("同義詞替換",a_words, slots)
		augmented_sentences.append(a_words)
		new_slot.append(a_slots)   

	#隨機加入
	for _ in range(num_new_per_technique):
		a_words, a_slots = random_addition(words, slots, n_ri)
		#print("隨機加入",a_words,a_slots)
		augmented_sentences.append(a_words)
		new_slot.append(a_slots) 

	#隨機交換
	for _ in range(num_new_per_technique):
		a_words, a_slots = random_swap(words, slots, n_rs)
		#print("隨機交換",a_words,a_slots)
		augmented_sentences.append(a_words)
		new_slot.append(a_slots) 

	#隨機刪除
	for _ in range(num_new_per_technique):
		a_words, a_slots = random_deletion(words, slots, alpha_rd)
		#print("隨機刪除",a_words,a_slots)
		augmented_sentences.append(a_words)
		new_slot.append(a_slots) 

	#augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	#shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
		new_slot = new_slot[:num_aug]
	else:#num_aug 的值小於 augmented_sentences 的長度時
		keep_prob = num_aug / len(augmented_sentences)

		selected_sentences, selected_slot = [], []
		for i, s in enumerate(augmented_sentences):
			if random.uniform(0, 1) < keep_prob:
				selected_sentences.append(s)
				selected_slot.append(new_slot[i])
		
		augmented_sentences = selected_sentences
		new_slot = selected_slot

	for i in range(len(augmented_sentences)):#
		new_intent.append(intents)

	return augmented_sentences, new_slot, new_intent
