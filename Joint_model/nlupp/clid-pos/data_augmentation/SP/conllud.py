# encoding: utf-8

"""
Created by Gözde Gül Şahin
20.05.2018
Read UD treebanks into a data structure suitable to cropping and flipping
"""

__author__ = 'Gözde Gül Şahin'

import codecs
import copy
import spacy#用來句子解析
from spacy import displacy
from spacy_alignments import get_alignments

def align_tokens(original_tokens, spacy_doc):
    # 獲取原始文本和 spaCy 標記的對齊
    original_text = ' '.join(original_tokens)
    spacy_tokens = [token.text for token in spacy_doc]
    
    a2b, b2a = get_alignments(original_tokens, spacy_tokens)
    
    return a2b, b2a

def convert_to_conllu(sentences):
    # 加載英文模型
    nlp = spacy.load("en_core_web_sm")

    conllu = []
    conllu_order=[]
    for sentence in sentences:
        #print(len(sentence),sentence)
        original_tokens=sentence
        sentence = ' '.join(sentence)
        #sentence='Google is a nice search engine.'
        #print(sentence)
        doc = nlp(sentence)  # 處理句子
        #displacy.serve(doc, style='dep')#你可以用瀏覽器打開來，就可以看到相依樹。
        # 獲取對齊結果
        a2b, b2a = align_tokens(original_tokens, doc)
        
        conllu_output = ""
        for i, token in enumerate(doc):
            head_index = 0 if token.dep_ == "ROOT" else token.head.i + 1 #root指向0，小寫
            deprel = "root" if token.dep_ == "ROOT" else token.dep_
            conllu_output += "\t".join([
                str(token.i + 1),       # 依存樹中的id
                token.text,             # 詞的文本
                token.lemma_,           # 詞的詞根
                token.pos_,             # 詞性標註
                token.tag_,             # 精確詞性標註
                str(token.morph),       # 詞形特徵，轉換為字符串
                str(head_index),        # 詞的依存關係中的頭詞索引，+1 因為 CoNLL-U 格式從1開始
                deprel,                 # 詞的依存關係標籤
                deprel                  # 詞的依存關係標籤（與deprel相同）
            ])
            if i != len(doc) - 1:  # 如果不是最後一個 token
                conllu_output += "\n"  # 添加換行符號
        # 將句子的 CoNLL-U 格式添加到列表中

        conllu.append(conllu_output)
        conllu_order.append(b2a)
        #print(conllu_output)
        #input()
    return conllu, conllu_order

class conllUD:

    def __init__(self, fpath=None):
        """
        初始化方法，用於讀取並存儲句子到 conllUDsent 結構中。
        """
        self.sents = []
        if fpath is not None:
            self.sents = self._read_file(fpath)
            
        else:
            print("File can not be opened, check path")

    def _read_file(self, sentences):
        """
        _read_file 用於讀取 token 並為所有的 UD 句子創建依存樹。
        """
        conllsentences = []
        sent_no = 0

        for sent in sentences:#遍歷每個句子 sent
            if(len(sent)>0):#如果句子不為空，則將句子按行分割成 lines。
                lines = sent.split("\n")
                csent = conllUDsent()#創建 conllUDsent 對象 csent。
                rows = []#初始化 rows 列表，用於存儲句子的行信息。
                tok_index = 1#初始化 tok_index，用於計算句子中的詞索引。
                for line in lines:#遍歷 lines 中的每一行 line：
                    #print(line.split("\t"))
                    ctoken = conllUDtoken(line.split("\t"),tok_index)#創建 conllUDtoken 對象 ctoken，並添加到 csent。
                    csent.add_token(ctoken)#將行分割後添加到 rows 列表中。
                    rows.append(line.split())
                    tok_index+=1#詞索引 tok_index 增加。
                csent.rows = rows

                #設置 csent 的 rows、tokenWords 和 tokenLemmas 屬性。
                csent.tokenWords = [row[1] for row in rows]#句子的詞
                csent.tokenLemmas = [row[2] for row in rows]#句子的詞的詞幹
                conllsentences.append(csent)

                #建立依存樹：遍歷 csent 的每個 token，將其添加到 deptree 中。
                for tok in csent.tokens:
                    #print(tok)
                    if tok.head in csent.deptree:
                        csent.deptree[tok.head].append(tok)
                    else:
                        csent.deptree[tok.head] = [tok]
                    #input()
        return conllsentences

class conllUDsent:#用於表示一個 CoNLL-U 句子，包含詞、詞幹、依存樹等信息。
    def __init__(self):
        self.tokens = []#存儲句子中的所有 token。
        self.tokenWords = []#存儲句子中的詞。
        self.tokenLemmas = []#存儲句子中詞的詞幹。
        self.deptree = {}#存儲依存樹，用於重新排序。
        self.rows = []#存儲句子中每個詞的資訊。

    def add_token(self, token):#放每一個詞的資訊
        self.tokens.append(token)

    def print_sent_ord(self,ord):#用於按照給定的順序 ord 印出句子。
        """
        print_sent_ord: print sentence as text with the given order
        """
        strsent = ""
        for i in ord:
            if i==0:
                continue
            tok = self.tokens[i-1]
            strsent+=(tok.word+' ')
        print(strsent)

    def reorder(self, neword, texts, slots):#用於根據給定的新順序 neword 重新排列句子的 token。
        """
        reorder: Reorder the tokens of the sentence with the given new order (neword)
        """
        new_texts = []
        new_slots = []

        # first put them together
        for i in neword:
            new_texts.append(texts[i])
            new_slots.append(slots[i])

        return new_texts, new_slots

class conllUDtoken:#用於表示一個 CoNLL-U token，包含詞的各種屬性，如詞形、詞幹、詞性標註等。
    def __init__(self, fields, tok_index):
        # index in the sentence (the order)
        self.index = tok_index
        # dependency id for the dependency tree
        self.depid = fields[0]
        self.word = fields[1]
        self.lemma = fields[2]
        self.pos = fields[3]
        self.ppos = fields[4]
        self.feat = fields[5]
        self.head = fields[6]
        self.deprel = fields[7]
        self.pdeprel = fields[8]

