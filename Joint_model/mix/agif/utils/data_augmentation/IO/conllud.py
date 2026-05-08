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

def convert_to_conllu(sentences):
    # 加載英文模型
    nlp = spacy.load("en_core_web_sm")

    conllu = []
    for sentence in sentences:
        sentence = ' '.join(sentence)
        doc = nlp(sentence)  # 處理句子
        conllu_output = ""
        for i, token in enumerate(doc):
            head_id = token.head.i + 1 if token.dep_ != "ROOT" else 0
            feats = "|".join(f"{key}={value}" for key, value in token.morph.to_dict().items()) if token.morph else "_"
            # 格式化每個 token 的 CoNLL-U 行
            conllu_output += "\t".join([
                str(i + 1),        # ID
                token.text,       # FORM
                token.lemma_,     # LEMMA
                token.pos_,       # UPOS
                token.tag_,       # XPOS
                feats,            # FEATS
                str(head_id),     # HEAD
                token.dep_,       # DEPREL
                "_",              # DEPS
                "_"               # MISC
            ])
            if i != len(doc) - 1:  # 如果不是最後一個 token
                conllu_output += "\n"  # 添加換行符號
        # 將句子的 CoNLL-U 格式添加到列表中
        conllu.append(conllu_output)
    return conllu

class conllUD:

    def __init__(self, fpath=None):
        """
        init: read sentences into conllUDsent structure
        """
        self.sents = []
        if fpath is not None:
            self.sents = self._read_file(fpath)
            
        else:
            print("File can not be opened, check path")

    def _read_file(self, sentences):
        """
        read_file: read tokens and create a dependency tree for all UD sentences
        """
        conllsentences = []
        sent_no = 0

        #sentences = strIn.split("\n\n")
        for sent in sentences:
            #print(sent)
            #input()
            if(len(sent)>0):
                lines = sent.split("\n")
                # create new conllud sentence
                csent = conllUDsent()
                rows = []
                tok_index = 1#計算句子的第幾個詞
                for line in lines:
                    #print(line.split("\t"))
                    ctoken = conllUDtoken(line.split("\t"),tok_index)
                    csent.add_token(ctoken)
                    rows.append(line.split())
                    tok_index+=1
                csent.rows = rows

                # add token words
                csent.tokenWords = [row[1] for row in rows]#句子的詞
                csent.tokenLemmas = [row[2] for row in rows]#句子的詞的詞幹
                conllsentences.append(csent)

                # build the dependency tree
                # head -> all children tokens
                for tok in csent.tokens:
                    #print(tok)
                    if tok.head in csent.deptree:
                        csent.deptree[tok.head].append(tok)
                    else:
                        csent.deptree[tok.head] = [tok]
                    #input()

        return conllsentences

class conllUDsent:
    def __init__(self):
        self.tokens = []
        # token words and lemmas
        self.tokenWords = []
        self.tokenLemmas = []
        # for reordering purposes
        self.deptree = {}
        self.rows = []#句子裡的詞的資訊

    def add_token(self, token):#放每一個詞的資訊
        self.tokens.append(token)

    def print_sent_ord(self,ord):
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

    def reorder(self, neword, slots):
        """
        reorder: Reorder the tokens of the sentence with the given new order (neword)
        """
        newrows = []

        # mapping[oldorder]=neworder
        mapping = {}
        mapping[u"0"]=u"0"

        # first put them together
        for i, j in zip(neword,range(len(neword))):
            if i==0:
                continue
            newrows.append(copy.deepcopy(self.rows[i-1]))
            mapping[self.tokens[i-1].depid] = self.tokens[j-1].depid

        # replace old ids with new ids
        for r in newrows:
            # change id
            r[0] = str(mapping[r[0]])
            if r[6] in mapping:
                r[6] = str(mapping[r[6]])
            else:
                r[6] = "_"

        # reorder slots based on new token order
        new_slots = []
        for idx in neword:
            if idx == 0:
                continue
            new_slots.append(slots[idx - 1])

        return newrows, new_slots

class conllUDtoken:
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

