# encoding: utf-8

"""
Created by Gözde Gül Şahin
20.05.2018
Flipper (Rotatation) and Cropper classes

"""
__author__ = 'Gözde Gül Şahin'

from data_augmentation.SP.chunker import *
import numpy
class rotator:
    def __init__(self, asent, aloi = [u"nsubj", u"dobj", u"iobj", u"obj", u"obl"],
                                pl = u"root",
                         multilabs = [u"case", u"fixed", u"flat", u"cop", u"compound"],
                              prob = 0.3):

        self.sent = asent
        self.cnkr = chunker(asent, aloi, pl, multilabs)#深入了解
        self.flex_chunks = self.cnkr.get_all_chunks()
        self.threshold = int(prob*10)

    def _reorder(self, chunk_order):#一個代表片段順序的列表 chunk_order，並根據這個順序將句子中的標記重新排列
        """
        Only insert interesting chunks - get rid of adverbial stuff
        :param chunk_order: order of the chunks (0,2,1)...
        :return: new index list
        """
        chnks = self.flex_chunks
        # the new order
        nlst = [0]
        for chunk_id in chunk_order:
            chnk = chnks[chunk_id]
            # add nodes in order
            for j in range(chnk.min, chnk.max+1):
                nlst.append(j)
        return nlst

    def rotate(self, fields_order, texts, slots):
        """
        Shuffle the sentence by moving the flexible chunks around
        :param maxshuffle: Maximum number of shuffles per sentence
        :return: shuffled sentences + original sentences
        """
        shufledSents, shufledSents_slot = [], []

        # if there are enough flexible chunks
        if len(self.flex_chunks) > 1:#已經有大於0了

           # permlst = list(itertools.permutations(range(len(self.flex_chunks))))# 生成所有可能的排列
            num_shuffles = perm(len(self.flex_chunks))#maxshuffle就是我要增強出3個旋轉句子，如果可以的話
            permlst = list(itertools.permutations(range(len(self.flex_chunks))))
            poss_perms = random.sample(permlst, num_shuffles)#itertools.permutations 生成所有可能的排列，並隨機選擇 num_shuffles 個排列。

            for chunk_order in poss_perms:
                num = numpy.random.choice(numpy.arange(1, 11))# 生成 1 到 10 之間的一個隨機數

                if num <= self.threshold:# 如果隨機數小於等於閾值，就進行資料擴增
                    neworder = self._reorder(chunk_order)# 重新排序區塊
                    sen_order=[]
                    for i, j in zip(neworder,range(len(neworder))):
                        if i==0:
                            continue
                        sen_order.append(fields_order[i-1][0])
                    seen = set()
                    unique_order = [x for x in sen_order if x not in seen and not seen.add(x)]
                    new_texts, new_slots = self.sent.reorder(unique_order, texts, slots)# 根據新順序重新排序句子和槽位
                    shufledSents.append(new_texts)# 將新句子添加到結果列表中
                    shufledSents_slot.append(new_slots)# 將新槽位添加到結果列表中

        return shufledSents, shufledSents_slot


class cropper:
    def __init__(self, asent, aloi = [u"nsubj", u"dobj", u"iobj", u"obj", u"obl"],
                                pl = u"root",
                         multilabs = [u"case", u"fixed", u"flat", u"cop", u"compound"],
                              prob = 0.3):

        self.sent = asent
        self.cnkr = chunker(asent, aloi, pl, multilabs)
        self.flex_chunks = self.cnkr.get_all_chunks()
        self.threshold = int(prob*10)

    def _reorder(self, chunk_order):
        """
        Only insert interesting chunks - get rid of adverbial stuff
        :param chunk_order: order of the chunks (0,2,1)...
        :return: new index list
        """
        chnks = self.flex_chunks
        # the new order
        nlst = [0]
        for chunk_id in chunk_order:
            chnk = chnks[chunk_id]
            # add nodes in order
            for j in range(chnk.min, chnk.max+1):
                nlst.append(j)
        return nlst

    def _get_root_chunk_ix(self):
        """
        Get the index of the root chunk
        :return: index of root chunk
        """
        for i, cnk in enumerate(self.flex_chunks):
            if cnk.type==u"root":
               return i


    def crop(self, fields_order, texts, slots):
        """
        Crop the sentence into meaningful small sentences
        :return: cropped sentences + original sentences
        """
        croppedSents, croppedSents_slot = [], []

        # if there are enough flexible chunks
        if len(self.flex_chunks) > 1:# 如果有足夠的彈性區塊
            # get the root chunk
            root_ix = self._get_root_chunk_ix()# 獲取根區塊索引
            for i in range(len(self.flex_chunks)):#遍歷每個靈活區塊。
                if i==root_ix:#跳過根區塊。
                    continue
                elif i < root_ix:# 當前區塊在根區塊之前
                    chunk_order = [i, root_ix]
                elif i > root_ix:# 當前區塊在根區塊之後
                    chunk_order = [root_ix, i]
                # if above threshold
                num = numpy.random.choice(numpy.arange(1, 11))# 生成1到10之間的一個隨機數
                # augment data if we are above the probability threshold
                if num <= self.threshold:# 如果隨機數小於等於閾值，則進行資料擴增
                    neworder = self._reorder(chunk_order)
                    sen_order=[]
                    for i, j in zip(neworder,range(len(neworder))):
                        if i==0:
                            continue
                        sen_order.append(fields_order[i-1][0])
                    seen = set()
                    unique_order = [x for x in sen_order if x not in seen and not seen.add(x)]
                    new_texts, new_slots = self.sent.reorder(unique_order, texts, slots)# 根據新順序重新排序句子和槽位                   
                    croppedSents.append(new_texts)
                    croppedSents_slot.append(new_slots)

        return croppedSents, croppedSents_slot