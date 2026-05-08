# -*- coding: utf-8 -*-#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.parameter import Parameter
import numpy as np
from utils.process import normalize_adj
from collections import Counter
import random
# 將隨機種子設定為0
random.seed(0)
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1,
                                                                                                   2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                for j in range(self.nheads):
                    self.add_module('attention_{}_{}'.format(i + 1, j),
                                    GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True))

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        input = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                temp = []
                x = F.dropout(x, self.dropout, training=self.training)
                cur_input = x
                for j in range(self.nheads):
                    temp.append(self.__getattr__('attention_{}_{}'.format(i + 1, j))(x, adj))
                x = torch.cat(temp, dim=2) + cur_input
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x + input


class Encoder(nn.Module): #Self-Attentive Encoder
    def __init__(self, args):
        super().__init__()

        self.__args = args

        # Initialize an LSTM Encoder object.
        self.__encoder = LSTMEncoder( 
            self.__args.word_embedding_dim, #64
            self.__args.encoder_hidden_dim, #256
            self.__args.dropout_rate        #0.4
        )

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.__args.word_embedding_dim,   #32
            self.__args.attention_hidden_dim, #1024
            self.__args.attention_output_dim, #128
            self.__args.dropout_rate          #0.4
        )

    def forward(self, word_tensor, seq_lens):#seq_lens=[42, 17, 16, 16, 14, 13, 13, 12, 12, 9, 8, 8, 7, 6, 6, 4]包含了每個序列的原始長度的列表。
        lstm_hiddens = self.__encoder(word_tensor, seq_lens) #LSTMEncoder forward lstm_hiddens(16, 42, 256)
        attention_hiddens = self.__attention(word_tensor, seq_lens) # SelfAttention forward attention_hiddens(16, 42, 128)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=2) #E = [H || A], E = hiddens (16, 42, 384)    
        return hiddens
        ###########AGIF##################
        #c = self.__sentattention(hiddens, seq_lens) #UnflatSelfAttention forward，在CLID論文裡是不需要的。 原本是return hiddens, c(16, 384)
        #return hiddens, c



class ModelManager(nn.Module):#從這裡建構模型的，從train.py所使用。

    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()#調用了父類 nn.Module 的構造函数，確保正确地初始化了模型。

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        #建了一个嵌入層（Embedding Layer），用於將输入的token轉換為詞嵌入向量。
        #這個嵌入層的大小由 self.__num_word 决定，每个單詞索引將映射為一個 self.__args.word_embedding_dim 维度的詞嵌入向量。
        self.__embedding = nn.Embedding(
            self.__num_word,                   #token的總數
            self.__args.word_embedding_dim     #每個token的嵌入維度。
        )
        #這是GRU?
        self.G_encoder = Encoder(args) # Self-Attentive Encoder 的 __init__

        # Initialize an Decoder object for intent.
        self.__intent_decoder = nn.Sequential(#預測意圖（intent）的神经網路模型的一部分 intent decoder
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                      self.__args.encoder_hidden_dim + self.__args.attention_output_dim),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim, self.__num_intent),
        )

        self.__intent_embedding = nn.Parameter(#這是做什麼的?
            torch.FloatTensor(self.__num_intent, self.__args.intent_embedding_dim))  # 191, 32
        nn.init.normal_(self.__intent_embedding.data)
        
        # Chunk-Level Intent Detection 的 IntentBiLSTM
        self.__intent_lstm = LSTMEncoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.dropout_rate
        )

        # Chunk-Level Intent Detection 的 window-self-attention
        self.__WindowSelfattention = SelfAttention(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.attention_hidden_dim,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.dropout_rate
        )

        # Initialize an Decoder object for slot.
        self.__slot_decoder = LSTMDecoder(
            args,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot, self.__args.dropout_rate,
            embedding_dim=self.__args.slot_embedding_dim)

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot embedding:			    {};'.format(self.__args.slot_embedding_dim))
        print('\tdimension of slot decoder hidden:  	    {};'.format(self.__args.slot_decoder_hidden_dim))
        print('\thidden dimension of self-attention:        {};'.format(self.__args.attention_hidden_dim))
        print('\toutput dimension of self-attention:        {};'.format(self.__args.attention_output_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def generate_adj_gat(self, index, batch):#看不懂什麼意思?
        intent_idx_ = [[torch.tensor(0)] for i in range(batch)]
        for item in index:
            intent_idx_[item[0]].append(item[1] + 1)
        intent_idx = intent_idx_
        adj = torch.cat([torch.eye(self.__num_intent + 1).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in intent_idx[i]:
                adj[i, j, intent_idx[i]] = 1.
        if self.__args.row_normalized:
            adj = normalize_adj(adj)
        if self.__args.gpu:
            adj = adj.cuda()
        return adj

    def get_major_intent(self, sub_utterance_intent): #決定子對話的意圖標籤
        if len(sub_utterance_intent) == 0:
            return None
        intents, counts = torch.unique(sub_utterance_intent, return_counts=True)
        max_count = torch.max(counts).item()
        major_intents = intents[counts == max_count]
        if len(major_intents) == 1:
            return major_intents.item()
        else:
            return random.choice(major_intents).item()    

    def forward(self, text, seq_lens, window_seq_lens, window_size= 1, n_predicts=None, forced_slot=None, forced_intent=None):#在process.py會用到 n_predicts, forced_slot是slot的向量
        word_tensor = self.__embedding(text)# text的張量是(16，42) = 一批輸入文本數據 進行詞嵌入AGIF使用64維。word_tensor的張量是(16，42，64)
        g_hiddens = self.G_encoder(word_tensor, seq_lens) #Encoder forward
        intent_lstm_out = self.__intent_lstm(g_hiddens, seq_lens)  # hiddens (batch_size, seq_len, dimension)
        win_h = intent_lstm_out
        batch_size, seq_len, dimension = win_h.size()
        window_num = max(seq_len - window_size + 1, 1)  # chunk數量
        chunk_h = torch.zeros(batch_size, window_num, dimension).to(win_h.device)
 
        for j in range(window_num):
            temp = win_h[:, j:j + window_size, :]  # 切片獲取子张量，同时保留 batch_size 维度
            At = self.__WindowSelfattention(temp, window_size)  # self-attention At=(3, 384)         
            chunk_h[:, j] = torch.sum(At, dim=1)  # 在维度1上求和，即对窗口内的不同位置求和            
        chunk_h = F.dropout(chunk_h, p=self.__args.dropout_rate, training=self.training)   
        pred_intent = self.__intent_decoder(chunk_h)#pred_intent = [16, 46, 18]

        ####Chunk-Intent Detection and Intent Transition Point Identification######################################
        pred_intent_window = torch.argmax(torch.sigmoid(pred_intent), dim=-1)#dim=-1指定最內層的維度。
        TP_index = self.__num_intent - 1 # TP 標籤的index
        all_sub_utterance_intents = []

        for i in range(batch_size):#batch_size = 16 
            intent_window = pred_intent_window[i, 0:(window_seq_lens[i])]
            #print(intent_window)
            tp_indices = [j for j in range(window_seq_lens[i]) if intent_window[j] == TP_index]
            current_utterance_intents = []
            
            prev_tp_idx = 0  # 前一個tp位置
            for tp_idx in tp_indices:
                sub_utterance_intent = intent_window[prev_tp_idx:tp_idx]
                major_intent = self.get_major_intent(sub_utterance_intent)
                if major_intent is not None:
                    current_utterance_intents.append([i, major_intent])
                prev_tp_idx = tp_idx + 1

            # 處理最後一個子對話
            sub_utterance_intent = intent_window[prev_tp_idx:window_seq_lens[i]]
            major_intent = self.get_major_intent(sub_utterance_intent)
            if major_intent is not None:
                current_utterance_intents.append([i, major_intent])

            unique_utterance_intents = list(map(list, set(map(tuple, current_utterance_intents))))  # 避免有重複的多意圖標籤
            all_sub_utterance_intents.extend(unique_utterance_intents)   
        
        intent_index = torch.tensor(all_sub_utterance_intents, dtype=torch.long)
        ######slot filling################################
        adj = self.generate_adj_gat(intent_index, len(pred_intent))#adj = "adjacency matrix"，即鄰接矩陣。鄰接矩陣是用於表示圖形或網絡中節點之間關係的一種數學結構。
        pred_slot = self.__slot_decoder( #LSTMDecoder forward
            g_hiddens, seq_lens,
            forced_input=forced_slot,
            adj=adj,
            intent_embedding=self.__intent_embedding
        )
        #################################################
        if n_predicts is None:
            pred_intent = torch.cat([pred_intent[i][:seq_lens[i] - window_size + 1] for i in range(len(seq_lens))], dim=0)#處理intent_labl。  
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1)#logsigmoid
        else:#prediction()時，才會進這個條件式

            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()#將從 GPU 上的 PyTorch 張量轉換為 CPU 上的 Python 列表的形式


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim #64
        self.__hidden_dim = hidden_dim // 2  #256
        self.__dropout_rate = dropout_rate   #0.4

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return padded_hiddens #H = {h1, h2, . . . , hT }.


class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, args, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        :param embedding_dim: if it's not None, the input and output are relevant.
        :param extra_dim: if it's not None, the decoder receives information tensors.
        """

        super(LSTMDecoder, self).__init__()
        self.__args = args
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                torch.randn(1, self.__embedding_dim),
                requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.
        if self.__extra_dim is not None and self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim + self.__embedding_dim
        elif self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        elif self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__embedding_dim
        else:
            lstm_input_dim = self.__input_dim

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1
        )

        self.__graph = GAT(
            self.__hidden_dim,
            self.__args.decoder_gat_hidden_dim,
            self.__hidden_dim,
            self.__args.gat_dropout_rate, self.__args.alpha, self.__args.n_heads,
            self.__args.n_layers_decoder)

        self.__linear_layer = nn.Linear(
            self.__hidden_dim,
            self.__output_dim
        )

    def forward(self, encoded_hiddens, seq_lens, forced_input=None,  # extra_input=None,
                adj=None, intent_embedding=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :param forced_input: is truth values of label, provided by teacher forcing.
        :param extra_input: comes from another decoder as information tensor.
        :return: is distribution of prediction labels.
        """

        input_tensor = encoded_hiddens
        output_tensor_list, sent_start_pos = [], 0
        if self.__embedding_dim is not None and forced_input is not None:

            forced_tensor = self.__embedding_layer(forced_input)[:, :-1]
            prev_tensor = torch.cat((self.__init_tensor.unsqueeze(0).repeat(len(forced_tensor), 1, 1), forced_tensor),
                                    dim=1)
            combined_input = torch.cat([input_tensor, prev_tensor], dim=2)
            dropout_input = self.__dropout_layer(combined_input)
            packed_input = pack_padded_sequence(dropout_input, seq_lens, batch_first=True)
            lstm_out, _ = self.__lstm_layer(packed_input)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            for sent_i in range(0, len(seq_lens)):
                if adj is not None:#它檢查變數 adj 是否為非空（即是否有圖形結構的資訊）。如果 adj 不為空，則條件成立。
                    lstm_out_i = torch.cat((lstm_out[sent_i][:seq_lens[sent_i]].unsqueeze(1),  
                                            intent_embedding.unsqueeze(0).repeat(seq_lens[sent_i], 1, 1)), dim=1)
                    lstm_out_i = self.__graph(lstm_out_i, adj[sent_i].unsqueeze(0).repeat(seq_lens[sent_i], 1, 1))[:, 0]
                else:
                    lstm_out_i = lstm_out[sent_i][:seq_lens[sent_i]]
                linear_out = self.__linear_layer(lstm_out_i)
                output_tensor_list.append(linear_out)
        else:
            prev_tensor = self.__init_tensor.unsqueeze(0).repeat(len(seq_lens), 1, 1)
            last_h, last_c = None, None
            for word_i in range(seq_lens[0]):
                combined_input = torch.cat((input_tensor[:, word_i].unsqueeze(1), prev_tensor), dim=2)
                dropout_input = self.__dropout_layer(combined_input)
                if last_h is None and last_c is None:
                    lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                else:
                    lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))

                if adj is not None:
                    lstm_out = torch.cat((lstm_out,
                                          intent_embedding.unsqueeze(0).repeat(len(lstm_out), 1, 1)), dim=1)
                    lstm_out = self.__graph(lstm_out, adj)[:, 0]

                lstm_out = self.__linear_layer(lstm_out.squeeze(1))
                output_tensor_list.append(lstm_out)

                _, index = lstm_out.topk(1, dim=1)
                prev_tensor = self.__embedding_layer(index.squeeze(1)).unsqueeze(1)
            output_tensor = torch.stack(output_tensor_list)
            output_tensor_list = [output_tensor[:length, i] for i, length in enumerate(seq_lens)]

        return torch.cat(output_tensor_list, dim=0)


class QKVAttention(nn.Module): #Self-Attentive Encoder 的 Self-Attention
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ) / math.sqrt(self.__hidden_dim), dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module): #Self-Attentive Encoder 的 Self-Attention

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )
        return attention_x


class UnflatSelfAttention(nn.Module):#自注意力（Self-Attention）層，在CLID論文裡是不需要的。
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1) #pt = softmax(we et + b)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)# c = Σt(ptet)
        return context

