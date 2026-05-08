#這裡是用來獲得將每個詞標註意圖標籤和tp的地方
#先執行python annotate.py
#再執行python TP.py
import nltk
def annotate_words(sentence, sentences, annotations):
    words_in_sentence = sentence.split()
    annotation_positions = [None] * len(words_in_sentence)
    
    for sentence, annotation in zip(sentences, annotations):
        words_to_find = sentence.split()
        sentence_length = len(words_to_find)
        for i in range(len(words_in_sentence) - sentence_length + 1):
            if words_in_sentence[i:i+sentence_length] == words_to_find:
                for j in range(sentence_length):
                    annotation_positions[i+j] = '0'
    for i, word in enumerate(annotation_positions):#空白的就標註TP
        if word == None:  # Check if the word is already annotated
            annotation_positions[i] = 'TP'

    return annotation_positions

ATIS_label_path = '/home/cyl22/work/joint_model/mix/clid-mark/annotate_tp/SNIPS/label'
ATIS_seq_path = '/home/cyl22/work/joint_model/mix/clid-mark/annotate_tp/SNIPS/seq.in'
MixATIS_label_path = '/home/cyl22/work/joint_model/mix/clid-mark/annotate_tp/MixSNIPS_clean/label'
MixATIS_seq_path = '/home/cyl22/work/joint_model/mix/clid-mark/annotate_tp/MixSNIPS_clean/seq.in'

MixATIS_seq,MixATIS_label,ATIS_seq,ATIS_label=[],[],[],[]
MixATIS_sen, MixATIS_lab= [], []

with open(MixATIS_seq_path, 'r', encoding="utf8") as fr:#ATIS資料集
    MixATIS_seq = fr.readlines()

with open(MixATIS_label_path, 'r', encoding="utf8") as fr:#ATIS資料集
    MixATIS_label = fr.readlines()

with open(ATIS_seq_path, 'r', encoding="utf8") as fr:#ATIS資料集
    ATIS_seq = fr.readlines()

with open(ATIS_label_path, 'r', encoding="utf8") as fr:#ATIS資料集
    ATIS_label = fr.readlines()

output_path = '/home/cyl22/work/joint_model/mix/clid-mark/annotate_tp/MixSNIPS_clean_tp.txt'
with open(output_path, 'w') as output_file:
    for i, MixATIS_line in enumerate(MixATIS_seq):  # 讀取每一行
        print(MixATIS_line)
    
        MixATIS_line = MixATIS_line.strip()  # 移除行末的换行符
        for j,ATIS_line in enumerate(ATIS_seq):  # 讀取每一行
            ATIS_line = ATIS_line.strip()  # 移除行末的換行符
            if ATIS_line in MixATIS_line:
                if ATIS_label[j].strip() in MixATIS_label[i].strip():#ATIS的意圖標籤有在MixATIS的多意圖標籤裡。
                    #print(ATIS_label[j])
                    #input()
                    MixATIS_lab.append(ATIS_label[j].strip())
                    MixATIS_sen.append(ATIS_line)
        #input()
        annotated_sentence = annotate_words(MixATIS_line, MixATIS_sen, MixATIS_lab)#開始標註TP

        #words = MixATIS_line.split()
        #tagged_words = nltk.pos_tag(words)#詞性標註

        #for k, tagged_word in enumerate(tagged_words):  # 讀取每一行
            #print(tagged_word[1],annotated_sentence[k])
        #    if tagged_word[1] != 'CC' and annotated_sentence[k]=='TP':
        #        annotated_sentence[k] = '0'
        #print(annotated_sentence)
        output_file.write('\n'.join(map(str, annotated_sentence)))
        output_file.write('\n\n\n')
        #print(MixATIS_lab)
        #print(MixATIS_sen)
        MixATIS_sen = []
        MixATIS_lab = []

        #input()



print('Finish!')