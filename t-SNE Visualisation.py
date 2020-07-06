	# %%
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import json
from pprint import *

import numpy
from collections import Counter
import keras
from keras.models import *
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dropout, Dense, concatenate
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.datasets import conll2000
from sklearn.model_selection import train_test_split
from keras_self_attention import SeqSelfAttention
from keras.models import model_from_json
from keras.models import load_model
import json
from collections import Counter
from keras.preprocessing.sequence import pad_sequences


# %%
# with open('configure_file') as f:
    # data = json.load(f)

# %%
max_len = None
max_len_char = 0
EPOCHS = 10
EMBED_DIM = 300
BiRNN_UNITS = 200
POSTagging_file_path = r'bhoj.txt'
wordtag = dict()

# %%
def load_data(POSTagging_file_path, min_freq=1):
    file_train = _parse_data(open(POSTagging_file_path))
    word_counts = Counter(row[1].lower() for sample in file_train for row in sample if len(row) >= 3)
    vocab = ['<pad>', '<unk>']
    vocab += [w for w, f in iter(word_counts.items()) if f >= min_freq]
    # in alphabetic order
    # print(file_train[1:3])
    # print(len(file_train))
    # pos_tags = sorted(list(set(sample[row[1]] for sample in file_train for row in range(len(sample)))))
    nl = set()
    for sample in file_train:
        for kl in sample:
            if len(kl) < 3:
                continue
            else:
                nl.add(kl[2])
                wordtag[kl[1]]=kl[2]
    # print(nl)
    nl.add("null")
    pos_tags = sorted(list(nl))
    characters = set()
    for sample in file_train:
        for row in sample:
            if len(row) >= 3:
                for j in row[1]:
                    characters.add(j)
    characters = sorted(characters)
    characters.insert(0, '<unk>')
    characters.insert(0, '<pad>')
    global max_len_char
    for sample in file_train:
        for row in sample:
            if len(row) >= 3:
                max_len_char = max(max_len_char, len(row[1]))

    # in alphabetic order
    # chunk_tags = sorted(list(set(row[2] for sample in file_train for row in sample)))

    train = _process_data(file_train, vocab, pos_tags)
    return train, (vocab, pos_tags, characters)


# %%
def _parse_data(fh):
    string = fh.read()
    # print(string)
    data = []
    for sample in string.strip().split('\n\n'):
        data.append([row.split() for row in sample.split('\n')])
    fh.close()
    return data


def _process_data(data, vocab, pos_tags, maxlen=None, onehot=False):
    global max_len
    if max_len is None:
        max_len = max(len(s) for s in data)

    maxlen = max_len
    # print(vocab)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    # set to <unk> (index 1) if not in vocab
    # x = [[word2idx.get(w[1].lower(), 1) for w in s] for s in data]
    x = []
    # print(word2idx)
    for i in data:
        sl = []
        for w in i:
            if len(w) < 3:
                sl.append(word2idx.get('null', 1))
            else:
                sl.append(word2idx.get(w[1].lower(), 1))
        x.append(sl)

    y_pos = []
    for s in data:
        t = []
        for w in s:
            if len(w) < 3:
                t.append(pos_tags.index('null'))
            else:
                t.append(pos_tags.index(w[2]))
        y_pos.append(t)

    x = pad_sequences(x, maxlen)  # left padding

    # lef padded with -1. Indeed, any integer works as it will be masked
    y_pos = pad_sequences(y_pos, maxlen, value=-1)
    print("y")
    print(y_pos.shape)
    

    if onehot:
        y_pos = numpy.eye(len(pos_tags), dtype='float32')[y]
    else:
        y_pos = numpy.expand_dims(y_pos, 2)
        print("y_pos")
        print(y_pos.shape)
    return x, y_pos


def padd_word(seq, max__len):
    leng = len(seq)
    ll = [0 for i in range(max__len - leng)]
    seq = ll + seq
    return seq


def process_data_to_characters(characters, vocab, train_x):
    sl = []
    char2idx = dict((w, i) for i, w in enumerate(characters))
    idx_2_word = dict((i, w) for i, w in enumerate(vocab))
    seq_ = []
    for s in train_x:
        nl = []
        for word in s:
            if idx_2_word[word] == '<pad>':
                nl.append([0] * max_len_char)

            elif idx_2_word[word] == '<unk>':
                nl.append([1] * (max_len_char))

            else:
                l2 = []
                for c in idx_2_word[word]:
                    l2.append(char2idx.get(c, 0))
                l2 = padd_word(l2, max_len_char)
                nl.append(l2)
        sl.append(nl)

    return numpy.asarray(sl), idx_2_word


# %%
def classification_report(y_true, y_pred, labels):
    '''Similar to the one in sklearn.metrics,
    reports per classs recall, precision and F1 score'''
    # print(y_pred)
    # print(y_true)
    y_true = numpy.asarray(y_true).ravel()
    y_pred = numpy.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('',
                                                    'recall',
                                                    'precision',
                                                    'f1-score',
                                                    'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = list(zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report]))
    N = len(y_true)
    print(formatter('avg / total',
                    sum(report2[0]) / N,
                    sum(report2[1]) / N,
                    sum(report2[2]) / N, N) + '\n')
    actual = Counter(y_true)
    del actual[-1]
    accuracy = sum(corrects.values()) / sum(actual.values())
    print('Accuracy:', accuracy)


# %%
train, voc = load_data(POSTagging_file_path)
(train_x, train_y) = train
(vocab, class_labels, characters) = voc
X_train2, x_test2, Y_train2, y_test2 = train_test_split(train_x, train_y, test_size=0.2, random_state=2018)
Y_train = Y_train2
y_test = y_test2
# print(x_test2)
X_train, _ = process_data_to_characters(characters, vocab, X_train2)
x_test, _ = process_data_to_characters(characters, vocab, x_test2)
print(X_train.shape)
# print(X_train2[0:10])
print(class_labels)

# %%
def get_sentence(POSTagging_file_path):
    file_train = _parse_data(open(POSTagging_file_path))
    sentences = list()
    for sample in file_train:
        l = list()
        for row in sample:
            if (len(row)>=3):
                l.append(row[1].lower())
        sentences.append(l)
    return sentences
    
sent = get_sentence(POSTagging_file_path)

from gensim.models import Word2Vec
# model_ted = Word2Vec(sentences=sent, size= 2 * (EMBED_DIM // 5), window=3, min_count=1, workers=4)
# model_ted2 = Word2Vec(sentences=sent, size= 2 * (EMBED_DIM // 5), window=3, min_count=30, workers=4)
model = Word2Vec(sent, size=2 * (EMBED_DIM // 5), window=3, min_count=1, workers=4,sg=1)
# for word in model.wv.vocab:
#   print(word)
#   print(model[word].shape)
#   print(wordtag[word])
#   break

        
############### T-SNE ###############################
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


import seaborn as sns

%matplotlib inline
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    # classes = ['JJ', 'NNP', 'PRP', 'VAUX', 'NST']
    classes = ['VAUX', 'NST', 'RP', 'QC', 'QF']
    labels = []
    tokens = []

    for word in model.wv.vocab:
      if wordtag[word] in classes:
        tokens.append(model[word])
        labels.append(wordtag[word])

    # from collections import Counter
    # print(Counter(labels))

    # from sys import exit
    # exit()

    tsne_model = TSNE(perplexity=23, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    palette = sns.color_palette("bright", 5)
    sns.scatterplot(x,y, hue=labels, legend='full', palette=palette)

    # hindi_font = FontProperties(fname = "/content/drive/Shared drives/Explo/nirmala.ttf", size = 12)
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom') #, fontproperties=hindi_font)
    plt.show()

tsne_plot(model)
from sys import exit
exit()
########################################3
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab)
    # define weight matrix dimensions with all 0
    weight_matrix = numpy.zeros((vocab_size, 2 * (EMBED_DIM // 5)))
    # step vocab, store vectors using the Tokenizer's integer mapping
    #wordvectors = embedding.wv
    for i,word in enumerate(vocab):
        #weight_matrix[i] = wordvectors.word_vec(word)
        try:
            weight_matrix[i] = embedding[word]
        except:
            pass
    return weight_matrix

weight_matrix = get_weight_matrix(model_ted,vocab)

# %%
# Defining the layers of Bi-LSTM + CRF model
main_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name='main_input')
word_input = Input(shape=(X_train2.shape[1],), name='word_input')
embedded_word = (Embedding(len(vocab), 2 * (EMBED_DIM // 5), weights =[ weight_matrix], mask_zero=True, trainable= True))(word_input)
char_in = (
    TimeDistributed(Embedding(input_dim=len(characters), output_dim=10, input_length=max_len_char, mask_zero=True)))(
    main_input)
char_emb = (TimeDistributed(LSTM(3 * (EMBED_DIM // 5), return_sequences=False)))(char_in)

# Concatinating Word and Character Embeddings
inp = concatenate([embedded_word, char_emb])
o = (Bidirectional(LSTM(BiRNN_UNITS, return_sequences=True, dropout=0.2)))(inp)

# Regualrized Self attention layer at top of Bi-LSTM
'''o = (SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                      kernel_regularizer=keras.regularizers.l2(1e-4),
                      bias_regularizer=keras.regularizers.l1(1e-4),
                      attention_regularizer_weight=1e-4,
                      name='Attention'))(o)'''

crf = CRF(len(class_labels), sparse_target=True, name='crf')

o = (crf)(o)
model = Model(input=[word_input, main_input], output=o)

# %%
model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
X_train = numpy.array(X_train).reshape(len(X_train), max_len, max_len_char)
print(max_len_char)
# print(X_train.shape)
print(X_train2.shape)
print(X_train.shape)
print(len(Y_train))
print(len(Y_train2))
model.fit([X_train2, X_train], numpy.array(Y_train).reshape(len(Y_train2), max_len, 1), batch_size=32, epochs=EPOCHS,
          validation_split=0.1)

# %%
# model_json = model.to_json()
x_test = numpy.array(x_test).reshape(len(x_test), max_len, max_len_char)
y = model.predict([x_test2, x_test]).argmax(-1)

# %%
l4 = [0] * max_len_char
l4 = numpy.asarray(l4)
y_pred = []
test_y_true = []
x_test = numpy.array(x_test).reshape(len(x_test), max_len, max_len_char)
y = model.predict([x_test2, x_test]).argmax(-1)
for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        if (numpy.array_equal(x_test[i][j], l4)):
            continue
        y_pred.append(y[i][j])
        l1 = [y_test[i][j], ]
        test_y_true.append(l1)
y_pred = numpy.asarray(y_pred)
test_y_true = numpy.asarray(test_y_true)

# %%
print('\n---- Result of BiLSTM-CRF ----\n')
classification_report(test_y_true, y_pred, class_labels)