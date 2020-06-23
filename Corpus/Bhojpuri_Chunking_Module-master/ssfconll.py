import os
file_in = './bhojpuri-chunked_cleaned.txt'
error= './error.txt'
temp_file = './Chunk_file_temp_iob.txt'
error1=open(error,'w',encoding='utf-8')
def sublist_spliter(sequence):
    group = []
    for seq in sequence:
        if seq[0] != '))':
            group.append(seq)
        elif group:
            yield group
            group = []
def file_writer(file_obj, list_list):
    for chunk in list_list:
        if (chunk==[]):
            continue
        new_str=chunk[0][1]+'\t'+chunk[0][2]+'\t'+"B-"+chunk[0][3]+'\n'
        file_obj.write(new_str)
        for word in chunk[1:]:
            if (word[1]=='।'):
                new_str=word[1]+'\t'+word[2]+'\t'+"O"+'\n'
            else:
                new_str = word[1]+'\t'+word[2]+'\t'+"I-"+word[3]+'\n'
            file_obj.write(new_str)
chunk_temp_file = open(temp_file, 'w', encoding='utf-8')

sentence_ = []
with open(file_in, 'r', encoding='utf-8') as f1:
    for line in f1:
        if line != '\n':
            pair = line.strip().split('\t')
            sentence_.append(pair)
            # print(pair)
        if line == '\n':
            new_sent = list(sublist_spliter(sentence_))
            for each_chunk in new_sent:
                for each_pair in each_chunk[1:]:
                    each_pair.append(each_chunk[0][2])
                each_chunk.pop(0)
            print(new_sent)
            file_writer(chunk_temp_file, new_sent)
            chunk_temp_file.write('\n')
            sentence_.clear()
