from __future__ import absolute_import

import os
import re
import numpy as np
import tensorflow as tf
from itertools import chain
import json

stop_words=set(["a","an","the"])


def load_candidates(data_dir, task_id):
    assert task_id > 0 and task_id < 7
    candidates=[]
    candidates_f=None
    candid_dic={}
    candidates_f='../personalized-dialog-candidates.txt'
    with open('personalized-dialog-dataset//personalized-dialog-candidates.txt') as f:
        for i,line in enumerate(f):
            candid_dic[line.strip().split(' ',1)[1]] = i
            line=tokenize(line.strip())[1:]
            candidates.append(line)
    # return candidates,dict((' '.join(cand),i) for i,cand in enumerate(candidates))
    return candidates,candid_dic


def load_dialog_task(data_dir, task_id, candid_dic, isOOV):
    '''Load the nth task. There are 5 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 6

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'personalized-dialog-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    if isOOV:
        test_file = [f for f in files if s in f and 'tst-OOV' in f][0]
    else: 
        test_file = [f for f in files if s in f and 'tst.' in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = get_dialogs(train_file,candid_dic)
    test_data = get_dialogs(test_file,candid_dic)
    val_data = get_dialogs(val_file,candid_dic)
    return train_data, test_data, val_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent=sent.lower()
    if sent=='<silence>':
        return [sent]
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() not in stop_words]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result

def parse_dialogs_per_response(lines,candid_dic):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data=[]
    context=[]
    context_profile=[]
    u=None
    r=None
    for line in lines:
        line=line.strip()
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                attribs = line.split(' ')
                for attrib in attribs:
                    r=tokenize(attrib)
                    r.append('$r')
                    # r.append('#'+str(nid))
                    context_profile.append(r)
            else:
                if '\t' in line:
                    u, r = line.split('\t')
                    a = candid_dic[r]
                    u = tokenize(u)
                    r = tokenize(r)
                    # temporal encoding, and utterance/response encoding
                    # data.append((context[:],u[:],candid_dic[' '.join(r)]))
                    data.append((context_profile[:],context[:],u[:],a))
                    u.append('$u')
                    # u.append('#'+str(nid))
                    r.append('$r')
                    # r.append('#'+str(nid))
                    context.append(u)
                    context.append(r)
                else:
                    r=tokenize(line)
                    r.append('$r')
                    # r.append('#'+str(nid))
                    context.append(r)
        else:
            # clear context
            context=[]
            context_profile=[]
    return data

def get_dialogs(f,candid_dic):
    '''Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a single dialog.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines(),candid_dic)

def vectorize_candidates_sparse(candidates,word_idx):
    shape=(len(candidates),len(word_idx)+1)
    indices=[]
    values=[]
    for i,candidate in enumerate(candidates):
        for w in candidate:
            indices.append([i,word_idx[w]])
            values.append(1.0)
    return tf.SparseTensor(indices,values,shape)

def vectorize_candidates(candidates,word_idx,sentence_size):
    shape=(len(candidates),sentence_size)
    C=[]
    for i,candidate in enumerate(candidates):
        lc=max(0,sentence_size-len(candidate))
        C.append([word_idx[w] if w in word_idx else 0 for w in candidate] + [0] * lc)
    return tf.constant(C,shape=shape)

def getid():
  fd = open('word2id.txt','r')
  ids = fd.read().split('\n')
  idx = {}
  for id in ids[:-1]:
    word,id = id.split('\t')
    idx[word] = id
  return idx

def word2id():
#   idx = getid()
    fv = open('entityVector.txt','r')
    embs = fv.read().split('\n')
    fv.close()
    res = []
    entidx = {}
    for emb in embs[:-1]:
        word, vec = emb.split('\t')
        #   res.append(id + '\t' + vec)
        entidx[word] = eval(vec)

    return entidx

# def get_entity_size(data):
#     long_size = 0
#     entity_idx = word2id()
#     for _, s, _, _ in data:
#         for sent in s:
#             size = 0
#             for word in sent:
#                 if word in entity_idx:
#                     size+=1
#             long_size=max(long_size,size)
#     return long_size#max(map(inentidx,chain.from_iterable(s for _, s, _, _ in data)))

# def inentidx(sent):
#     entity_idx = word2id()
#     entity_size = 0
#     for word in sent:
#         if word in entity_idx:
#             entity_size += 1
#     return entity_size
    

def vectorize_data(data, word_idx, sentence_size, batch_size, candidates_size, max_memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    P = []
    S = []
    Q = []
    A = []
    data.sort(key=lambda x:len(x[0]),reverse=True)
    for i, (profile, story, query, answer) in enumerate(data):
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))
        pp = []
        for i, sentence in enumerate(profile, 1):
            lp = max(0, sentence_size - len(sentence))
            pp.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * lp)
         

        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)
            # sse.append([entity_idx[w] if w in entity_idx else [0]*60 for w in sentence] + [[0]*60] * ls)
            # le = max(0,4 - len(entities))
            

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]
        # sse = sse[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)
        
        # lm = max(0, memory_size - len(sse))
        # for _ in range(lm):
        #     sse.append([[0]*60] * sentence_size)

        # sse = np.concatenate(sse, axis=0) 

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq
        
        P.append(np.array(pp))
        S.append(np.array(ss))
        # SE.append(np.array(sse))
        Q.append(np.array(q))
        A.append(np.array(answer))
    return P, S, Q, A

def set_entity_embs(widx):
    # new = {}
    # for word in widx:
    #     new[word]=str([0]*40)
    
    fe = open(r'.//facebook//entity2id.txt','r',encoding='utf-8')
    ents = fe.readlines()
    fe.close()
    res = []
    for e in ents:
        ent = e.split('\t')[0]
        try:
            nid = widx[ent]
        except KeyError:
            nid = len(widx)
            widx[ent] = len(widx)
        res.append('%s\t%d'%(ent,nid))
    res = '\n'.join(res)

    fe = open(r'.//facebook//entity2id.txt','w',encoding='utf-8')
    fe.write(res)
    fe.close()

    return widx

def set_entity_init(widx):
    # fe = open(r'.\facebook\entity2id.txt','r',encoding='utf-8')
    # ents = fe.readlines()
    # fe.close()

    # fv = open(r'.\facebook\emb.txt','r',encoding='utf-8')
    # values = fv.readlines()
    # fv.close()

    # novalue = str([0.0]*20)[1:-1].replace(',',' ')

    # res = ['']*(len(widx)+1)
    # res[0] = novalue

    # for ent,value in zip(ents,values):
    #     no = int(ent.split('\t')[1][:-1])
    #     res[no] = value[:-1]
    
    # ress = []
    # for r in res:
    #     if r == '':
    #         ress.append(novalue)
    #     else:
    #         ress.append(r)
    # ress = '\n'.join(ress)
    # f = open('emb_init.txt', 'w', encoding='utf-8')
    # f.write(ress)
    # f.close()

    fent = open(r'kg//entity2id.txt','r',encoding='utf-8') 
    entities = fent.readlines()
    fent.close()
    fe = open(r'kg//emb.txt','r',encoding='utf-8') 
    embs = fe.readlines()
    fe.close()
    eidx = {}
    for item,emb in zip(entities,embs):
        word = item.split('\t')[0]
        eidx[word] = emb.strip()
    
    null = str([0.0]*20)[1:-1].replace(',',' ')
    res = [null] * (len(widx)+1)

    for word in widx:
        if word in eidx:
            index = widx[word]
            emb = eidx[word]
            res[index] = emb
        else:
            continue
    res = '\n'.join(res)
    fres = open(r'kg//w2e.txt','w',encoding='utf-8')
    fres.write(res)
    fres.close()

    set_neighbor_emb(widx)

def set_neighbor_emb(widx):
    fr = open(r'kg//triple.txt','r',encoding='utf-8')
    emap = {}
    nothave = 0
    for tt in fr:
        array = tt.split('\t')
        if len(array) != 3:  # to skip the first line in triple2id.txt
            continue
        try:
            head = widx[array[0]]
            tail = widx[array[1]]
        except KeyError as e:
            continue
        if head in emap:
            emap[head].append(tail)
        else:
            emap[head] = [tail]
        if tail in emap:
            emap[tail].append(head)
        else:
            emap[tail] = [head]
    fr.close()
    
    # for ent in emap:
    #   print(emap[ent])
    embs = loademb()
    frc = open(r'kg//w2e.txt','r',encoding='utf-8')
    nowembs = frc.readlines()
    frc.close()
    for em in emap:
        if len(emap[em]) < 20:
            index = em - 1
            try:
                aveemb = np.average(embs[emap[em]],axis=0).tolist()
                aveemb = str(aveemb)[1:-1].replace(',','')
                nowembs[index-1] = aveemb
            except:
                print(1)
        else:
            continue
    ave = open(r'kg//w2e_ave.txt','w',encoding='utf-8')
    res = [now.replace('\n','') for now in nowembs]
    res = '\n'.join(res)
    ave.write(res)
    ave.close()

def loademb():
    emb = np.loadtxt(r'kg//w2e.txt')
    return emb

def cand_rels(widx):
    rels = ['R_phone','R_cuisine','R_address','R_location','R_number','R_price',
      'R_rating','R_type','R_speciality','R_social_media','R_parking','R_public_transport']

    fc = open(r'personalized-dialog-dataset\personalized-dialog-candidates.txt','r',encoding='utf-8')
    cands = fc.readlines()
    fc.close()

    cand_rel = []
    cand_item = []
    for cidx,cand in enumerate(cands):
        cand_rel.append([0,0,0,0,0,0,0,0,0,0,0,0])
        cand_item.append(0)
        for r_idx,rel in enumerate(rels):
            if rel[1:] in cand:
                # if r_idx!=2:
                cand_rel[cidx][r_idx] = 1
                words = cand.split(' ')
                for word in words:
                    if rel[1:] in word and cand_item[cidx]==0:
                        if word[:-len(rel)][-1]=='_':
                            cand_item[cidx] =  word[:-len(rel)+1]
                        else:
                            cand_item[cidx] =  word[:-len(rel)]
                    # break
                else:
                    continue

    # output1 = tf.constant(cand_rel,dtype=tf.float32)
    output2 = []
    for item in cand_item:
        if item != 0:
            output2.append(widx[item])
        else:
            output2.append(0)
    # output2 = tf.constant(output2)
    frel = open('cand_rel.txt','w',encoding='utf-8')
    for index,rel in enumerate(cand_rel):
        if index != len(cand_rel)-1:
            frel.write(str(rel)[1:-1].replace(',',' ')+'\n')
        else:
            frel.write(str(rel)[1:-1].replace(',',' '))
    frel.close()

    fitem = open('cand_item.txt','w',encoding='utf-8')
    for index,item in enumerate(output2):
        if index != len(cand_item)-1:
            fitem.write(str(item)+'\n')
        else:
            fitem.write(str(item))
    fitem.close()

    return cand_rel,output2