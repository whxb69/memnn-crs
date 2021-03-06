from __future__ import absolute_import
from __future__ import print_function

from data_utils import *
from sklearn import metrics
from memn2n.memn2n_dialog import MemN2NDialog
from itertools import chain
from six.moves import range, reduce
import sys
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import datalook
import random
import getdata

tf.app.flags.DEFINE_string("f","","kernel")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 2, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 30, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 250, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "task id, 1 <= id <= 5")
tf.flags.DEFINE_integer("random_state", 4, "Random state.")
tf.flags.DEFINE_string("data_dir", "personalized-dialog-dataset/small/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("model_dir", "model/", "Directory containing memn2n model checkpoints")
tf.flags.DEFINE_boolean('train', True, 'if True, begin to train')
tf.flags.DEFINE_boolean('interactive', False, 'if True, interactive')
tf.flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')
tf.flags.DEFINE_boolean('save_vocab', False, 'if True, saves vocabulary')
tf.flags.DEFINE_boolean('load_vocab', False, 'if True, loads vocabulary instead of building it')
FLAGS = tf.flags.FLAGS

class chatBot(object):
    def __init__(self, data_dir, model_dir, task_id, isInteractive=True, OOV=False, memory_size=250, random_state=None,
                 batch_size=32, learning_rate=0.01, epsilon=1e-8, max_grad_norm=40.0, evaluation_interval=10, hops=3,
                 epochs=250, embedding_size=20, save_vocab=False, load_vocab=False):
        self.data_dir = data_dir
        self.task_id = task_id
        self.model_dir = model_dir
        # self.isTrain=isTrain
        self.isInteractive = isInteractive
        self.OOV = OOV
        self.memory_size = memory_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.ebatch_size = 200
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.evaluation_interval = evaluation_interval
        self.hops = hops
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.save_vocab = save_vocab
        self.load_vocab = load_vocab
        self.ent_path = './facebook/'

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        
        # 在创建session的时候把config作为参数传进去

        candidates, self.candid2indx = load_candidates(self.data_dir, self.task_id)
        self.candidates = candidates
        self.n_cand = len(candidates)
        print("Candidate Size", self.n_cand)
        self.indx2candid = dict((self.candid2indx[key], key) for key in self.candid2indx)
        # task data
        self.trainData, self.testData, self.valData = load_dialog_task(self.data_dir, self.task_id, self.candid2indx,
                                                                       self.OOV)
        data = self.trainData + self.testData + self.valData
        self.build_vocab(data, candidates, self.save_vocab, self.load_vocab)
        # set_entity_embs(self.word_idx,self.embedding_size)
        # self.candidates_vec=vectorize_candidates_sparse(candidates,self.word_idx)
        self.candidates_vec = vectorize_candidates(candidates, self.word_idx, self.candidate_sentence_size)
        self.candidates_rel,self.candidates_item,self.candidates_per = cand_rels(self.word_idx)
        self.itemattrs,self.totalitem = getdata.get_itemattrs(r'personalized-dialog-dataset//triple_reset.txt',self.word_idx)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.silence = tf.constant([[self.word_idx['<silence>']]+[0]*(self.sentence_size-1)])
        self.silence_u = tf.constant([ [self.word_idx['<silence>']]+[0]*(self.sentence_size-1) ])
        self.model = MemN2NDialog(self.batch_size, self.vocab_size, self.n_cand, self.sentence_size,self.candidate_sentence_size,
                                  self.embedding_size, self.candidates_vec, self.silence,max_dialog = self.max_dialog,
                                  session=self.sess, hops=self.hops, max_grad_norm=self.max_grad_norm, optimizer=optimizer,
                                  task_id=task_id,random_state=self.random_state,silence_u=self.silence_u)
        self.saver = tf.train.Saver(max_to_keep=50)


        # self.summary_writer = tf.train.SummaryWriter(self.model.root_dir, self.model.graph_output.graph)
        self.summary_writer = tf.summary.FileWriter(self.model.root_dir, self.model.graph_output.graph)

    def build_vocab(self, data, candidates, save=False, load=False):
        if load:
            vocab_file = open('vocab' + str(self.task_id) + '.obj', 'rb')
            vocab = pickle.load(vocab_file)
        else:
            vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for p, s, q, a in data))
            vocab |= reduce(lambda x, y: x | y, (set(list(chain.from_iterable(p)) + q) for p, s, q, a in data))
            vocab |= reduce(lambda x, y: x | y, (set(candidate) for candidate in candidates))
            vocab = sorted(vocab)
        self.rels = ['r_phone','r_cuisine','r_address','r_location','r_number','r_price',
      'r_rating','r_type','r_speciality','r_social_media','r_parking','r_public_transport']

        # self.rel_idx = {}
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        # for rel in rels:
        #     self.rel_idx[rel] = self.word_idx[rel]


        max_story_size = max(map(len, (s for _, s, _, _ in data)))
        mean_story_size = int(np.mean([len(s) for _, s, _, _ in data]))
        self.sentence_size = max(map(len, chain.from_iterable(s for _, s, _, _ in data)))
        self.max_dialog = get_max_dialog(data)
        # self.candidate_sentence_size = max(map(len, candidates))
        self.candidate_sentence_size = self.sentence_size        
        query_size = max(map(len, (q for _, _, q, _ in data)))
        self.memory_size = min(self.memory_size, max_story_size)
        self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
        self.sentence_size = max(query_size, self.sentence_size)  # for the position
        # params
        print("vocab size:", self.vocab_size)
        print("Longest sentence length", self.sentence_size)
        print("Longest candidate sentence length", self.candidate_sentence_size)
        print("Longest story length", max_story_size)
        print("Average story length", mean_story_size)

        if save:
            vocab_file = open('vocab' + str(self.task_id) + '.obj', 'wb')
            pickle.dump(vocab, vocab_file)

    def train(self):
        self.relations = [self.word_idx[rel] for rel in self.rels]
        trainP, trainS1, trainS2, trainQ, trainA= vectorize_data(self.trainData, self.word_idx, self.sentence_size,
                                                        self.batch_size, self.n_cand, self.memory_size,self.max_dialog)
        valP, valS1, valS2, valQ, valA= vectorize_data(self.valData, self.word_idx, self.sentence_size, self.batch_size,
                                                self.n_cand, self.memory_size,self.max_dialog)
        # while len(valA)%self.batch_size!=0:
        #     index = random.randint(len(valP)-10,len(valP)-1)
        #     for item in [valP, valS, valQ, valA]:
        #         item.append(item[index])

        # while len(trainA)%self.batch_size!=0:
        #     index = random.randint(len(trainP)-10,len(trainP)-1)
        #     for item in [trainP, trainS, trainQ, trainA]:
        #         item.append(item[index])
        n_train = len(trainS1)
        n_val = len(valS1)
        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        np.random.seed(self.random_state)
        batches = zip(range(0, n_train - self.batch_size, self.batch_size),
                      range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]
        best_validation_accuracy = 0

        Taccs = []
        Vaccs = []
        cost = []
        for t in range(1, self.epochs + 1):
            print('Epoch', t)
            np.random.shuffle(batches)
            total_cost = 0.0
            total_coste = 0.0
            num = 0
            _, _, triple_total = getdata.get_data(self.ent_path)
            for start, end in batches:
                p = trainP[start:end]
                s1 = trainS1[start:end]
                s2 = trainS2[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                # pos_h_batch, pos_r_batch, pos_t_batch, neg_h_batch, neg_r_batch, neg_t_batch = \
                # item,attrs = getdata.get_batch(self.ebatch_size, self.ent_path, triple_total)                
                item,attrs = getdata.getdatas(self.ebatch_size, self.itemattrs, self.totalitem)
                cost_t ,cost_e,prob,q,m= self.model.batch_fit(p, s1, s2, q, a, self.relations,item,attrs)
                                            # pos_h_batch, pos_r_batch, pos_t_batch, 
                                            # neg_h_batch, neg_r_batch, neg_t_batch)
                # if t == 11:
                #     for i in range(len(q)):
                #         if q[i][0] == 14 and num < 5:
                #             for j in range(len(s2)):
                #                 print((' ').join([w for x in s2[i][j] if x!=0 for w in self.word_idx if self.word_idx[w]==x]) + '\t' +str(prob[i][j]))
                            # num+=1
                        # break
                total_cost += cost_t
                total_coste += cost_e
            cost.append(total_cost)
            if t % 5 == 0:
                # pos_h_batch, pos_r_batch, pos_t_batch, neg_h_batch, neg_r_batch, neg_t_batch = getdata.get_batch(len(trainS), self.ent_path, triple_total)
                train_preds = self.batch_predict(trainP, trainS1, trainS2, trainQ, n_train)
                val_preds = self.batch_predict(valP, valS1, valS2, valQ, n_val)
                train_acc = metrics.accuracy_score(np.array(train_preds), trainA)
                val_acc = metrics.accuracy_score(val_preds, valA)
                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Total eCost:', total_coste)
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                print('-----------------------')
                Taccs.append(train_acc)
                Vaccs.append(val_acc)
                # write summary
                # train_acc_summary = tf.scalar_summary('task_' + str(self.task_id) + '/' + 'train_acc', tf.constant((train_acc), dtype=tf.float32))
                # val_acc_summary = tf.scalar_summary('task_' + str(self.task_id) + '/' + 'val_acc', tf.constant((val_acc), dtype=tf.float32))
                # merged_summary = tf.merge_summary([train_acc_summary, val_acc_summary])
                train_acc_summary = tf.summary.scalar('task_' + str(self.task_id) + '/' + 'train_acc',
                                                      tf.constant((train_acc), dtype=tf.float32))
                val_acc_summary = tf.summary.scalar('task_' + str(self.task_id) + '/' + 'val_acc',
                                                    tf.constant((val_acc), dtype=tf.float32))
                merged_summary = tf.summary.merge([train_acc_summary, val_acc_summary])
                summary_str = self.sess.run(merged_summary)
                self.summary_writer.add_summary(summary_str, t)
                self.summary_writer.flush()
                # datalook.attentionshow(train_prob,str(t))
                # datalook.attentionshow(trans_att,str(t)+'-sparsemax')

                if val_acc > best_validation_accuracy:
                    best_validation_accuracy = val_acc
                    self.saver.save(self.sess, self.model_dir + 'model.ckpt', global_step=t)
        datalook.figshow(self.task_id, Taccs, Vaccs)
        
    def test(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
        if self.isInteractive:
            self.interactive()
        else:
            testP, testS1, testS2, testQ, testA = vectorize_data(self.testData, self.word_idx, self.sentence_size,
                                                        self.batch_size, self.n_cand, self.memory_size,self.max_dialog)
            n_test = len(testS1)
            print("Testing Size", n_test)
            test_preds = self.batch_predict(testP, testS1, testS2, testQ, n_test)
            test_acc = metrics.accuracy_score(test_preds, testA)
            print("Testing Accurac", test_acc)

            # print(testA)
            # for pred in test_preds:
            #    print(pred, self.indx2candid[pred])

    def batch_predict(self, P, S1, S2, Q, n):
        preds = []
        num = 0 
        _, _, triple_total = getdata.get_data(self.ent_path)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            p = P[start:end]
            s1 = S1[start:end]
            s2 = S2[start:end]
            q = Q[start:end]
            # pos_h_batch, pos_r_batch, pos_t_batch, neg_h_batch, neg_r_batch, neg_t_batch = getdata.get_batch(self.ebatch_size, self.ent_path, triple_total)
            items, attrs = getdata.getdatas(self.ebatch_size, self.itemattrs, self.totalitem)
            # for item in [pos_h_batch, pos_r_batch, pos_t_batch, neg_h_batch, neg_r_batch, neg_t_batch]:
                # item = item[:len(p)]
            # pred= self.model.predict(p, s, q, self.relations, pos_h_batch, pos_r_batch, pos_t_batch, neg_h_batch, neg_r_batch, neg_t_batch)
            
            pred= self.model.predict(p, s1, s2, q, self.relations, items, attrs)
            # if num < 3:
            #     for i in range(len(pred[0])):
            #         qq = [w for x in q[i] if x!=0 for w in self.word_idx if self.word_idx[w]==x]
            #         aa = self.candidates[pred[0][i]]
            #         print('-'+(' ').join(qq).replace(" ' ","'"))
            #         print('-'+(' ').join(aa).replace(" ' ","'"))
            #     num += 1
            preds += list(pred[0])
        return preds

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':
    # time.sleep(3600)
    for taskid in [3]:
        print("Started Task:", taskid)
        model_dir = "task" + str(taskid) + "_" + FLAGS.model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        chatbot = chatBot(FLAGS.data_dir, model_dir, taskid, OOV=FLAGS.OOV, isInteractive=FLAGS.interactive,
                        batch_size=FLAGS.batch_size, memory_size=FLAGS.memory_size, epochs=FLAGS.epochs, hops=FLAGS.hops,
                        save_vocab=FLAGS.save_vocab, load_vocab=FLAGS.load_vocab, learning_rate=FLAGS.learning_rate,
                        embedding_size=FLAGS.embedding_size,random_state = FLAGS.random_state)
        # chatbot.run()
        if FLAGS.train:
            chatbot.train()
            chatbot.test()
        chatbot.close_session()
        if taskid != 5:
            print(taskid,'\t完成')
            time.sleep(2)
    # time.sleep(10)
    # os.system('shutdown -s -t ')