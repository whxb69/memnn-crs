from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
from datetime import datetime
import random
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops, dtypes, function
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from layers import CrossCompressUnit, Dense
import getdata


def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.
    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    # with tf.op_scope([t], name, "zero_nil_slot") as name:
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        # z = tf.zeros([1, s])
        return tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], axis = 0,  name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    """
    # with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        if t is None:
            return t
        else:
            t = tf.convert_to_tensor(t,dtype= tf.float32, name="t")
            gn = tf.random_normal(tf.shape(t), stddev=stddev)
            return tf.add(t, gn, name=name)

class MemN2NDialog(object):
    """End-To-End Memory Network."""
    def __init__(self, batch_size, vocab_size, candidates_size, sentence_size,candidates_sentence_size, embedding_size,
        candidates_vec,silence=None, silence_u=None,
        # candidates_rel,
        # candidates_item,
        max_dialog=60,
        hops=3,
        max_grad_norm=40.0,
        nonlin=None,
        initializer=tf.random_normal_initializer(stddev=0.1),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
        session=tf.Session(),
        name='MemN2N',
        task_id=1,
        random_state=None):
        """Creates an End-To-End Memory Network
        Args:
            batch_size: The size of the batch.
            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.
            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).
            candidates_size: The size of candidates
            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).
            embedding_size: The size of the word embedding.
            candidates_vec: The numpy array of candidates encoding.
            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.
            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.
            nonlin: Non-linearity. Defaults to `None`.
            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.
            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.
            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.
            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.
            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self.task_id = task_id
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._candidates_size = candidates_size
        self._sentence_size = sentence_size
        self._candidates_sentence_size = candidates_sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._opt_e = tf.train.AdamOptimizer(learning_rate=0.001)
        self._name = name
        self._candidates=candidates_vec
        # self._candidates_rel=candidates_rel
        # self._candidates_item=candidates_item
        self.h = 4
        self._head_num = 4
        
        self.L1_flag = True # 在loss中是否加入L1正则化
        self.hidden_size = self._embedding_size # 实体与关系的词向量长度,在知识图谱中关系数量会远远小于实体个数，所以该超参调整不能太大
        self.sizeE = self._embedding_size # 实体词向量长度，仅在TransR中使用
        self.sizeR = self._embedding_size # 关系词向量长度，仅在TransR中使用，关系向量长度与实体向量长度不宜差距太大
        # self.ebatch_size = 100 # 每个批度输入的三元组个数
        self.margin = 1.0 # 合页损失函数中的标准化项
        self.relation_total = 0 # 知识图谱关系数，不需要修改，后续从输入数据中获得
        self.entity_total = 0 # 知识图谱实体数，不需要修改，后续从输入数据中获得
        self.triple_total = 0 # 知识图谱三元组个数，不需要修改，后续从数据输入中获得
        self.flie_path = 'facebook/' # 存放训练文件的路径，该路径下应该有训练时需要的三个文件，entity2id,relation2id,triple
        self.relation_total, self.entity_total, self.triple_total = getdata.get_data(self.flie_path)        

        self.max_dialog = max_dialog
        self._silence = silence
        self._silence_u = silence_u

        self.rel_num = 12
        tf.set_random_seed(random_state)

        self._build_inputs()
        self._build_vars()
        
        # define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = 'Summary\\'+"%s_%s_%s_%s/" % ('task', str(task_id),'summary_output', timestamp)
        
        
        # cross entropy
        logits,eloss,prob,q,m = self._inference(self._profile, self._stories1, self._stories2, self._queries, self._relation)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = self._answers, name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")
        
        # t = tf.expand_dims(tf.stack([cross_entropy_sum,eloss]),1)
        # gama = tf.reduce_sum(tf.nn.sigmoid(tf.matmul(self.W_e,t)))

        # loss op
        loss_op = cross_entropy_sum
        loss_op_l2 = tf.nn.l2_loss(self.vars_rs[0])
        for var in self.vars_rs[1:]:
            loss_op_l2+=tf.nn.l2_loss(var)
        loss_op = loss_op + loss_op_l2 * 1e-7
        # gradient pipeline

        #设置参数列表
        vlist = []
        [vlist.append(var) for var in tf.trainable_variables() if var.name != 'MemN2N/B:0']
        grads_and_vars = self._opt.compute_gradients(loss_op,var_list=vlist) 
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if g is not None:
                g = tf.clip_by_value(g,-15.,15.)
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars,name="train_op")
        etrain_op = self._opt_e.minimize(eloss)

        # predict ops
        # predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_op = prob
        # epredict_op = epredict
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.eloss_op = eloss
        self.predict_op = predict_op
        # self.epredict_op = epredict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op
        self.etrain_op = etrain_op
        self.prob = prob
        self.q = q
        self.m = m


        self.graph_output = self.loss_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)
        

    def _build_inputs(self):
        self._profile = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="profile")
        self._stories1 = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
        self._stories2 = tf.placeholder(tf.int32, [None, self.max_dialog, self._sentence_size], name="stories")
        self._relation = tf.placeholder(tf.int32, [self.rel_num], name="relation")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answers")

        self.items = tf.placeholder(tf.int32, [None])
        self.attrs = tf.placeholder(tf.int32, [None,12,2])
    
    def _build_vars(self):
        self.profile_num = 4
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.A = tf.Variable(A, name="A")

            B = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.B = tf.Variable(B, name="B")
    
            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            self.H_m = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H_m")
        
            W = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            # W = tf.concat([ nil_word_slot, tf.Variable(np.loadtxt(r'.\\facebook\\pre_train.txt'),dtype=tf.float32) ], axis=0)
            self.W = tf.Variable(W, name="W")

            We = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.We = tf.Variable(We, name="We")

            self.W_o = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W_o")
            self.W_oe = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W_oe")

            self.P_p = tf.Variable(self._init([self._embedding_size, self.rel_num]), name="P_p")
            self.P_m = tf.Variable(self._init([self._embedding_size, self.rel_num]), name="P_m")
            self.P_i = tf.Variable(self._init([self._embedding_size, self.rel_num]), name="P_i")
            self.P = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="P")
            self.P_pro = tf.Variable(self._init([4]), name="P_pro")            

            self.Wm = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="Wm")
            self.Wp = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="Wp")            
            self.Wk = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="Wk")            
            
            self.R1 = tf.Variable(self._init([self._embedding_size, 5*self._embedding_size]), name="R")            
            self.R2 = tf.Variable(self._init([5*self._embedding_size, self._embedding_size]), name="R")            
            self.Rc = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="Rc")            
            self.b1 = tf.Variable(self._init([self._embedding_size*5]), name="b1")            
            self.b2 = tf.Variable(self._init([self._embedding_size]), name="b2")            
            # nil_word_slot_e = tf.zeros([1, self._embedding_size])
        
        self.cand_rel = tf.Variable(np.loadtxt(r'cand_rel.txt'), name="c_rel", dtype=tf.float32)
        self.cand_item = tf.Variable(np.loadtxt(r'cand_item.txt'), name="c_item", dtype=tf.float32)
        self.cand_per = tf.Variable(np.loadtxt(r'cand_per.txt'), name="c_per", dtype=tf.int32)

        self._nil_vars = set([self.A.name,self.W.name])
        self.vars_rs = []
        [self.vars_rs.append(var) for var in [self.A, self.W, self.W_o]]
    
    def normalize(self, inputs, epsilon = 1e-8, scope="ln", reuse=None):
        '''Applies layer normalization.
        Args:
        inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        
        Returns:
        A tensor with the same shape and data dtype as `inputs`.
        '''
        
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
            
        return outputs

    def self_attention(self, t, T, n, h, kv=None):
        t_reshape = tf.reshape(t,[-1, self._embedding_size])
        kv_reshape = t_reshape
        if kv is not None:
            kv_reshape = tf.reshape(kv,[-1, self._embedding_size])
        if T == 'queries':
            exec('self.W_q_q_' + str(n) +'= tf.get_variable(name="W_q_q_'+ str(n) +'",initializer=self._init([self._embedding_size, int(self._embedding_size/h)]))')
            exec('self.W_q_k_' + str(n) +'= tf.get_variable(name="W_q_k_'+ str(n) +'",initializer=self._init([self._embedding_size, int(self._embedding_size/h)]))')
            exec('self.W_q_v_' + str(n) +'= tf.get_variable(name="W_q_v_'+ str(n) +'",initializer=self._init([self._embedding_size, int(self._embedding_size/h)]))')
            dim = tf.shape(t)[1]
            if kv is not None:
                dim = tf.shape(kv)[1]
            query = tf.reshape(tf.matmul(t_reshape, eval('self.W_q_q_'+str(n))),[-1,tf.shape(t)[1],int(self._embedding_size/h)])
            key = tf.reshape(tf.matmul(kv_reshape, eval('self.W_q_k_'+str(n))),[-1,dim,int(self._embedding_size/h)])
            value = tf.reshape(tf.matmul(kv_reshape, eval('self.W_q_v_'+str(n))),[-1,dim,int(self._embedding_size/h)])
            output = tf.matmul(query,tf.transpose(key,[0,2,1]))/tf.sqrt(tf.cast(self._embedding_size,dtype=tf.float32))
        elif T == 'stories':
            exec('self.W_s_q_' + str(n) +'= tf.get_variable(name="W_s_q_'+str(n)+r'",initializer=self._init([self._embedding_size, int(self._embedding_size/h)]))')
            exec('self.W_s_k_' + str(n) +'= tf.get_variable(name="W_s_k_'+str(n)+r'",initializer=self._init([self._embedding_size, int(self._embedding_size/h)]))')
            exec('self.W_s_v_' + str(n) +'= tf.get_variable(name="W_s_v_'+str(n)+r'",initializer=self._init([self._embedding_size, int(self._embedding_size/h)]))')
            # t_reshape = tf.reshape(t,[-1, self._embedding_size])
            query = tf.reshape(tf.matmul(t_reshape, eval('self.W_s_q_'+str(n))),[-1,tf.shape(t)[2],int(self._embedding_size/h)])
            key = tf.reshape(tf.matmul(kv_reshape, eval('self.W_s_k_'+str(n))),[-1,tf.shape(t)[2],int(self._embedding_size/h)])
            value = tf.reshape(tf.matmul(kv_reshape, eval('self.W_s_v_'+str(n))),[-1,tf.shape(t)[2],int(self._embedding_size/h)])
            output = tf.matmul(query,tf.transpose(key,[0,2,1]))/tf.sqrt(tf.cast(self._embedding_size,dtype=tf.float32))
    
            #对key进行mask
            t_r = tf.reshape(t,[-1,tf.shape(t)[2],self._embedding_size])
            key_masks = tf.sign(tf.abs(tf.reduce_sum(t_r, axis=-1)))
            key_masks = tf.tile(tf.expand_dims(key_masks,1), [1, self._sentence_size, 1]) # (h*N, T_q, T_k)
            paddings = tf.ones_like(output)*(-2)
            output = tf.where(tf.equal(key_masks, 0), paddings, output) # (h*N, T_q, T_k)
            
            #Future blinding
            # diag_vals = tf.ones_like(output[0, :, :]) # (T_q, T_k)
            # tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            # masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(output)[0], 1, 1]) # (h*N, T_q, T_k)
            # paddings = tf.ones_like(masks)*(-2)
            # output = tf.where(tf.equal(masks, 0), paddings, output) # (h*N, T_q, T_k)

            output = tf.nn.softmax(output)
            
            # #对query进行mask
            query_masks = tf.sign(tf.abs(tf.reduce_sum(t_r, axis=-1))) # (N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, self._sentence_size]) # (h*N, T_q, T_k)
            output *= query_masks # broadcasting. (N, T_q, C)

        if T == 'queries':
            output = tf.nn.softmax(output)
            # outputr = tf.reshape(output,[-1,self._sentence_size])
            # outputr = self.sparsemax(outputr)
            # output = tf.reshape(output,[-1,tf.shape(output)[1],self._sentence_size])
        output = tf.layers.dropout(output,0.2)
        
        output_log = tf.reduce_mean(output,axis=0)
        output = tf.matmul(output,value)
        return [tf.layers.dropout(output,0.2),output_log]
    
    def multi_head(self, inputs, T, sign=None, h=4, kv=None):
        attentions = []
        logs = []
        if kv is None:
            for n in range(h):
                attentions.append(self.self_attention(inputs, T, n, h, kv)[0])
                logs.append(self.self_attention(inputs, T, n, h, kv)[1])
        else:
            for n in range(h):
                attentions.append(self.self_attention(inputs, T, sign+str(n), h, kv)[0])
                logs.append(self.self_attention(inputs, T, sign+str(n), h, kv)[1])
        MH = tf.concat(attentions,axis=2)
        MH_reshape = tf.reshape(MH,[-1, self._embedding_size])
        if kv is None:
            output = tf.matmul(MH_reshape,self.W_o)
        else:
            output = tf.matmul(MH_reshape,self.W_oe)
        dims = [tf.shape(inputs)[i] for i in range(tf.shape(inputs).shape.dims[0].value)]
        logs = tf.reduce_mean(logs,axis=0)
        out = tf.reshape(output,dims) + inputs
        return [out,logs]
    
    def multi_head1(self, inputs, keys=None ,num_heads=4):
        queries = inputs
        Q = tf.layers.dense(inputs, self._embedding_size, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, self._embedding_size, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(inputs, self._embedding_size, activation=tf.nn.relu) # (N, T_k, C)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        keyMask = True
        if keyMask:
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
    
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
        
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        queryMask = True
        if queryMask:
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs *= query_masks # broadcasting. (N, T_q, C)
        
        outputs = tf.layers.dropout(outputs, rate=0.2)
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
        # Residual connection
        # outputs += Q
        # Normalize
        # outputs = self.normalize(outputs) # (N, T_q, C)
        return outputs


    def FFN(self,inputs,num_units=[80, 20]):
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = self.normalize(outputs)
        # Residual connection    
        return tf.layers.dropout(outputs,0.2)

    def position_encoding(self,inputs, num_units=20,scope="positional_encoding"):
        B, T, D = inputs.get_shape().as_list()
        embedding_size = D
        sentence_size = T
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = T+1
        le = D+1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        # Make position encoding of time words identity to avoid modifying them 
        # encoding[:, -1] = 1.0
        encoding = np.flipud(encoding)

        outputs = tf.tile(tf.expand_dims(tf.transpose(encoding),0),[tf.shape(inputs)[0],1,1])

        outputs = tf.where(tf.equal(inputs,0),inputs,outputs)

        return outputs

    def calc(self, e, n):
        norm = tf.nn.l2_normalize(n, 1)
        return e - tf.reduce_sum(e * norm, 1, keepdims=True) * norm

    def transh(self):
        with tf.name_scope("embedding"):

            key_a = tf.nn.embedding_lookup(self.A, self.items)
            key_b = tf.nn.embedding_lookup(self.B, self.items)
            # key_c = tf.nn.embedding_lookup(self.W, self.items)
            
            values_a = tf.nn.embedding_lookup(self.A, self.attrs)
            values_a = tf.reduce_sum(values_a,2)

            # values_c = tf.nn.embedding_lookup(self.W, self.attrs)
            # values_c = tf.reduce_sum(values_c,2)

            # key_b = tf.nn.tanh(tf.matmul(key_b,self.R1) + self.b1)
            # key_b = tf.matmul(key_b,self.R2) + self.b2
            # newkey_c = tf.matmul(key_c,self.Rc)
            
            con_a = tf.reduce_sum(tf.concat([tf.expand_dims(key_a,1),values_a],1),1)
            # con_c = tf.reduce_sum(tf.concat([tf.expand_dims(key_c,1),values_c],1),1)

            # pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)

            # self.scores_kge = tf.nn.sigmoid(tf.reduce_sum(pos_t_e * self.tail_pred, axis=1))
            # eloss = tf.reduce_mean(
                # tf.sqrt(tf.reduce_sum(tf.square(newkey_m - con_m), axis=1)))

            # eloss_c = tf.reduce_mean(
                # tf.sqrt(tf.reduce_sum(tf.square(newkey_c - con_c), axis=1) / self._embedding_size))
            key_b = tf.reduce_sum(abs(key_b),1,keepdims=True)
            con_a = tf.reduce_sum(abs(con_a),1,keepdims=True)
            eloss = tf.reduce_sum(tf.maximum(con_a - key_b + 1,0)) #+ eloss_c
            return eloss
    
    def text_cnn(self,inputs):
        num_filters = 20
        filter_size = 3
        filter_shape = [filter_size,self._embedding_size,num_filters]
        W_c = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W_c")
        b_c = tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b_c")
        conv = tf.nn.conv1d(
            inputs,W_c,stride=1,padding="SAME",
            name="conv"
        )
        h = tf.nn.relu(tf.nn.bias_add(conv,b_c),name="relu")
        hs = tf.expand_dims(h,-1)
        pooled = tf.nn.max_pool(
                hs,ksize=[1,self._sentence_size - filter_size +1,1,1],
                strides=[1,1,1,1],padding="SAME",name="pool"
        )
        output = tf.squeeze(pooled, -1)  
        return output
    
    def _length(self, seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def _inference(self, profile, stories1, stories2, queries, relations):
        with tf.variable_scope(self._name, reuse = tf.AUTO_REUSE):
            q_emb = tf.nn.embedding_lookup(self.A, queries)
            silences = tf.tile(tf.to_float(self._silence),[tf.shape(queries)[0],1])
            slc_flag = tf.equal(tf.reduce_sum(tf.to_float(queries),-1,keepdims=True),tf.reduce_sum(silences,-1,keepdims=True))
            slc_flag = tf.sign(tf.to_float(slc_flag))
            
            q_emb = self.multi_head1(q_emb,q_emb)
            u_0 = tf.reduce_sum(q_emb, 1)
            u = [u_0]
            u_profile = [u_0]
            
            eloss = self.transh()
            candidates_emb = tf.nn.embedding_lookup(self.W, self._candidates)

            candidates_emb_sum=tf.reduce_sum(candidates_emb,1)
            # cands = tf.tile(tf.expand_dims(self._candidates,0),[tf.shape(u_0)[0],1,1])
            for _ in range(self._hops):
                # stories = tf.concat([stories1,stories2],1)
                m_emb1 = tf.reduce_sum(tf.nn.embedding_lookup(self.A, stories1),2)
                attrs = m_emb1

                keys = tf.batch_gather(stories1, tf.tile(tf.expand_dims(tf.constant([[0]]), 0), [tf.shape(m_emb1)[0],tf.shape(m_emb1)[1],1]))
                padd = tf.ones_like(keys) * (-1)
                keys_ = tf.where(tf.equal(keys,0),padd,keys)
                flag_list = []
                # flags = tf.to_float(tf.equal(tf.to_float(self._candidates), tf.expand_dims(tf.expand_dims(tf.to_float(keys_),-1),-1)))
                # flags = tf.sign(tf.reduce_sum(flags, [1,2,4]))
                for key in tf.split(keys_,24,1):
                    flag = tf.to_float(tf.equal(tf.to_float(self._candidates), tf.expand_dims(tf.expand_dims(tf.to_float(key),-1),-1)))      
                    flag = tf.reduce_sum(flag, [2,4])
                    flag = tf.reduce_sum(flag, 1, keepdims=True)
                    flag_list.append(flag)
                flags = tf.sign(tf.reduce_sum(tf.concat(flag_list,1),1))
                keys = tf.reduce_sum(tf.nn.embedding_lookup(self.W, keys),2)
                
                rels = tf.batch_gather(stories1, tf.tile(tf.expand_dims(tf.constant([[1]]), 0), [tf.shape(m_emb1)[0],tf.shape(m_emb1)[1],1]))
                rels = tf.reduce_sum(tf.nn.embedding_lookup(self.A, rels),2)
                
                varange = [x for x in range(2,self._sentence_size)]
                values = tf.batch_gather(stories1, tf.tile(tf.expand_dims(tf.constant([varange]), 0), [tf.shape(m_emb1)[0],tf.shape(m_emb1)[1],1]))
                values = tf.reduce_sum(tf.nn.embedding_lookup(self.A, values),2)

                m_emb2 = tf.nn.embedding_lookup(self.A, stories2)
                m_emb2 = tf.reduce_sum(m_emb2, 2)
                m_emb2 *= self.position_encoding(m_emb2)
                m_emb2 = self.multi_head1(m_emb2, m_emb2)
                # history = self.text_cnn(m_emb2)
                history = m_emb2

                m_emb_profile = tf.nn.embedding_lookup(self.A, profile)
                m_profile = tf.reduce_sum(m_emb_profile, 2)

                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                # u_temps = tf.tile(u_temp,[1,tf.shape(m)[1],1])
                u_temp_profile = tf.transpose(tf.expand_dims(u_profile[-1], -1), [0, 2, 1])
                
                # dot_m = tf.concat([m,u_temps,m-u_temps,m * u_temps],-1)
                # dot_m = tf.layers.dense(dot_m,self._embedding_size*4,activation=tf.nn.sigmoid)
                # dot_m = tf.layers.dense(dot_m,self._embedding_size)                
                dotted_a = tf.reduce_sum(attrs*u_temp,2)
                padd = tf.ones_like(dotted_a) * (-32)
                dotted_a = tf.where(tf.equal(tf.reduce_sum(m_emb1,-1),0), padd,dotted_a)

                m = tf.concat([m_emb1,m_emb2],1)
                dotted_h = tf.reduce_sum(history*u_temp,2)
                dotted_hh = tf.reduce_sum(m*u_temp_profile,2)
                padd = tf.ones_like(dotted_h) * (-32)
                dotted_h = tf.where(tf.equal(tf.reduce_sum(m_emb2,-1),0), padd,dotted_h)
                padd = tf.ones_like(dotted_hh) * (-32)
                dotted_hh = tf.where(tf.equal(tf.reduce_sum(m,-1),0), padd,dotted_hh)


                dotted_profile = tf.reduce_sum(m_profile * u_temp_profile, 2)

                # # # Calculate probabilities
                probs_a = tf.nn.softmax(dotted_a)
                probs_h = tf.nn.softmax(dotted_h)
                probs_hh = tf.nn.softmax(dotted_hh)
                # flag_profile = tf.sign(tf.nn.relu(dotted_profile))
                probs_profile = tf.nn.softmax(dotted_profile)
                # probs_profile = tf.multiply(weight_profile,flag_profile) 
                
                probs_temp_a = tf.transpose(tf.expand_dims(probs_a, -1), [0, 2, 1])
                probs_temp_h = tf.transpose(tf.expand_dims(probs_h, -1), [0, 2, 1])
                probs_temp_hh = tf.transpose(tf.expand_dims(probs_hh, -1), [0, 2, 1])
                probs_temp_profile = tf.transpose(tf.expand_dims(probs_profile, -1), [0, 2, 1])

                c_temp_a = tf.transpose(attrs, [0, 2, 1])
                c_temp_h = tf.transpose(history, [0, 2, 1])
                c_temp_hh = tf.transpose(m, [0, 2, 1])
                c_temp_profile = tf.transpose(m_profile, [0, 2, 1])

                o_k_a = tf.reduce_sum(c_temp_a * probs_temp_a, 2)
                o_k_h = tf.reduce_sum(c_temp_h * probs_temp_h, 2)
                o_k_hh = tf.reduce_sum(c_temp_hh * probs_temp_hh, 2)
                o_k_h = tf.matmul(o_k_h, self.H_m) + o_k_hh
                o_k_profile = tf.reduce_sum(c_temp_profile * probs_temp_profile, 2)


                u_k = tf.matmul(u[-1], self.H) + o_k_h #+ o_keys# + m_m
                u_k_profile = tf.matmul(u_profile[-1], self.H) + o_k_profile# + m_p

                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)
                    u_k_profile = self._nonlin(u_k_profile)
                
                u.append(u_k)
                u_profile.append(u_k_profile)

            u_final = tf.add(u_k, u_k_profile)

            # dot_r = tf.reduce_sum(tf.matmul(tf.expand_dims(u_final, 1), rels, transpose_b = True),1)
            # padd = tf.ones_like(dot_r) * (-32)
            # dot_r = tf.where(tf.equal(tf.reduce_sum(rels,-1),0), padd,dot_r)
            # prob_r = tf.nn.softmax(dot_r)
            
            dot_v = tf.reduce_sum(tf.matmul(tf.expand_dims(o_k_profile, 1), values, transpose_b = True),1)
            padd = tf.ones_like(dot_v) * (-32)
            dot_v = tf.where(tf.equal(tf.reduce_sum(values,-1),0), padd,dot_v)
            # dot_v *= prob_r
            prob_v = tf.expand_dims(tf.nn.softmax(dot_v), -1)
            o_keys = tf.layers.dense(tf.reduce_sum(keys * prob_v, 1), self._embedding_size) * slc_flag

            if self._nonlin:
                u_final = self._nonlin(u_final)
            
            if self.task_id in [4]:
                rel_emb = tf.nn.embedding_lookup(self.A,relations)
                dotted_rel = tf.matmul(u_final,rel_emb,transpose_b=True)
                rel_weight = tf.nn.softmax(dotted_rel)
                v_p = tf.nn.relu(tf.matmul(tf.reduce_sum(m_profile,1),self.P_p))
                # v_p = self.gelu(tf.matmul(tf.reduce_sum(m_profile,1),self.P_p))
                v_p = v_p*rel_weight
                v_m = tf.nn.relu(tf.matmul(tf.reduce_sum(m,1),self.P_m))
                # v_m = self.gelu(tf.matmul(tf.reduce_sum(m,1),self.P_m))
                v_m = v_m*rel_weight
                # candidates_item_emb = tf.nn.embedding_lookup(self.A,self._candidates_item)
                cand_item_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.cand_item,1),1),1)
                #下面两行因数据输入变动可能有问题
                find_item = tf.equal(tf.to_float(stories1), cand_item_exp)
                find_item_r = tf.reshape(find_item,[tf.shape(candidates_emb)[0], tf.shape(m_emb1)[0], -1])
                item_res = tf.sign(tf.reduce_sum(tf.to_float(find_item_r),-1))

                bias_p = tf.matmul(v_p, self.cand_rel, transpose_b=True)
                bias_m = tf.matmul(v_m, self.cand_rel, transpose_b=True)
                item_res_trans = tf.transpose(item_res)
                bias_p_final = tf.multiply(bias_p, item_res_trans) 
                bias_m_final = tf.multiply(bias_m, item_res_trans) 
                result = tf.matmul(u_k, tf.transpose(candidates_emb_sum))
                final_res = result + bias_p_final + bias_m_final
                # final_res = result + bias_p + bias_m
            else:
                # pa_prob = tf.nn.softmax(tf.reduce_sum(tf.matmul(attrs, m_profile, transpose_b=True), -1))
                # pa_score = tf.reduce_sum(tf.expand_dims(pa_prob,-1) * attrs ,1) * slc_flag
                
                # a_score = tf.reduce_sum(self.multi_head1(attrs, attrs), 1) * slc_flag
                
                # score_pa = tf.expand_dims(tf.nn.relu(tf.matmul(pa_score, tf.transpose(candidates_emb_sum))), 1)
                # score_a = tf.expand_dims(tf.nn.relu(tf.matmul(a_score, tf.transpose(candidates_emb_sum))), 1)

                # pa_d = tf.layers.dense(pa_score, 1)
                # a_d = tf.layers.dense(a_score, 1)
                # att = tf.expand_dims(tf.nn.softmax(tf.concat([pa_d,a_d], -1)), -1)
                # score = tf.reduce_sum(tf.concat([score_pa, score_a], 1) * att,1)
                
                # attrs = tf.nn.embedding_lookup(self.A, self.attrs)
                # attrs = tf.reduce_sum(tf.reduce_sum(attrs, 2), 1)
                # tf.reduce_sum(history, 1) * attrs

                silences = tf.to_float(tf.equal(tf.reduce_sum(tf.to_float(stories2),-1), tf.reduce_sum(tf.to_float(self._silence_u),-1)))
                pos = tf.reduce_sum(silences,-1,keepdims=True)

                # u_d = tf.layers.dense(u_final, 1, tf.nn.tanh)
                # o_d = tf.layers.dense(o_keys, 1, tf.nn.tanh)
                # att = tf.expand_dims(tf.nn.softmax(tf.concat([u_d,o_d], -1)), -1)
                # u_final = tf.reduce_sum(tf.concat([tf.expand_dims(u_final, 1), tf.expand_dims(o_keys, 1)], 1) * att,1)

                score = tf.nn.relu(tf.matmul(o_keys, tf.transpose(candidates_emb_sum)))
                basic = tf.ones_like(flags)*(1-slc_flag)
                flags = tf.where(tf.equal(basic,0),flags,basic)

                rk = tf.layers.dense(tf.concat([tf.reduce_sum(history, 1), slc_flag], 1), 1, tf.nn.sigmoid)
                padd = tf.ones_like(rk)
                rk = tf.where(tf.equal(slc_flag, 0), padd, rk)
                score *= (2-rk)
                final_res = tf.matmul(u_final, tf.transpose(candidates_emb_sum)) +score*flags
                # final_res = tf.matmul(u_final, tf.transpose(candidates_emb_sum))  +score*flags
                
                
                sort_res = tf.nn.top_k(final_res,tf.shape(final_res)[1]).indices
                no = tf.to_float(tf.expand_dims(tf.range(tf.shape(m)[0]),1))
                no = tf.concat([no,pos],1)
                result_s = tf.gather_nd(sort_res,tf.cast(no,tf.int32))
                result = tf.argmax(final_res,output_type=tf.int32,axis=1)
                res = tf.where(tf.equal(tf.reduce_sum(slc_flag,1),0),result,result_s)
                

            return final_res, eloss, res, o_keys, o_keys
            # logits=tf.matmul(u_k, self.W)
            # return tf.transpose(tf.sparse_tensor_dense_matmul(self._candidates,tf.transpose(logits))),prob_log, trans_att
    def gelu(self, input_tensor):
	    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
	    return input_tensor*cdf

    def batch_fit(self, profile, stories1, stories2, queries, answers, relations,
                items,attrs):
        """Runs the training algorithm over the passed batch
        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)
        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._profile: profile, self._stories1: stories1, 
                    self._queries: queries, self._answers: answers,
                    self._relation: relations,self._stories2: stories2,
                    self.items: items, self.attrs: attrs}
        loss, eloss, _, _,prob,q,m = self._sess.run([self.loss_op, self.eloss_op, self.train_op, self.etrain_op,self.prob,self.q,self.m], feed_dict=feed_dict)
        return loss,eloss,prob,q,m

    def predict(self, profile, stories1, stories2, queries, relations,
            items, attrs):
        """Predicts answers as one-hot encoding.
        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._profile: profile, self._stories1: stories1,
                    self._queries: queries,self._relation:relations,
                    self.items: items, self.attrs: attrs,self._stories2:stories2}
        return self._sess.run([self.predict_op], feed_dict=feed_dict)
    
    # def add_global_fun(self):
    #     self._sess.run([self.add_global])
    
