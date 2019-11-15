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
# from layers import CrossCompressUnit, Dense
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
    def __init__(self, batch_size, vocab_size, candidates_size, sentence_size, embedding_size,
        candidates_vec,
        # candidates_rel,
        # candidates_item,
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

        self.rel_num = 12
        tf.set_random_seed(random_state)

        self._build_inputs()
        self._build_vars()
        
        # define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = 'Summary\\'+"%s_%s_%s_%s/" % ('task', str(task_id),'summary_output', timestamp)
        
        
        # cross entropy
        logits,eloss = self._inference(self._profile, self._stories, self._queries, self._relation)
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
        [vlist.append(var) for var in tf.trainable_variables() if var.name != 'MemN2N/We:0']
        grads_and_vars = self._opt.compute_gradients(loss_op) 
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
        predict_op = tf.argmax(logits, 1, name="predict_op")
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
        # self.k = k


        self.graph_output = self.loss_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)
        

    def _build_inputs(self):
        self._profile = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="profile")
        self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
        self._relation = tf.placeholder(tf.int32, [self.rel_num], name="relation")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answers")

        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])
        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])
    
    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.A = tf.Variable(A, name="A")
    
            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            self.H_p = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H_p")
        
            W = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.W = tf.Variable(W, name="W")

            We = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.We = tf.Variable(We, name="We")

            self.W_o = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W_o")
            self.W_oe = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W_oe")

            self.P_p = tf.Variable(self._init([self._embedding_size, self.rel_num]), name="P_p")
            self.P_m = tf.Variable(self._init([self._embedding_size, self.rel_num]), name="P_m")
            self.P_i = tf.Variable(self._init([self._embedding_size, self.rel_num]), name="P_i")
            self.P = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="P")
            
            self.Wm = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="Wm")
            self.Wp = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="Wp")            
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
    
    def multi_head1(self, inputs, num_heads=2):
        queries = keys = inputs
        Q = tf.layers.dense(inputs, self._embedding_size, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(inputs, self._embedding_size, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(inputs, self._embedding_size, activation=tf.nn.relu) # (N, T_k, C)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        keyMask = False
        if keyMask:
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
    
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
        
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        queryMask = False
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
        outputs += queries
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
        return tf.layers.dropout(outputs,0.1)

    def position_encoding(self,inputs,type, num_units=20,scope="positional_encoding"):
        # inputs_r = tf.reshape(inputs,[-1,tf.shape(inputs)[1]])
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if type == 'query':
                position_ind = tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1])
            else:
                position_ind = tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[2]), 0), [tf.shape(inputs)[1], 1])
                position_ind = tf.tile(tf.expand_dims(position_ind, 0), [tf.shape(inputs)[0], 1, 1])


            # First part of the PE function: sin and cos argument
            # position_enc = np.array([
            #     [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            #     for pos in range(T)])

            # Second part, apply the cosine to even columns and sin to odds.
            # position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            # position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[self._sentence_size, self._embedding_size],
                                       initializer=tf.contrib.layers.xavier_initializer())
            outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
            # outputs = tf.where(tf.equal(inputs, 0), inputs, tf.to_float(outputs))
            # outputs = tf.reshape(outputs,[-1,self._sentence_size, self._embedding_size])
            return tf.to_float(outputs)

    def sparsemax(self, logits):
        obs = array_ops.shape(logits)[0]
        dims = array_ops.shape(logits)[1]

        # z = logits - math_ops.reduce_mean(logits, axis=1)[:, array_ops.newaxis]
        z = logits

        # sort z
        z_sorted, _ = nn.top_k(z, k=dims)

        # calculate k(z)
        z_cumsum = math_ops.cumsum(z_sorted, axis=1)
        k = math_ops.range(
            1, math_ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype
        )
        z_check = 1 + k * z_sorted > z_cumsum
        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.float32), axis=1)

        # calculate tau(z)
        indices = array_ops.stack([tf.to_float(math_ops.range(0, obs)), k_z - 1.0], axis=1)
        tau_sum = array_ops.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / math_ops.cast(k_z, logits.dtype)

        # calculate p
        return math_ops.maximum(
            math_ops.cast(0, logits.dtype),
            z - tau_z[:, array_ops.newaxis]
        )

    def calc(self, e, n):
        norm = tf.nn.l2_normalize(n, 1)
        return e - tf.reduce_sum(e * norm, 1, keepdims=True) * norm

    def transh(self):
        with tf.name_scope("embedding"):
            # self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[self.entity_total, self.hidden_size],
            #                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[self.relation_total, self.hidden_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.normal_vector = tf.get_variable(name="normal_vector", shape=[self.relation_total, self.hidden_size],
                                                 initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            pos_h_e = tf.nn.embedding_lookup(self.We, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.We, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)

            neg_h_e = tf.nn.embedding_lookup(self.We, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.We, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            pos_norm = tf.nn.embedding_lookup(self.normal_vector, self.pos_r)
            neg_norm = tf.nn.embedding_lookup(self.normal_vector, self.neg_r)

            pos_h_e = self.calc(pos_h_e, pos_norm)
            pos_t_e = self.calc(pos_t_e, pos_norm)
            neg_h_e = self.calc(neg_h_e, neg_norm)
            neg_t_e = self.calc(neg_t_e, neg_norm)

        if self.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
            # self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
            self.predict = pos

        with tf.name_scope("output"):
            eloss = tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0))
        
        return eloss

    def _inference(self, profile, stories, queries, relations):
        with tf.variable_scope(self._name, reuse = tf.AUTO_REUSE):
            q_emb = tf.nn.embedding_lookup(self.A, queries)
            # q_emb *= self.position_encoding(queries,'query')
            u_0 = self.multi_head1(q_emb)
            u_0 = tf.reduce_sum(u_0, 1)
            u = [u_0]
            u_profile = [u_0]
            c_his = []
            
            eloss = self.transh()
            candidates_emb = tf.nn.embedding_lookup(self.W, self._candidates)
            candidates_emb_sum=tf.reduce_sum(candidates_emb,1)
            cands = tf.tile(tf.expand_dims(candidates_emb_sum,0),[tf.shape(u_0)[0],1,1])
            for count in range(self._hops):
                m_emb = tf.nn.embedding_lookup(self.A, stories)                    
                # m_emb *= self.position_encoding(stories,'story')
                
                m_emb_profile = tf.nn.embedding_lookup(self.A, profile)                
                # if count == 0:
                #     m,trans_att = self.multi_head(m_emb,'stories')
                # else: 
                #     m,trans_att = self.multi_head(f[-1],'stories')
                # m = self.FFN(m)
                # f.append(m)
                m = tf.reduce_sum(m_emb, 2)
                
                # m_profile = self.multi_head(m_emb_profile,'stories')[0]
                m_profile = tf.reduce_sum(m_emb_profile, 2)
                m_profile = self.multi_head1(m_profile)

                m = self.multi_head1(m)
                #Dynamic Explainable Recommendation based on Neural Attentive Models                                                                                                                                                                                                                                             
                p_tile = tf.tile(tf.reduce_sum(m_profile,1,keepdims=True),[1,tf.shape(m)[1],1])
                h_p = tf.reshape(tf.matmul(tf.reshape(p_tile,[-1,self._embedding_size]),self.Wp),[-1,tf.shape(m)[1],self._embedding_size])
                f_m = tf.reshape(tf.matmul(tf.reshape(m,[-1,self._embedding_size]),self.Wm),[-1,tf.shape(m)[1],self._embedding_size])
                product = tf.multiply(h_p,f_m)
                m_1 = tf.layers.dense(product,self._embedding_size)
                m_2 = tf.layers.dense(m_1,self._embedding_size,activation=tf.nn.relu)
                m = tf.multiply(m,m_2)

                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                
                u_temp_profile = tf.transpose(tf.expand_dims(u_profile[-1], -1), [0, 2, 1])
                
                dotted = tf.reduce_sum(m * u_temp, 2)
                dotted_profile = tf.reduce_sum(m_profile * u_temp_profile, 2)

                # # Calculate probabilities
                probs = tf.nn.softmax(dotted)
                # probs = self.sparsemax(dotted)
                probs_profile = tf.nn.softmax(dotted_profile)
                # probs_profile = self.sparsemax(dotted_profile)
                
                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                probs_temp_profile = tf.transpose(tf.expand_dims(probs_profile, -1), [0, 2, 1])

                c_temp = tf.transpose(m, [0, 2, 1])
                c_temp_profile = tf.transpose(m_profile, [0, 2, 1])

                o_k = tf.reduce_sum(c_temp * probs_temp, 2)
                o_k_profile = tf.reduce_sum(c_temp_profile * probs_temp_profile, 2)

                u_pdt = u[-1]*o_k
                u_sum = tf.matmul(u[-1], self.H) + o_k
                u_k = tf.layers.dense(tf.concat([u_pdt,u_sum],1),self._embedding_size)
                # u_k = tf.matmul(u[-1], self.H) + o_k
                u_pdt_profile = u_profile[-1]*o_k_profile
                u_sum_profile = tf.matmul(u_profile[-1], self.H) + o_k_profile
                # u_k_profile = tf.matmul(u_profile[-1], self.H) + o_k_profile
                u_k_profile = tf.layers.dense(tf.concat([u_pdt_profile,u_sum_profile],1),self._embedding_size)

                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)
                    u_k_profile = self._nonlin(u_k_profile)

                u.append(u_k)
                u_profile.append(u_k_profile)

            u_kps = tf.reduce_sum(tf.transpose(tf.stack(u_profile),[1,0,2]),1)
            u_final = tf.add(u_k, u_kps)
            # m_p = tf.reduce_sum(m,1) + m_id
            # p_p = tf.reduce_sum(m_profile,1) + p_id
            # u_concat = tf.concat([u_final,m_p*p_p],1)
            # u_ff = tf.layers.dense(u_concat,self._embedding_size)    
            if self._nonlin:
                u_final = self._nonlin(u_final)

            
            # candidates_emb = self.multi_head1(candidates_emb)
            # candidates_emb = tf.nn.embedding_lookup(self.W, self._candidates)
            
            if self.task_id in [4]:
                rel_emb = tf.nn.embedding_lookup(self.A,relations)
                dotted_rel = tf.matmul(u_final,rel_emb,transpose_b=True)
                rel_weight = tf.nn.softmax(dotted_rel)
                v_p = tf.nn.relu(tf.matmul(tf.reduce_sum(m_profile,1),self.P_p))
                v_p = v_p*rel_weight
                v_m = tf.nn.relu(tf.matmul(tf.reduce_sum(m,1),self.P_m))
                v_m = v_m*rel_weight
                # candidates_item_emb = tf.nn.embedding_lookup(self.A,self._candidates_item)
                cand_item_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.cand_item,1),1),1)
                find_item = tf.equal(tf.to_float(stories), cand_item_exp)
                find_item_r = tf.reshape(find_item,[tf.shape(candidates_emb)[0], tf.shape(m_emb)[0], -1])
                item_res = tf.sign(tf.reduce_sum(tf.to_float(find_item_r),-1))

                bias_p = tf.matmul(v_p, self.cand_rel, transpose_b=True)
                bias_m = tf.matmul(v_m, self.cand_rel, transpose_b=True)
                item_res_trans = tf.transpose(item_res)
                bias_p_final = tf.multiply(bias_p, item_res_trans) 
                bias_m_final = tf.multiply(bias_m, item_res_trans) 
                result = tf.matmul(u_final, tf.transpose(candidates_emb_sum))
                final_res = result + bias_p_final + bias_m_final
                # final_res = result + bias_p + bias_m
            else:
                # candidates_embe = tf.nn.embedding_lookup(self.We, self._candidates)
                # candidates_embe_sum=tf.reduce_sum(candidates_embe,1)
                # m_emb_w = tf.nn.embedding_lookup(self.We,stories)
                # m_w = tf.reduce_sum(m_emb_w,2)
                # dot = tf.matmul(cands,m,transpose_b=True)
                # flag = tf.sign(tf.nn.relu(dot))
                # weight = tf.nn.softmax(dot)
                # weight = tf.multiply(weight,flag)
                # bias = tf.matmul(weight,m)
                # new_cand = cands+bias

                # mw_sum = tf.reduce_sum(m_w,1)
                # dote = tf.matmul(mw_sum,candidates_embe_sum,transpose_b=True)
                # flag = tf.sign(tf.nn.relu(dote))

                # bias = tf.reduce_mean(weight,1)
                # bias_f = tf.multiply(bias,flag)
                # c_dot = c_his[0]
                # for i in range(1,len(c_his)):
                #     c_dot = tf.multiply(c_dot,c_his[i])
                # weight = tf.nn.softmax(c_dot)
                # new_cand = tf.multiply(cands,weight)

                # dot_mp = tf.matmul(m,m_profile,transpose_b=True)
                # prob_mp = tf.expand_dims(tf.nn.softmax(tf.reduce_sum(dot_mp,2)),2)
                # m_sum = tf.reduce_sum(m*prob_mp,1)

                # flag = tf.sign(tf.nn.relu(tf.matmul(m_sum,candidates_emb_sum,transpose_b=True)))
                # m_tile = tf.tile(tf.expand_dims(m_sum,1),[1,tf.shape(cands)[1],1])
                # new_cand = cands + m_tile*tf.expand_dims(flag,2)
                
                user_concat = tf.concat([tf.reduce_sum(m,1)*u_0,tf.reduce_sum(m,1),u_0],1)
                user = tf.layers.dense(user_concat,self._embedding_size)
                user_item = tf.expand_dims(user,1)*candidates_emb_sum

                FM_W = tf.get_variable('FM_W', initializer=tf.constant(0.1, shape=[self._embedding_size, 1],dtype=tf.float32))
                FM_V = tf.get_variable('FM_V', initializer=tf.truncated_normal([self._embedding_size, self._embedding_size], stddev=0.1,dtype=tf.float32))

                ui_rhp = tf.reshape(user_item,[-1,self._embedding_size])
                fir_inter = tf.reshape(tf.squeeze(tf.matmul(ui_rhp, FM_W), [1]),[-1,tf.shape(candidates_emb)[0]])
                sec = tf.square(tf.matmul(ui_rhp, FM_V)) - tf.matmul(tf.square(ui_rhp), tf.square(FM_V))
                sec_inter = 0.5 * tf.reduce_sum(tf.reshape(sec,[-1,tf.shape(candidates_emb)[0],self._embedding_size]), 2)
                fm_res = tf.nn.relu(fir_inter + sec_inter)
                final_res = tf.matmul(u_k_profile, tf.transpose(candidates_emb_sum))
                final_res += fm_res
                
                # res = 0.5 * tf.reduce_sum(tf.square(final_res) - tf.matmul(tf.square(final_res), tf.square(FM_V)), 1)
                # final_res = tf.multiply(final_res,weight)
                # res = tf.matmul(tf.expand_dims(u_final,1), tf.transpose(new_cand,[0,2,1]))
                # final_res = tf.reduce_sum(res,1)
                # final_res = final_res + bias_f

            return final_res, eloss
            # logits=tf.matmul(u_k, self.W)
            # return tf.transpose(tf.sparse_tensor_dense_matmul(self._candidates,tf.transpose(logits))),prob_log, trans_att

    def batch_fit(self, profile, stories, queries, answers, relations,
                pos_h_batch, pos_r_batch, pos_t_batch, neg_h_batch, neg_r_batch, neg_t_batch):
        """Runs the training algorithm over the passed batch
        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)
        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._profile: profile, self._stories: stories, 
                    self._queries: queries, self._answers: answers,
                    self._relation: relations,
                    self.pos_h: pos_h_batch, self.pos_t: pos_t_batch,
                    self.pos_r: pos_r_batch, self.neg_h: neg_h_batch,
                    self.neg_t: neg_t_batch, self.neg_r: neg_r_batch}
        loss, eloss, _, _ = self._sess.run([self.loss_op, self.eloss_op, self.train_op, self.etrain_op], feed_dict=feed_dict)
        return loss,eloss

    def predict(self, profile, stories, queries,relations,
            pos_h_batch, pos_r_batch, pos_t_batch, neg_h_batch, neg_r_batch, neg_t_batch):
        """Predicts answers as one-hot encoding.
        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._profile: profile, self._stories: stories,
                    self._queries: queries,self._relation:relations,
                    self.pos_h: pos_h_batch,self.pos_t: pos_t_batch, 
                    self.pos_r: pos_r_batch, self.neg_h: neg_h_batch, 
                    self.neg_t: neg_t_batch, self.neg_r: neg_r_batch}
        return self._sess.run([self.predict_op], feed_dict=feed_dict)
    
    # def add_global_fun(self):
    #     self._sess.run([self.add_global])
    
