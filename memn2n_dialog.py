from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
from datetime import datetime
import random
import getdata
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops, dtypes, function
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn


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
        hops=3,
        max_grad_norm=40.0,
        nonlin=None,
        initializer=tf.random_normal_initializer(stddev=0.1),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
        session=tf.Session(),
        name='MemN2N',
                 task_id=1):
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
        self._name = name
        self._candidates=candidates_vec
        self.h = 4

        self._build_inputs()
        self._build_vars()
        # self.get_entity_emb()

        self.L1_flag = True # 在loss中是否加入L1正则化
        self.hidden_size = 200 # 实体与关系的词向量长度,在知识图谱中关系数量会远远小于实体个数，所以该超参调整不能太大
        self.sizeE = 40 # 实体词向量长度，仅在TransR中使用
        self.sizeR = 40 # 关系词向量长度，仅在TransR中使用，关系向量长度与实体向量长度不宜差距太大
        # self.ebatch_size = 100 # 每个批度输入的三元组个数
        self.margin = 1.0 # 合页损失函数中的标准化项
        self.relation_total = 0 # 知识图谱关系数，不需要修改，后续从输入数据中获得
        self.entity_total = 0 # 知识图谱实体数，不需要修改，后续从输入数据中获得
        self.triple_total = 0 # 知识图谱三元组个数，不需要修改，后续从数据输入中获得
        self.flie_path = 'facebook/' # 存放训练文件的路径，该路径下应该有训练时需要的三个文件，entity2id,relation2id,triple
        self.relation_total, self.entity_total, self.triple_total = getdata.get_data(self.flie_path)        
        self.rel_init = None # 关系向量预训练文件，每一行是一个向量，张量大小和要训练的参数一致，仅在TransR中使用
        self.ent_init = None # 实体向量预训练文件，每一行是一个向量，张量大小和要训练的参数一致，仅在TransR中使用
        
        # define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = "%s_%s_%s_%s/" % ('task', str(task_id),'summary_output', timestamp)
        
        
        # cross entropy
        logits, eloss, prob, prob_p = self._inference(self._profile, self._stories, self._queries, self.pos_h, self.pos_t, self.pos_r, self.neg_h, self.neg_t, self.neg_r) # (batch_size, candidates_size)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = self._answers, name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")
        
        # neg_logits = []
        # logits_l = tf.split(logits,batch_size,axis=0)
        # for logit in logits_l:
        #     negrand = []
        #     while len(negrand)<5:
        #         r = random.randint(1,self._candidates_size)
        #         if r not in negrand:
        #             negrand.append(r)
        #     neg_logits.append(tf.stack([logit[0][i] for i in negrand]))
        # neg_logits = tf.stack(neg_logits)        
        # w = tf.get_variable("w", [1,1], regularizer=tf.contrib.layers.l2_regularizer(0.001))
        # b = tf.get_variable("b", [1], regularizer=tf.contrib.layers.l2_regularizer(0.001))
        # pos_s = tf.log_sigmoid(tf.cast(self._answers,dtype=tf.float32)*w + b)
        # neg_s = tf.log_sigmoid(-(neg_logits*w) - b)
        # neg_loss = - pos_s - neg_s

        # loss op
        loss_op = cross_entropy_sum + eloss

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op) 
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op
        self.prob_log = prob
        self.prob_p_log = prob_p


        self.graph_output = self.loss_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)
        

    def _build_inputs(self):
        self._profile = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="profile")
        self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
        
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answers")

        with tf.name_scope("read_inputs"):
            self.pos_h = tf.placeholder(tf.int32, [self._batch_size])
            self.pos_t = tf.placeholder(tf.int32, [self._batch_size])
            self.pos_r = tf.placeholder(tf.int32, [self._batch_size])
            self.neg_h = tf.placeholder(tf.int32, [self._batch_size])
            self.neg_t = tf.placeholder(tf.int32, [self._batch_size])
            self.neg_r = tf.placeholder(tf.int32, [self._batch_size])
        
    def get_entity_emb(self):
        entity = []
        f = open('.\entityVectorNew.txt','r',encoding='utf-8')
        lines = f.read().split('\n')
        for line in lines:
            entity.append(eval(line))
        
        for k in range(len(entity)):
            rand = random.random()
            if rand < 0.05:
                entity[k] = entity[random.randint(0,len(entity)-1)]

        self._entityemb = tf.cast(entity,dtype=tf.float32,name="entity")
    
    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.A = tf.Variable(A, name="A")
            B = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.B = tf.Variable(B, name="B")
            # self.C = tf.Variable(A, name="B")
            # C = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            
            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            self.C_1 = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="C_1")
            self.C_2 = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="C_2")
            self.H_e = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H_e")
            W = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.W = tf.Variable(W, name="W")
            self.W_o = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W")
            self.W_k1 = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W_k1")
            self.W_k2 = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W_k2")
            self.M = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="M")
            
            self.W_i = tf.Variable(self._init([3*self._embedding_size, self._embedding_size]), name="W_k")
            # self.W_t = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W_t")
            # self.W_e = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W_e")
            self.b = tf.Variable(self._init([self._embedding_size]), name="b")
            # self.b_t = tf.Variable(self._init([self._embedding_size]), name="b_t")
            self.b_p = tf.Variable(self._init([1]), name="b_p")
        self._nil_vars = set([self.A.name,self.W.name])
    
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

    def self_attention(self, t, T, n, h):
        if T == 'queries':
            exec('self.W_q_q_' + str(n) +'= tf.get_variable(name="W_q_q_'+ str(n) +'",initializer=self._init([self._embedding_size, 10]))')
            exec('self.W_q_k_' + str(n) +'= tf.get_variable(name="W_q_k_'+ str(n) +'",initializer=self._init([self._embedding_size, 10]))')
            exec('self.W_q_v_' + str(n) +'= tf.get_variable(name="W_q_v_'+ str(n) +'",initializer=self._init([self._embedding_size, 10]))')
            t_reshape = tf.reshape(t,[-1, self._embedding_size])
            query = tf.reshape(tf.matmul(t_reshape, eval('self.W_q_q_'+str(n))),[-1,tf.shape(t)[1],10])
            key = tf.reshape(tf.matmul(t_reshape, eval('self.W_q_k_'+str(n))),[-1,tf.shape(t)[1],10])
            value = tf.reshape(tf.matmul(t_reshape, eval('self.W_q_v_'+str(n))),[-1,tf.shape(t)[1],10])
            output = tf.matmul(query,tf.transpose(key,[0,2,1]))/tf.sqrt(tf.cast(64,dtype=tf.float32))
        elif T == 'stories':
            exec('self.W_s_q_' + str(n) +'= tf.get_variable(name="W_s_q_'+str(n)+r'",initializer=self._init([self._embedding_size, int(self._embedding_size/h)]))')
            exec('self.W_s_k_' + str(n) +'= tf.get_variable(name="W_s_k_'+str(n)+r'",initializer=self._init([self._embedding_size, int(self._embedding_size/h)]))')
            exec('self.W_s_v_' + str(n) +'= tf.get_variable(name="W_s_v_'+str(n)+r'",initializer=self._init([self._embedding_size, int(self._embedding_size/h)]))')
            t_reshape = tf.reshape(t,[-1, self._embedding_size])
            query = tf.reshape(tf.matmul(t_reshape, eval('self.W_s_q_'+str(n))),[-1,tf.shape(t)[2],int(self._embedding_size/h)])
            key = tf.reshape(tf.matmul(t_reshape, eval('self.W_s_k_'+str(n))),[-1,tf.shape(t)[2],int(self._embedding_size/h)])
            value = tf.reshape(tf.matmul(t_reshape, eval('self.W_s_v_'+str(n))),[-1,tf.shape(t)[2],int(self._embedding_size/h)])
            output = tf.matmul(query,tf.transpose(key,[0,2,1]))/tf.sqrt(tf.cast(64,dtype=tf.float32))
    
            #对key进行mask
            t_r = tf.reshape(t,[-1,tf.shape(t)[2],self._embedding_size])
            # key_masks = tf.sign(tf.abs(tf.matmul(t_r,t_r,transpose_b=True))) # (N, T_k)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(t_r, axis=-1)))
            # key_masks = tf.matmul(key_masks, key_masks,transpose_a=True)            
            # key_masks = tf.reshape(key_masks,[-1,self._sentence_size])
            # key_masks = tf.tile(tf.expand_dims(key_masks, 0), [tf.shape(output)[0], 1, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks,1), [1, self._sentence_size, 1]) # (h*N, T_q, T_k)
            paddings = tf.ones_like(output)*(-2**32+1)
            output = tf.where(tf.equal(key_masks, 0), paddings, output) # (h*N, T_q, T_k)
            # ones = tf.ones_like(output,dtype=tf.float32)
            # zeros = tf.zeros_like(output,dtype=tf.float32)
            # diagpadd = tf.matrix_band_part(ones, 0 , 0)
            # diagpadd = tf.where(tf.equal(diagpadd,0),ones,zeros)
            # zeros = tf.ones_like(output)*(-2**32+1)
            # paddings = tf.ones_like(output)*(-2**32+1)
            # output = tf.where(tf.equal(diagpadd,0),paddings,output)
            # output = tf.nn.softmax(output)
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
    
    def multi_head(self, input, T, h=4):
        attentions = []
        logs = []
        for n in range(h):
            attentions.append(self.self_attention(input, T, n, h)[0])
            logs.append(self.self_attention(input, T, n, h)[1])
        MH = tf.concat(attentions,axis=2)
        MH_reshape = tf.reshape(MH,[-1, self._embedding_size])
        output = tf.matmul(MH_reshape,self.W_o)
        dims = [tf.shape(input)[i] for i in range(tf.shape(input).shape.dims[0].value)]
        logs = tf.reduce_mean(logs,axis=0)
        return [tf.reshape(output,dims) + input,logs]
    
    def FFN(self,inputs,num_units=[160, 40]):
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

    def position_encoding(self,inputs,type, num_units=40,scope="positional_encoding"):
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

    def aggregator(self,wi,ei):
        def gelu(x):
	        return x * 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
        wi_r = tf.reshape(wi,[-1,self._embedding_size])
        ei_r = tf.reshape(ei,[-1,self._embedding_size])
        h = gelu(tf.matmul(wi_r,self.W_t)+tf.matmul(ei_r,self.W_e)+self.b)
        wj = gelu(tf.matmul(h,self.W_t)+ self.b_t)
        ej = gelu(tf.matmul(h,self.W_e)+ self.b_e)
        wj = tf.reshape(wj,[-1,tf.shape(wi)[1],self._embedding_size])
        ej = tf.reshape(ej,[-1,tf.shape(ei)[1],self._embedding_size])
        return tf.layers.dropout(wj,0.1),tf.layers.dropout(ej,0.1)

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
        k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=1)

        # calculate tau(z)
        indices = array_ops.stack([math_ops.range(0, obs), k_z - 1], axis=1)
        tau_sum = array_ops.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / math_ops.cast(k_z, logits.dtype)

        # calculate p
        return math_ops.maximum(
            math_ops.cast(0, logits.dtype),
            z - tau_z[:, array_ops.newaxis]
        )


    def csoftmax(self,inputs,p=False):
        def fun_t(value):
            self.res_temp.append(value)
            return 0
        def fun_f():
            return 1

        z = inputs
        sortz = tf.nn.top_k(z,tf.shape(z)[1],sorted=True)[0]
        zs = tf.split(sortz,self._batch_size,axis=0)
        res = []
        for zz in zs:
            # start = tf.shape(zz)[0]-1
            if p:
                zzs = tf.split(zz,4,0)
            else:
                zzs = tf.split(zz,self._sentence_size,0)
            self.res_temp = []
            for i,zzz in enumerate(zzs):
                cur = tf.gather_nd(zz,[0,i])
                cur_sum = tf.reduce_sum(tf.slice(zz,[0,0],[1,i+1]))
                _ = tf.cond(1+tf.to_float(i+1)*cur > cur_sum,lambda: fun_t(i),lambda : fun_f())
            
            cur = tf.gather_nd(zz,[0,max(self.res_temp)]) 
            tao = (cur_sum - 1) / cur
            zeros = tf.zeros_like(zz)
            judge = tf.cast(zz-tao>zeros,dtype=tf.float32)
            out = tf.where(tf.equal(judge,0),zeros,zz)
            res.append(out)
        output = tf.concat(res,0)
        return output

    def mreinforce(self, m):
        pre_ms = math_ops.cumsum(m)
        m_n_pre = tf.concat([m,pre_ms], axis=2)
        m_n_pre_r = tf.reshape(m_n_pre, [-1, self._embedding_size*2])
        alpha_r = tf.sigmoid(tf.matmul(m_n_pre_r, self.W_i) + self.b)
        alpha = tf.reshape(alpha_r, [-1, tf.shape(m_n_pre)[1],self._embedding_size])
        alpha = tf.nn.softmax(alpha)
        m = m*alpha
        return m
    
    def mupdata(self, m, cur_m, pre_m):
        cur_mm = tf.expand_dims(cur_m,1)
        pre_mm = tf.expand_dims(pre_m,1)

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self._embedding_size)
        zero_state = cell.zero_state(batch_size=tf.shape(cur_m)[0], dtype=tf.float32)
        # a = tf.random_normal([2, 3, 4])
        out_c, state_c = tf.nn.dynamic_rnn(
            cell=cell,
            initial_state=zero_state,
            inputs=cur_mm
        )

        # out_c = tf.reduce_sum(out_c,1)
        out_p, state_p = tf.nn.dynamic_rnn(
            cell=cell,
            initial_state=zero_state,
            inputs=pre_mm
        )
        # out_p = tf.reduce_sum(out_p,1)

        _, state_m = tf.nn.dynamic_rnn(
            cell=cell,
            initial_state=zero_state,
            inputs=m
        )

        state_m = state_m.h
        state_c = state_c.h
        state_p = state_p.h
        outs = tf.concat([state_c,state_p,state_m],axis=1)
        # outs_r = tf.reshape(outs, [-1, self._embedding_size*2])
        gama = tf.sigmoid(tf.matmul(outs, self.W_i) + self.b)
        # gama = tf.reshape(gama_r,[-1, self._embedding_size])

        output = [cur_m*gama,pre_m*gama]
        return output

    def projection(self,inputs):
        r_inputs = tf.reshape(inputs,[-1,self._embedding_size])
        project = tf.matmul(r_inputs,self.M)
        dims = [tf.shape(inputs)[i] for i in range(tf.shape(inputs).shape.dims[0].value)]
        output = tf.reshape(project,dims)
        return output

    def transr(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[self.entity_total, self.sizeE],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[self.relation_total, self.sizeR],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        rel_matrix = np.zeros([self.relation_total, self.sizeR * self.sizeE], dtype=np.float32)
        for i in range(self.relation_total):
            for j in range(self.sizeR):
                for k in range(self.sizeE):
                    if j == k:
                        rel_matrix[i][j * self.sizeE + k] = 1.0
        self.rel_matrix = tf.Variable(rel_matrix, name="rel_matrix")

        with tf.name_scope('lookup_embeddings'):
            pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_h), [-1, self.sizeE, 1])
            pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, pos_t), [-1, self.sizeE, 1])
            pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, pos_r), [-1, self.sizeR])
            neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_h), [-1, self.sizeE, 1])
            neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, neg_t), [-1, self.sizeE, 1])
            neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, neg_r), [-1, self.sizeR])
            pos_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, pos_r), [-1, self.sizeR, self.sizeE])
            neg_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, neg_r), [-1, self.sizeR, self.sizeE])

            pos_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_h_e), [-1, self.sizeR]), 1)
            pos_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_t_e), [-1, self.sizeR]), 1)
            neg_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(neg_matrix, neg_h_e), [-1, self.sizeR]), 1)
            neg_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(neg_matrix, neg_t_e), [-1, self.sizeR]), 1)

        
        pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
        neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
        self.predictt = pos

        with tf.name_scope("output"):
            self.eloss = tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0))
        
        return self.eloss, pos_matrix

    def _inference(self, profile, stories, queries, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        with tf.variable_scope(self._name, reuse = tf.AUTO_REUSE):
            eloss, self.M_r = self.transr(pos_h, pos_t, pos_r, neg_h, neg_t, neg_r)
            q_emb = tf.nn.embedding_lookup(self.A, queries)
            # qe_emb = tf.nn.embedding_lookup(self.ent_embeddings,queries)
            #TODO:调整投影矩阵维度使能和emb做matmul
            q_emb += tf.matmul(q_emb, self.M_r)
            q_emb += tf.nn.embedding_lookup(self.ent_embeddings, queries)
            q_emb += self.position_encoding(queries,'query')
            u_0 = self.multi_head(q_emb,'queries')[0]
            # u_0 = self.FFN(u_0)
            u_0 = tf.reduce_sum(u_0, 1)
            u = [u_0]
            u_profile = [u_0]
            f = []
            probs_log = []
            sprobs_log = []
            probs_p_log = []
            for count in range(self._hops):
                m_emb = tf.nn.embedding_lookup(self.A, stories)
                # me_emb = tf.nn.embedding_lookup(self.ent_embeddings, stories)

                m_emb_r = tf.matmul(tf.reshape(m_emb,[-1,tf.shape(m_emb)[1]*tf.shape(m_emb)[2],self._embedding_size]), self.M_r)
                m_emb = tf.reshape(m_emb_r,[-1,tf.shape(m_emb)[1], tf.shape(m_emb)[2],self._embedding_size])
                m_emb += tf.nn.embedding_lookup(self.ent_embeddings, stories)
                m_emb += self.position_encoding(stories,'story')
                
                m_emb_profile = tf.nn.embedding_lookup(self.A, profile)
                # me_emb_profile = tf.nn.embedding_lookup(self.ent_embeddings, stories)
                m_emb_profile_r = tf.matmul(tf.reshape(m_emb_profile,[-1,tf.shape(m_emb_profile)[1]*tf.shape(m_emb_profile)[2],self._embedding_size]), self.M_r)
                m_emb_profile = tf.reshape(m_emb_profile_r,[-1,tf.shape(m_emb_profile)[1], tf.shape(m_emb_profile)[2],self._embedding_size])
                m_emb_profile += tf.nn.embedding_lookup(self.ent_embeddings, profile)
                m_emb_profile += self.position_encoding(profile, 'story')
                
                # m = tf.reduce_sum(m_emb, 2)
                if count == 0:
                    m,trans_att = self.multi_head(m_emb,'stories')
                else: 
                    m = self.multi_head(f[-1],'stories')[0]
                # m = self.FFN(m)
                f.append(m)
                m = tf.reduce_sum(m, 2)
                # m = self.mreinforce(m)
                
                # entity = tf.nn.embedding_lookup(self._entityemb,stories)
                # # entity = tf.reduce_sum(entity_emb,2)
                # # entity += self.position_encoding(entity)               
                
                # e = self.multi_head(tf.to_float(tf.reduce_sum(entity, 2)),'stories')
                

                # for ii in range(3):
                #     if ii == 0:
                #         wi,ei = self.aggregator(m,e)
                #     else:
                #         wi = self.multi_head(wi,'stories')
                #         ei = self.multi_head(ei,'stories')
                #         wi,ei = self.aggregator(wi,ei)
                # ei = tf.transpose(ei,[0,2,1])
                # m = tf.layers.dropout(tf.contrib.layers.fully_connected(m,self._embedding_size),0.2)
                # w_e = tf.nn.softmax(tf.matmul(m,ei))              
                # m = tf.transpose(m,[0,2,1])
                # m = tf.transpose(tf.matmul(m,w_e),[0,2,1])
                # e = tf.reduce_sum(ei,2)
                                          
                m_profile = tf.reduce_sum(m_emb_profile, 2)
                
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                u_temp_profile = tf.transpose(tf.expand_dims(u_profile[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)
                dotted_profile = tf.reduce_sum(m_profile * u_temp_profile, 2)

                # Calculate probabilities
                # padd = tf.ones_like(dottede)*(-2**32+1)
                # dottede = tf.where(tf.equal(dottede,0),padd,dottede)
                probs = self.sparsemax(dotted)
                probs_profile = self.sparsemax(dotted_profile)

                # probs = self.sparsemax(dotted)
                # probs_profile = self.sparsemax(dotted_profile)

                # sprobs = self.sparsemax(dotted)
                # probs_profile = self.sparsemax(dotted_profile)
                
                # sprobs_log.append(sprobs)
                probs_log.append(probs)
                probs_p_log.append(probs_profile)
                # def setprobs(prob):
                #     ones = tf.ones_like(prob)
                #     zeros = ones*(self.b_p)

                #     reprob = tf.reshape(prob,[1,-1])
                #     sortprob = tf.nn.top_k(reprob,tf.shape(reprob)[1],sorted=True)[0]
                #     count = tf.to_float(tf.size(prob))
                #     loc = tf.cast(count*0.7,dtype=tf.int32)
                #     threshold = tf.gather_nd(sortprob,[0,loc])

                #     a = ones*threshold
                #     c = tf.cast(prob<a,dtype=tf.float32)
                #     prob = tf.where(tf.equal(c,0),prob,zeros)
                #     prob = tf.nn.softmax(prob)
                #     return prob

                # probs = setprobs(probs)
                # probs_profile = setprobs(probs_profile)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                probs_temp_profile = tf.transpose(tf.expand_dims(probs_profile, -1), [0, 2, 1])

                c_temp = tf.transpose(m, [0, 2, 1])
                c_temp_profile = tf.transpose(m_profile, [0, 2, 1])

                o_k = tf.reduce_sum(c_temp * probs_temp, 2)
                o_k_profile = tf.reduce_sum(c_temp_profile * probs_temp_profile, 2)

                # o_kk,u[-1] = self.mupdata(m, o_k, u[-1])
                u_k = tf.matmul(u[-1], self.H) + o_k
                u_k_profile = tf.matmul(u_profile[-1], self.H) + o_k_profile
                # u_k=u[-1]+tf.matmul(o_k,self.H)
                # u_k_profile=u_profile[-1]+tf.matmul(o_k_profile,self.H)
                
                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)
                    u_k_profile = self._nonlin(u_k_profile)

                u.append(u_k)
                u_profile.append(u_k_profile)

                # if count == 0:
                #     candidates_emb=tf.nn.embedding_lookup(self.W, self._candidates)
                #     candidates_emb_sum=tf.reduce_sum(candidates_emb,1)
                
                # cands = tf.matmul(u_k,tf.transpose(candidates_emb_sum))
                # candsre = tf.nn.softmax(cands)
                # indices = tf.nn.top_k(candsre,100).indices
                # cand = tf.reduce_sum(tf.gather(candidates_emb,indices),2)
                
                # cdotted = tf.reduce_sum(cand * u_temp, 2)

                # # Calculate probabilities
                # cprobs = tf.nn.softmax(cdotted)
                # cprobs_temp = tf.transpose(tf.expand_dims(cprobs, -1), [0, 2, 1])

                # c_temp = tf.transpose(cand, [0, 2, 1])

                # co_k = tf.reduce_sum(c_temp * cprobs_temp, 2)
                # u[-1] += co_k 
                    
            # u_k_k = tf.tanh(tf.matmul(u_k,self.W_k1))
            # u_k_profile_k = tf.tanh(tf.matmul(u_k_profile,self.W_k2))
            # k = tf.sigmoid(tf.matmul(tf.concat([u_k_k,u_k_profile_k],1),self.W_k),name='k')
            
            # ones = tf.ones_like(k)
            # u_k = tf.multiply(k,u_k)
            # u_k_profile = tf.multiply((1-k),u_k_profile)

            u_final = tf.add(u_k,u_k_profile)
            if self._nonlin:
                u_final = self._nonlin(u_final)
            
            asum = math_ops.cumsum(probs_log)[-1]
            prob_log = asum/len(probs_log)

            # sasum = math_ops.cumsum(sprobs_log)[-1]
            # sprob_log = sasum/len(sprobs_log)

            # asum_p = math_ops.cumsum(probs_p_log)[-1]
            # prob_p_log = asum_p/len(probs_p_log)

            candidates_emb=tf.nn.embedding_lookup(self.W, self._candidates)
            candidates_emb_r = tf.matmul(tf.reshape(candidates_emb,[-1,self._embedding_size]),tf.reduce_max(self.M_r,0))
            candidates_emb = tf.reshape(candidates_emb_r,[-1,tf.shape(candidates_emb)[1],self._embedding_size])
            candidates_emb+=tf.nn.embedding_lookup(self.ent_embeddings, self._candidates)
            # candidates_emb=tf.matmul(candidates_emb, self.M_r)
            # candidates_emb = tf.add(self.projection(candidates_emb), self.projection(candidatese_emb))
            candidates_emb_sum=tf.reduce_sum(candidates_emb,1)

            # setout(u_final, candidates_emb_sum)

            return tf.matmul(u_final,tf.transpose(candidates_emb_sum)), eloss, prob_log, trans_att
            # logits=tf.matmul(u_k, self.W)
            # return tf.transpose(tf.sparse_tensor_dense_matmul(self._candidates,tf.transpose(logits))),prob_log, trans_att

    def batch_fit(self, profile, stories, queries, answers, 
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
                    self.pos_h: pos_h_batch, self.pos_t: pos_t_batch,
                    self.pos_r: pos_r_batch, self.neg_h: neg_h_batch,
                    self.neg_t: neg_t_batch, self.neg_r: neg_r_batch}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, profile, stories, queries, 
            pos_h_batch, pos_r_batch, pos_t_batch, neg_h_batch, neg_r_batch, neg_t_batch):
        """Predicts answers as one-hot encoding.
        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._profile: profile, self._stories: stories,
                    self._queries: queries, self.pos_h: pos_h_batch, 
                    self.pos_t: pos_t_batch, self.pos_r: pos_r_batch, 
                    self.neg_h: neg_h_batch, self.neg_t: neg_t_batch,
                     self.neg_r: neg_r_batch}
        return self._sess.run([self.predict_op, self.prob_log, self.prob_p_log], feed_dict=feed_dict)
