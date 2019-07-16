from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
from datetime import datetime
import random


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
        self.h = 6

        self._build_inputs()
        self._build_vars()
        
        # define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = "%s_%s_%s_%s/" % ('task', str(task_id),'summary_output', timestamp)
        
        
        # cross entropy
        logits = self._inference(self._profile, self._stories, self._queries, self._weights) # (batch_size, candidates_size)
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
        loss_op = cross_entropy_sum

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
        
        self.graph_output = self.loss_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._profile = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="profile")
        self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answers")
        self._weights = tf.placeholder(tf.int32, [None, 4, 4], name="weights")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.B = tf.Variable(A, name="A")
            B = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.C = tf.Variable(A, name="B")
            C = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.A = tf.Variable(A, name="C")
            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            W = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], axis=0)
            self.W = tf.Variable(W, name="W")
            self.W_o = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W")
            self.W_k1 = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W_k1")
            self.W_k2 = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W_k2")
            self.W_k = tf.Variable(self._init([2*self._embedding_size, self._embedding_size]), name="W_k")
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

    def self_attention(self, t, T, n):
        if T == 'queries':
            exec('self.W_q_q_' + str(n) +'= tf.get_variable(name="W_q_q_'+ str(n) +'",initializer=self._init([self._embedding_size, 10]))')
            exec('self.W_q_k_' + str(n) +'= tf.get_variable(name="W_q_k_'+ str(n) +'",initializer=self._init([self._embedding_size, 10]))')
            exec('self.W_q_v_' + str(n) +'= tf.get_variable(name="W_q_v_'+ str(n) +'",initializer=self._init([self._embedding_size, 10]))')
            t_reshape = tf.reshape(t,[-1, self._embedding_size])
            query = tf.reshape(tf.matmul(t_reshape, eval('self.W_q_q_'+str(n))),[-1,t.get_shape()[1],10])
            key = tf.reshape(tf.matmul(t_reshape, eval('self.W_q_k_'+str(n))),[-1,t.get_shape()[1],10])
            value = tf.reshape(tf.matmul(t_reshape, eval('self.W_q_v_'+str(n))),[-1,t.get_shape()[1],10])
            score = tf.matmul(query,tf.transpose(key,[0,2,1]))/tf.sqrt(tf.cast(32,dtype=tf.float32))
        elif T == 'stories':
            exec('self.W_s_q_' + str(n) +'= tf.get_variable(name="W_s_q_'+str(n)+'",initializer=self._init([self._embedding_size, 10]))')
            exec('self.W_s_k_' + str(n) +'= tf.get_variable(name="W_s_k_'+str(n)+'",initializer=self._init([self._embedding_size, 10]))')
            exec('self.W_s_v_' + str(n) +'= tf.get_variable(name="W_s_v_'+str(n)+'",initializer=self._init([self._embedding_size, 10]))')
            t_reshape = tf.reshape(t,[-1, self._embedding_size])
            query = tf.reshape(tf.matmul(t_reshape, eval('self.W_s_q_'+str(n))),[-1,tf.shape(t)[1],10])
            key = tf.reshape(tf.matmul(t_reshape, eval('self.W_s_k_'+str(n))),[-1,tf.shape(t)[1],10])
            value = tf.reshape(tf.matmul(t_reshape, eval('self.W_s_v_'+str(n))),[-1,tf.shape(t)[1],10])
            score = tf.matmul(query,tf.transpose(key,[0,2,1]))/tf.sqrt(tf.cast(64,dtype=tf.float32))
    
        #对key进行mask
        key_masks = tf.sign(tf.reduce_sum(tf.abs(t),axis=-1)) # (N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks,1), [1, tf.shape(t)[1], 1]) # (h*N, T_q, T_k)
        paddings = tf.ones_like(score)*(-2**32+1)
        score = tf.where(tf.equal(key_masks, 0), paddings, score) # (h*N, T_q, T_k)
        attention = tf.nn.softmax(score)
        
        #对query进行mask
        query_masks = tf.sign(tf.reduce_sum(tf.abs(t), axis=-1)) # (N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(t)[1]]) # (h*N, T_q, T_k)
        attention *= query_masks # broadcasting. (N, T_q, C)

        attention = tf.layers.dropout(attention,0.2)
        output = tf.matmul(attention,value)
        return self.normalize(output)
    
    def multi_head(self, input, T, h=6):
        attentions = []
        for n in range(h):
            attentions.append(self.self_attention(input, T, n))
        MH = tf.concat(attentions,axis=2)
        MH_reshape = tf.reshape(MH,[-1, self._embedding_size])
        output = tf.matmul(MH_reshape,self.W_o)
        return tf.reshape(output,[-1,tf.shape(MH)[1],self._embedding_size]) + input
    
    def FFN(self,inputs,num_units=[240, 60]):
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
        return outputs

    def position_encoding(self,inputs,num_units=60,scope="positional_encoding"):
        inputs_r = tf.reshape(inputs,[-1,21])
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            N, T = inputs_r.get_shape().as_list()
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [tf.shape(inputs)[0], 1])

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
                for pos in range(T)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(position_enc)
            outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
            outputs = tf.where(tf.equal(inputs, 0), inputs, tf.to_float(outputs))
            return tf.to_float(outputs)

    def _inference(self, profile, stories, queries, weight):
        with tf.variable_scope(self._name, reuse = tf.AUTO_REUSE):
            q_emb = tf.nn.embedding_lookup(self.B, queries)
            q_emb += self.position_encoding(q_emb)
            u_0 = self.multi_head(q_emb,'queries')
            u_0 = self.FFN(u_0)
            u_0 = tf.reduce_sum(u_0, 1)
            u = [u_0]
            u_profile = [u_0]
            f = []
            for count in range(self._hops):
                m_emb = tf.nn.embedding_lookup(self.A, stories)
                m_emb_profile = tf.nn.embedding_lookup(self.A, profile)
                
                # m = tf.reduce_sum(m_emb, 2)
                if count == 0:
                    m = self.multi_head(tf.reduce_sum(m_emb, 2),'stories')
                else: 
                    m = self.multi_head(f[-1],'stories')
                m = self.FFN(m)
                f.append(m)
                m_profile = tf.reduce_sum(m_emb_profile, 2)
                
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                u_temp_profile = tf.transpose(tf.expand_dims(u_profile[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)
                dotted_profile = tf.reduce_sum(m_profile * u_temp_profile, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)
                probs_profile = tf.nn.softmax(dotted_profile)
                # probs = tf.Print(probs, ['memory', count, tf.shape(probs), probs], summarize=200)
                # probs_profile = tf.Print(probs_profile, ['profile', count, tf.shape(probs_profile), probs_profile], summarize=200)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                probs_temp_profile = tf.transpose(tf.expand_dims(probs_profile, -1), [0, 2, 1])

                m_o = tf.reduce_sum(tf.nn.embedding_lookup(self.C, stories),2)
                m_profile_o = tf.reduce_sum(tf.nn.embedding_lookup(self.C, profile),2)

                c_temp = tf.transpose(m_o, [0, 2, 1])
                c_temp_profile = tf.transpose(m_profile_o, [0, 2, 1])

                o_k = tf.reduce_sum(c_temp * probs_temp, 2)
                o_k_profile = tf.reduce_sum(c_temp_profile * probs_temp_profile, 2)

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

            candidates_emb=tf.nn.embedding_lookup(self.W, self._candidates)
            candidates_emb_sum=tf.reduce_sum(candidates_emb,1)
            
            u_k_k = tf.nn.tanh(tf.matmul(u_k,self.W_k1))
            u_k_profile_k = tf.nn.tanh(tf.matmul(u_k_profile,self.W_k2))
            k = tf.nn.sigmoid(tf.matmul(tf.concat([u_k_k,u_k_profile_k],1),self.W_k),name='k')
            print(k)
            # ones = tf.ones_like(k)

            # u_final = tf.add(tf.multiply(k,u_k), tf.multiply((ones-k),u_k_profile))
            u_final = tf.add(u_k,u_k_profile)
            if self._nonlin:
                u_final = self._nonlin(u_final)

            return tf.matmul(u_final,tf.transpose(candidates_emb_sum))
            # logits=tf.matmul(u_k, self.W)
            # return tf.transpose(tf.sparse_tensor_dense_matmul(self._candidates,tf.transpose(logits)))


    def batch_fit(self, profile, stories, queries, answers, weights):
        """Runs the training algorithm over the passed batch
        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)
        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._profile: profile, self._stories: stories, 
        self._queries: queries, self._answers: answers, self._weights: weights}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, profile, stories, queries, weights):
        """Predicts answers as one-hot encoding.
        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._profile: profile, self._stories: stories,
         self._queries: queries, self._weights: weights}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)
