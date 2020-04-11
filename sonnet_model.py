import tensorflow as tf
import numpy as np
import math
import time
from util import *
from collections import Counter



class SonnetModel(object):
    #class où on def le modèle de langage
    
    
    def get_last_hidden(self, h, xlen):

        ids = tf.range(tf.shape(xlen)[0])
        gather_ids = tf.concat([tf.expand_dims(ids, 1), tf.expand_dims(xlen-1, 1)], 1)
        return tf.gather_nd(h, gather_ids)


    #La porte qui selectionne
    def gated_layer(self, s, h, sdim, hdim):

        update_w = tf.compat.v1.get_variable("update_w", [sdim+hdim, hdim])
        update_b = tf.compat.v1.get_variable("update_b", [hdim], initializer=tf.constant_initializer(1.0))
        reset_w  = tf.compat.v1.get_variable("reset_w", [sdim+hdim, hdim])
        reset_b  = tf.compat.v1.get_variable("reset_b", [hdim], initializer=tf.constant_initializer(1.0))
        c_w      = tf.compat.v1.get_variable("c_w", [sdim+hdim, hdim])
        c_b      = tf.compat.v1.get_variable("c_b", [hdim], initializer=tf.constant_initializer())

        z = tf.sigmoid(tf.matmul(tf.concat([s, h], 1), update_w) + update_b)
        r = tf.sigmoid(tf.matmul(tf.concat([s, h], 1), reset_w) + reset_b)
        c = tf.tanh(tf.matmul(tf.concat([s, r*h], 1), c_w) + c_b)
        
        return (1-z)*h + z*c

    #Pareil qu'avant
    def selective_encoding(self, h, s, hdim):

        h1 = tf.shape(h)[0]
        h2 = tf.shape(h)[1]
        h_ = tf.reshape(h, [-1, hdim])
        s_ = tf.reshape(tf.tile(s, [1, h2]), [-1, hdim])

        attend_w = tf.compat.v1.get_variable("attend_w", [hdim*2, hdim])
        attend_b = tf.compat.v1.get_variable("attend_b", [hdim], initializer=tf.constant_initializer())

        g = tf.sigmoid(tf.matmul(tf.concat([h_, s_], 1), attend_w) + attend_b)

        return tf.reshape(h_* g, [h1, h2, -1])

    
    #Initialisation de la classe
    def __init__(self, is_training, batch_size, word_type_size, char_type_size, space_id, pad_id, cf):

        #pour la config
        self.config = cf


        ##############
        #placeholders#
        ##############

        #language model placeholders
        self.lm_x    = tf.compat.v1.placeholder(tf.int32, [None, None])
        self.lm_xlen = tf.compat.v1.placeholder(tf.int32, [None])
        self.lm_y    = tf.compat.v1.placeholder(tf.int32, [None, None])
        self.lm_hist = tf.compat.v1.placeholder(tf.int32, [None, None])
        self.lm_hlen = tf.compat.v1.placeholder(tf.int32, [None])


        ################
        #language model#
        ################
        with tf.compat.v1.variable_scope("language_model"):
            self.init_lm(is_training, batch_size, word_type_size)

    
    
    ################
    #Language model#
    ################
    
    
    # -- language model network
    def init_lm(self, is_training, batch_size, word_type_size):
        
        #On mets la config de base
        cf = self.config
        
        
        #shared word embeddings (used by encoder and decoder)
        self.word_embedding = tf.compat.v1.get_variable("word_embedding", [word_type_size, cf.word_embedding_dim],
            initializer=tf.random_uniform_initializer(-0.05/cf.word_embedding_dim, 0.05/cf.word_embedding_dim))
    
        #########
        #decoder#
        #########

        #define lstm cells
        lm_dec_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(cf.lm_dec_dim, use_peepholes=True, forget_bias=1.0)
        if is_training and cf.keep_prob < 1.0:
            lm_dec_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lm_dec_cell, output_keep_prob=cf.keep_prob)
        self.lm_dec_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lm_dec_cell] * cf.lm_dec_layer_size)

        #initial states
        self.lm_initial_state = self.lm_dec_cell.zero_state(batch_size, tf.float32)
        state = self.lm_initial_state

        #pad symbol vocab ID = 0; create mask = 1.0 where vocab ID > 0 else 0.0
        lm_mask = tf.cast(tf.greater(self.lm_x, tf.zeros(tf.shape(self.lm_x), dtype=tf.int32)), dtype=tf.float32)

        #embedding lookup
        word_inputs = tf.nn.embedding_lookup(self.word_embedding, self.lm_x)
        if is_training and cf.keep_prob < 1.0:
            word_inputs = tf.nn.dropout(word_inputs, cf.keep_prob)

        #process character encodings
        #concat last hidden state of fw RNN with first hidden state of bw RNN
        #fw_hidden = self.get_last_hidden(self.char_encodings[0], self.pm_enc_xlen)
        #char_inputs = tf.concat(1, [fw_hidden, self.char_encodings[1][:,0,:]])
        #char_inputs = tf.reshape(char_inputs, [batch_size, -1, cf.pm_enc_dim*2]) #reshape into same dimension as inputs
        
        #concat word and char encodings
        #inputs = tf.concat(2, [word_inputs, char_inputs])
        inputs = word_inputs

        #apply mask to zero out pad embeddings
        inputs = inputs * tf.expand_dims(lm_mask, -1)

        #dynamic rnn
        dec_outputs, final_state = tf.nn.dynamic_rnn(self.lm_dec_cell, inputs, sequence_length=self.lm_xlen, \
            dtype=tf.float32, initial_state=self.lm_initial_state)
        self.lm_final_state = final_state

        #########################
        #encoder (history words)#
        #########################

        #embedding lookup
        hist_inputs = tf.nn.embedding_lookup(self.word_embedding, self.lm_hist)
        if is_training and cf.keep_prob < 1.0:
            hist_inputs = tf.nn.dropout(hist_inputs, cf.keep_prob)

        #encoder lstm cell
        lm_enc_cell = tf.nn.rnn_cell.LSTMCell(cf.lm_enc_dim, forget_bias=1.0)
        if is_training and cf.keep_prob < 1.0:
            lm_enc_cell = tf.nn.rnn_cell.DropoutWrapper(lm_enc_cell, output_keep_prob=cf.keep_prob)

        #history word encodings
        hist_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lm_enc_cell, cell_bw=lm_enc_cell,
            inputs=hist_inputs, sequence_length=self.lm_hlen, dtype=tf.float32)

        #full history encoding
        full_encoding = tf.concat( [hist_outputs[0][:,-1,:], hist_outputs[1][:,0,:]], axis=1)

        #concat fw and bw hidden states
        hist_outputs = tf.concat(hist_outputs, axis=2)

        #selective encoding
        with tf.variable_scope("selective_encoding"):
            hist_outputs = self.selective_encoding(hist_outputs, full_encoding, cf.lm_enc_dim*2)

        #attention (concat)
        with tf.variable_scope("lm_attention"):
            attend_w = tf.get_variable("attend_w", [cf.lm_enc_dim*2+cf.lm_dec_dim, cf.lm_attend_dim])
            attend_b = tf.get_variable("attend_b", [cf.lm_attend_dim], initializer=tf.constant_initializer())
            attend_v = tf.get_variable("attend_v", [cf.lm_attend_dim, 1])

        enc_steps = tf.shape(hist_outputs)[1]
        dec_steps = tf.shape(dec_outputs)[1]

        #prepare encoder and decoder
        hist_outputs_t = tf.tile(hist_outputs, [1, dec_steps, 1])
        dec_outputs_t  = tf.reshape(tf.tile(dec_outputs, [1, 1, enc_steps]),
            [batch_size, -1, cf.lm_dec_dim])

        #compute e
        hist_dec_concat = tf.concat( [tf.reshape(hist_outputs_t, [-1, cf.lm_enc_dim*2]),
            tf.reshape(dec_outputs_t, [-1, cf.lm_dec_dim])], 1)
        e = tf.matmul(tf.tanh(tf.matmul(hist_dec_concat, attend_w) + attend_b), attend_v)
        e = tf.reshape(e, [-1, enc_steps])

        #mask out pad symbols to compute alpha and weighted sum of history words
        alpha    = tf.reshape(tf.nn.softmax(e), [batch_size, -1, 1])
        context  = tf.reduce_sum(tf.reshape(alpha * hist_outputs_t,
            [batch_size,dec_steps,enc_steps,-1]), 2)

        #save attention weights
        self.lm_attentions = tf.reshape(alpha, [batch_size, dec_steps, enc_steps])

        ##############
        #output layer#
        ##############

        #reshape both into [batch_size*len, hidden_dim]
        dec_outputs = tf.reshape(dec_outputs, [-1, cf.lm_dec_dim])
        context     = tf.reshape(context, [-1, cf.lm_enc_dim*2])
        
        #combine context and decoder hidden state with a gated unit
        with tf.variable_scope("gated_unit"):
            hidden = self.gated_layer(context, dec_outputs, cf.lm_enc_dim*2, cf.lm_dec_dim)
            #hidden = dec_outputs

        #output embeddings
        lm_output_proj = tf.get_variable("lm_output_proj", [cf.word_embedding_dim, cf.lm_dec_dim])
        lm_softmax_b   = tf.get_variable("lm_softmax_b", [word_type_size], initializer=tf.constant_initializer())
        lm_softmax_w   = tf.transpose(tf.tanh(tf.matmul(self.word_embedding, lm_output_proj)))

        #compute logits and cost
        lm_logits     = tf.matmul(hidden, lm_softmax_w) + lm_softmax_b
        lm_crossent   = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(self.lm_y, [-1]))
        lm_crossent_m = lm_crossent * tf.reshape(lm_mask, [-1])
        self.lm_cost  = tf.reduce_sum(lm_crossent_m) / batch_size

        if not is_training:
            self.lm_probs = tf.nn.softmax(lm_logits)
            return

        #run optimiser and backpropagate (clipped) gradients for lm loss
        lm_tvars = tf.trainable_variables()
        lm_grads, _ = tf.clip_by_global_norm(tf.gradients(self.lm_cost, lm_tvars), cf.max_grad_norm)
        self.lm_train_op = tf.train.AdagradOptimizer(cf.lm_learning_rate).apply_gradients(zip(lm_grads, lm_tvars))



    



    # -- sample a word given probability distribution (with option to normalise the distribution with temperature)
    # -- temperature = 0 means argmax
    def sample_word(self, sess, probs, temperature, unk_symbol_id, pad_symbol_id, idxword, avoid_words):


        if temperature == 0:
            return np.argmax(probs)

        probs = probs.astype(np.float64) #convert to float64 for higher precision
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / math.fsum(np.exp(probs))

        #avoid unk_symbol_id if possible
        sampled = None
        
        for i in range(1000):
            sampled = np.argmax(np.random.multinomial(1, probs, 1))

            #resample if it's a word to avoid
            if sampled in avoid_words:
                continue

            return sampled

        return None


    # -- generate a sentence by sampling one word at a time
    def sample_sent(self, sess, state, x, hist, hlen, avoid_symbols, stopwords,temp_min, temp_max,
        unk_symbol_id, pad_symbol_id, end_symbol_id, space_id, idxword, last_words, max_words):

        def filter_stop_symbol(word_ids):
            cleaned = set([])
            for w in word_ids:
                if w not in (stopwords | set([pad_symbol_id, end_symbol_id])) and not only_symbol(idxword[w]):
                    cleaned.add(w)
            return cleaned

        def get_freq_words(word_ids, freq_threshold):
            words     = []
            word_freq = Counter(word_ids)
            for k, v in word_freq.items():
                #if v >= freq_threshold and not only_symbol(idxword[k]) and k != end_symbol_id:
                if v >= freq_threshold and k != end_symbol_id:
                    words.append(k)
            return set(words)

        sent   = []

        while True:
            probs, state = sess.run([self.lm_probs, self.lm_final_state],
                {self.lm_x: x, self.lm_initial_state: state, self.lm_xlen: [1],
                self.lm_hist: hist, self.lm_hlen: hlen})

            #avoid words previously generated
            avoid_words = filter_stop_symbol(sent + hist[0])
            freq_words  = get_freq_words(sent + hist[0], 2) #avoid any words that occur >= N times
            avoid_words = avoid_words | freq_words | set(sent[-3:] + last_words + avoid_symbols + [unk_symbol_id])
            
            
            #self, sess, probs, temperature, unk_symbol_id, pad_symbol_id, wordxchar, idxword, avoid_words
            word = self.sample_word(sess, probs[0], np.random.uniform(temp_min, temp_max), unk_symbol_id,
                pad_symbol_id, idxword, avoid_words)

            if word != None:
                sent.append(word)
                x             = [[ sent[-1] ]]
                
                
            else:
                return None, None
                
            if sent[-1] == end_symbol_id or len(sent) >= max_words:

                if len(sent) > 1:
                    return sent, state
                else:
                    return None, None
    



    # -- quatrain generation
    def generate(self, sess, idxword, pad_symbol_id, end_symbol_id, unk_symbol_id, space_id,
        avoid_symbols, stopwords, temp_min, temp_max, max_lines, max_words, sent_sample, verbose=False):

        def reset():
            state       = sess.run(self.lm_dec_cell.zero_state(1, tf.float32))
            prev_state  = state
            x           = [[end_symbol_id]]
            sonnet      = []
            sent_probs  = []
            last_words  = []
            total_words = 0
            total_lines = 0
            
            return state, prev_state, x, sonnet, sent_probs, last_words, total_words, total_lines

        end_symbol = idxword[end_symbol_id]
        sent_temp  = 0.1 #sentence sampling temperature

        state, prev_state, x, sonnet, sent_probs, last_words, total_words, total_lines = reset()

        #verbose prints during generation
        if verbose:
            sys.stdout.write("  Number of generated lines = 0/4\r")
            sys.stdout.flush()

        while total_words < max_words and total_lines < max_lines:

            #add history context
            if len(sonnet) == 0 or sonnet.count(end_symbol_id) < 1:
                hist = [[unk_symbol_id] + [pad_symbol_id]*5]
            else:
                hist = [sonnet + [pad_symbol_id]*5]
            hlen = [len(hist[0])]

            

            #genereate N sentences and sample from them (using softmax(-one_pl) as probability)
            compteur = 0
            all_sent, all_state = [], []
            for _ in range(sent_sample):
            
           
                one_sent, one_state = self.sample_sent(sess, state, x, hist, hlen, avoid_symbols, stopwords, temp_min, temp_max, unk_symbol_id, pad_symbol_id, end_symbol_id, space_id, idxword, last_words, max_words)
                    
                if one_sent != None:
                    all_sent.append(one_sent)
                    all_state.append(one_state)
                    compteur += 1
                    
                else:
                    all_sent = []
                    break

            #unable to generate sentences; reset whole quatrain
            if len(all_sent) == 0:

                state, prev_state, x, sonnet, sent_probs, last_words, total_words, total_lines = reset()

            else:
                
                probs = [1/compteur] * compteur

                #sample a sentence
                sent_id = np.argmax(np.random.multinomial(1, probs, 1))
                sent    = all_sent[sent_id]
                state   = all_state[sent_id]

                total_words += len(sent)
                total_lines += 1
                prev_state   = state

                sonnet.extend(sent)
                last_words.append(sent[0])

            if verbose:
                sys.stdout.write("  Number of generated lines = %d/4\r" % (total_lines))
                sys.stdout.flush()

        #postprocessing
        sonnet = sonnet[:-1] if sonnet[-1] == end_symbol_id else sonnet
        sonnet = [ postprocess_sentence(item) for item in \
            " ".join(list(reversed([ idxword[item] for item in sonnet ]))).strip().split(end_symbol) ]

        return sonnet, list(reversed(sent_probs))

