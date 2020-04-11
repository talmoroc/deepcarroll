import config_etienne as cf
import sys
import codecs
import random
import time
import imp
import os
import tensorflow as tf
# tf.disable_eager_execution()
# tf.disable_v2_behavior()
import numpy as np
import gensim.models as g

from sonnet_model import SonnetModel
from RUN_epoch import run_epoch

from util import *

#constants
pad_symbol = "<pad>"
end_symbol = "<eos>"
unk_symbol = "<unk>"
dummy_symbols = [pad_symbol, end_symbol, unk_symbol]

#globals
wordxid = None
idxword = None
charxid = None
idxchar = None
wordxchar = None #word id to [char ids]
rhyme_thresholds = [0.9, 0.8, 0.7, 0.6]
stress_acc_threshold = 0.4
reset_scale = 1.05


sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

#set the seeds
random.seed(cf.seed)
np.random.seed(cf.seed)

#load word embedding model if given and set word embedding size
if cf.word_embedding_model:
    #print("\nLoading word embedding model...")
    mword = g.Word2Vec.load(cf.word_embedding_model)
    cf.word_embedding_dim= mword.vector_size

#load vocab
print("\n", "First pass to collect word and character vocabulary...")
idxword, wordxid, idxchar, charxid, wordxchar = load_vocab(cf.train_data, cf.word_minfreq, dummy_symbols)
print("\nWord type size =", len(idxword))
print("\nChar type size =", len(idxchar))

#load train and valid data
print("\n Loading train and valid data...")
train_word_data, train_char_data, train_nwords, train_nchars, train_rhyme_data = \
    load_data(cf.train_data, wordxid, idxword, charxid, idxchar, dummy_symbols)
# a rajouter


valid_word_data, valid_char_data, valid_rhyme_data, valid_nwords, valid_nchars = \
    load_data(cf.valid_data, wordxid, idxword, charxid, idxchar, dummy_symbols)
print_stats("\nTrain", train_word_data, train_nwords, train_nchars, train_rhyme_data)
print_stats("\nValid", valid_word_data, valid_rhyme_data, valid_nwords, valid_nchars)

#load test data if it's given
if cf.test_data:
    test_word_data, test_char_data, test_rhyme_data, test_nwords, test_nchars = \
        load_data(cf.test_data, wordxid, idxword, charxid, idxchar, dummy_symbols)
    print_stats("\nTest", test_word_data, test_rhyme_data, test_nwords, test_nchars)

if cf.word_embedding_model:
            word_emb = init_embedding(mword, idxword)

word_emb_dict = dict(zip(idxword, word_emb))




from collections import OrderedDict

def closest_word(word):
    
    dico = {}
    
    def cosinus_similarity(word1, word2):
        score =  np.dot(word_emb_dict[word1], word_emb_dict[word2])/(np.linalg.norm(word_emb_dict [word1])* \
                                                                         np.linalg.norm(word_emb_dict [word2]))
        return {word2 : score}
    
    
    for ele in word_emb_dict.keys():
        dico.update(cosinus_similarity(word, ele))
    
    sorted_dico = sorted(dico.items(), key=lambda kv: kv[1], reverse=True)
        
    return OrderedDict(sorted_dico)


create_word_batch(data=train_word_data, batch_size=32, lines_per_doc=14, nlines_per_batch=2, pad_symbol=pad_symbol,\
                  end_symbol=end_symbol, unk_symbol=unk_symbol, shuffle_data=True)[0]



with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        tf.set_random_seed(cf.seed)
        with tf.variable_scope("model", reuse=None):
            mtrain = SonnetModel(True, cf.batch_size, len(idxword), len(idxchar),
                charxid[" "], charxid[pad_symbol], cf)
        with tf.variable_scope("model", reuse=True):
            mvalid = SonnetModel(False, cf.batch_size, len(idxword), len(idxchar),
                charxid[" "], charxid[pad_symbol], cf)
        with tf.variable_scope("model", reuse=True):
            mgen = SonnetModel(False, 1, len(idxword), len(idxchar), charxid[" "], charxid[pad_symbol], cf)

        tf.compat.v1.global_variables_initializer().run()

        #initialise word embedding
        if cf.word_embedding_model:
            word_emb = init_embedding(mword, idxword)
            sess.run(mtrain.word_embedding.assign(word_emb))


        if cf.save_model:
            if not os.path.exists(cf.output_dir):
                os.makedirs(cf.output_dir)
            #create saver object to save model
            saver = tf.compat.v1.train.Saver(max_to_keep=0)


        #train model
        prev_lm_loss = None 

        for i in range(cf.epoch_size):

            print("\nEpoch =", i+1)

            #create batches for language model
            train_word_batch = create_word_batch(train_word_data, cf.batch_size,
                cf.doc_lines, cf.bptt_truncate, wordxid[pad_symbol], wordxid[end_symbol], wordxid[unk_symbol], True)

            valid_word_batch = create_word_batch(valid_word_data, cf.batch_size,
                cf.doc_lines, cf.bptt_truncate, wordxid[pad_symbol], wordxid[end_symbol], wordxid[unk_symbol], False)

            #train an epoch
            _ = run_epoch(sess, train_word_batch, mtrain, "TRAIN", True)
            lm_loss = run_epoch(sess, valid_word_batch, mvalid, "VALID", False)

            #create batch for test model and run an epoch if it's given
            if cf.test_data:
                test_word_batch = create_word_batch(test_word_data, cf.batch_size,
                    cf.doc_lines, cf.bptt_truncate, wordxid[pad_symbol], wordxid[end_symbol], wordxid[unk_symbol], False)
                run_epoch(sess, test_word_batch, mvalid, "TEST", False)


            #We save
            if cf.save_model:
                    if prev_lm_loss == None  or lm_loss <= prev_lm_loss : #or not train_lm :
                        saver.save(sess, os.path.join(cf.output_dir, "model.ckpt"))
                        prev_lm_loss = lm_loss

                    else:
                        saver.restore(sess, os.path.join(cf.output_dir, "model.ckpt"))
                        print("New valid performance is worse; restoring previous parameters...")
                        print("  lm loss: %.5f --> %.5f" % (prev_lm_loss, lm_loss))



#############################################
############### GENERATION ##################
#############################################

import os
import sys
import random
import codecs
import numpy as np
import tensorflow as tf
from collections import namedtuple
from sonnet_model import SonnetModel
from nltk.corpus import stopwords as nltk_stopwords
from util import *

#constants
seed =  2
num_samples = 1
#save_pickle = os.path(output_dir)
temp_min = 0.6
temp_max = 0.8
sent_sample = 10
verbose = False
pad_symbol = "<pad>"
end_symbol = "<eos>"
unk_symbol = "<unk>"
dummy_symbols = [pad_symbol, end_symbol, unk_symbol]
custom_stopwords = [ "thee", "thou", "thy", "'d", "'s", "'ll", "must", "shall" ]

###########
#functions#
###########

def reverse_dic(idxvocab):
    vocabxid = {}
    for vi, v in enumerate(idxvocab):
        vocabxid[v] = vi

    return vocabxid

######
#main#
######

def main():

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

    #set the seeds
    random.seed(seed)
    np.random.seed(seed)


    #symbols to avoid for generation
    avoid_symbols = ["(", ")", "“", "‘", "”", "’", "[", "]"]
    avoid_symbols = [ wordxid[item] for item in avoid_symbols ]
    stopwords = set([ wordxid[item] for item in (nltk_stopwords.words("english") + custom_stopwords) if item in wordxid ])

    quatrains = []
    #initialise and load model parameters
    with tf.Graph().as_default(), tf.Session() as sess:
        tf.set_random_seed(seed)

        with tf.variable_scope("model", reuse=None):
            mtest = SonnetModel(False, cf.batch_size, len(idxword), len(idxchar), charxid[" "], charxid[pad_symbol], cf)

        with tf.variable_scope("model", reuse=True):
            mgen = SonnetModel(False, 1, len(idxword), len(idxchar), charxid[" "], charxid[pad_symbol], cf)

        #load tensorflow model
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(cf.output_dir, "model.ckpt"))

        #quatrain generation
        for _ in range(num_samples):

            #generate some random sentences
            #print("\nTemperature =", temp_min, "-", temp_max)

            q, probs = mgen.generate(sess, idxword, wordxid[pad_symbol],
                wordxid[end_symbol], wordxid[unk_symbol], charxid[" "], avoid_symbols, stopwords,
                temp_min, temp_max, 12, 400, sent_sample, verbose)

            quatrains.append(q)
        return quatrains

main()

