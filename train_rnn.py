import csv
import nltk
import itertools
import numpy as np
import datetime, time
import sys
import os
import utils_rnn
from rnn_theano import RNNTheano


_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_N_EPOCH = int(os.environ.get('N_EPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

vocabulary_size = 8000
unknown_tok = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading CSV file...")
with open("data/reddit-comments-2015-08.csv", "rb") as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # split comnts into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

# gimme faster results
sentences = sentences[:len(sentences)/4]

print("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
# first elem outputs sth like [u'SENTENCE_START', u'i', u'joined', u'a', u'new', ...]
tokenized_sents = [nltk.word_tokenize(sent) for sent in sentences]
print(tokenized_sents[0])

# Count word freqs
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sents))
print("Found %d unique word tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
# line below outputs sth like (u'SENTENCE_START', 79170)
vocab = word_freq.most_common(vocabulary_size - 1)
print(vocab[0])
index_to_word = [x[0] for x in vocab]
# so now the first elem is the most frequent and so on
index_to_word.append(unknown_tok)
word_to_index = dict([(word, index) for index, word in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# replace all words not in vocab with unknown tok
for i, sent in enumerate(tokenized_sents):
    tokenized_sents[i] = [word if word in word_to_index else unknown_tok for word in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sents[0])

# Create the training data
# sent[:-1] is from the beginning upto second to last word
X_train = np.asarray([[word_to_index[word] for word in sent[:-1]] for sent in tokenized_sents])
Y_train = np.asarray([[word_to_index[word] for word in sent[1:]] for sent in tokenized_sents])

print("X for an example in training set: %s", X_train[0])
print("Y for that example: %s", Y_train[0])


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, n_epoch=50, eval_loss_after_n_epoch=1):
    losses = []
    n_examps_seen = 0
    for epoch in range(n_epoch):
        if epoch % eval_loss_after_n_epoch == 0:
            loss = model.calculate_avg_loss(X_train, y_train)
            losses.append((n_examps_seen, loss))
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after n_examps_seen=%d epoch=%d: %f" % (time, n_examps_seen, epoch, loss))
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate *= 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
            utils_rnn.save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz"
                                                   % (model.hidden_dim, model.word_dim, time), model)
            # now do the sgd steps for all the epoch
        for i in range(len(y_train)):
            # one step sgd with just one sample
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            n_examps_seen += 1
            if n_examps_seen % 1000 == 0:
                print("%s examples seen..." % n_examps_seen)

model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
t1 = time.time()
model.sgd_step(X_train[10], Y_train[10], _LEARNING_RATE)
t2 = time.time()
print("SGD step time: %f milliseconds" % ((t2 - t1) * 1000))

if _MODEL_FILE is not None:
    utils_rnn.load_model_parameters_theano(_MODEL_FILE, model)

train_with_sgd(model, X_train, Y_train, n_epoch=_N_EPOCH, learning_rate=_LEARNING_RATE)

