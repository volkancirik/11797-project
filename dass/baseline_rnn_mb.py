from __future__ import print_function

import numpy as np
import sys
import cPickle as pickle
import json, time, datetime, os

from prepare_data import *
from utils import *
import json, time, datetime, os
import cPickle as pickle

from keras.models import Sequential
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout, Merge
from keras.layers import recurrent
from keras.optimizers import RMSprop
from keras.regularizers import l1l2

from prepare_data import prepare_ass
from utils import get_parser

'''
MMMT baseline model - basic enc-dec for MT
'''
UNIT = {'rnn' : recurrent.SimpleRNN, 'gru' : recurrent.GRU, 'lstm' : recurrent.LSTM}

### get arguments
parser = get_parser()
p = parser.parse_args()

# Parameters
BATCH_SIZE = p.batch_size
EPOCH = p.n_epochs
DROPOUT = p.dropout
LAYERS = p.layers
PATIENCE = p.patience
HIDDEN_SIZE = p.n_hidden
RNN = UNIT[p.unit]
PREFIX = 'exp/'+p.prefix + '/'
os.system('mkdir -p '+PREFIX)
FOOTPRINT = 'H' + str(HIDDEN_SIZE) + '_L' + str(LAYERS) + '_D' + str(DROPOUT)

### get data
[X_tr_q,X_tr_a], Y_tr, [X_val_q,X_val_a], Y_val, [X_test_q,X_test_a],Y_test = prepare_ass(mini_batch = True)

print('data loaded...')

DIM = X_tr_q.shape[2]

print('building model...')

q_model = Sequential()
q_model.add(RNN(HIDDEN_SIZE, return_sequences=False,input_shape = (None,DIM)))

a_model = Sequential()
a_model.add(RNN(HIDDEN_SIZE, return_sequences=False,input_shape = (None,DIM)))

model = Sequential()
model.add(Merge([q_model, a_model], mode = 'concat', concat_axis = -1))

model.add(Dense(HIDDEN_SIZE,W_regularizer= l1l2(l1 = 0.00001, l2 = 0.00001)))
model.add(Activation('relu'))
for layer in xrange(LAYERS-1):
	model.add(Dense(HIDDEN_SIZE,W_regularizer= l1l2(l1 = 0.00001, l2 = 0.00001)))
	model.add(Activation('relu'))
	model.add(Dropout(DROPOUT))

model.add(Dense(1))
model.add(Activation('sigmoid'))
optimizer = RMSprop(clipnorm = 5)
print('compiling model...')
model.compile(optimizer = optimizer, loss='binary_crossentropy', class_mode="binary")

print("save architecture...")
with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
	json.dump(model.to_json(), outfile)

print("training model...")

pat = 0
train_history = {'acc' : [],'loss' : [], 'val_acc' : [], 'val_loss' : []}
best_val_loss = float('inf')

for iteration in xrange(EPOCH):
	print('_' * 50)
	print('iteration {}/{}'.format(iteration+1,EPOCH))

	eh = model.fit([X_tr_q,X_tr_a], Y_tr, batch_size = BATCH_SIZE, nb_epoch=1, verbose = 1, validation_data = ([X_val_q,X_val_a], Y_val))

 	train_history['loss'] += eh.history['loss']
 	train_history['val_loss'] += eh.history['val_loss'] 

 	print("TL {} bestVL {}, EVL : {} NO improv {} e".format(train_history['loss'][-1], best_val_loss, train_history['val_loss'][-1],pat))

	if train_history['val_loss'][-1] > best_val_loss:                          # is there improvement?
		pat += 1
	else:
		pat = 0
		best_val_loss = train_history['val_loss'][-1]
		model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
	if pat == PATIENCE:
		break

model.load_weights(PREFIX + FOOTPRINT + '.model')
print("predict test...")

pred = model.predict([X_test_q,X_test_a], batch_size = BATCH_SIZE, verbose = 0)
pred_c = model.predict_classes([X_test_q,X_test_a], batch_size = 128, verbose = 0)
outfile = open( PREFIX + FOOTPRINT + '.output','w')

for i in xrange(X_test_q.shape[0]):
	outfile.write("\t".join([str(pred[i][0]),str(pred_c[i][0]),str(int(Y_test[i][0]))+'\n']))

print("write experiment log...")
pickle.dump({'train_history' : train_history},open(PREFIX + FOOTPRINT + '.meta', 'w'))
