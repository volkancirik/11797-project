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
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout
from keras.layers import recurrent
from keras.optimizers import RMSprop
from keras.regularizers import l1l2

from prepare_data import prepare_ass
from utils import get_parser
'''
baseline model - wvec sum
'''
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
PREFIX = 'exp/'+p.prefix + '/'
os.system('mkdir -p '+PREFIX)
FOOTPRINT = 'H' + str(HIDDEN_SIZE) + '_L' + str(LAYERS) + '_D' + str(DROPOUT)

### get data
X_tr, Y_tr, X_val, Y_val, X_test,Y_test = prepare_ass()
print('data loaded...')

DIM = X_tr[0][0].shape[2]
N_tr = len(X_tr)
N_val = len(X_val)

print('building model...')
model = Sequential()
model.add(Dense(HIDDEN_SIZE, input_shape = (DIM,),W_regularizer= l1l2(l1 = 0.00001, l2 = 0.00001)))
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

	e_loss = []

	for i in xrange(N_tr):
		x_q,x_a = X_tr[i]
		x = np.sum(x_q,axis=1) + np.sum(x_a,axis=1)
		y = Y_tr[i].reshape((1,1))
		eh = model.fit(x, y, batch_size = 1, nb_epoch=1, verbose = 0)
		e_loss += eh.history['loss']

	e_val_loss = []
	for i in xrange(N_val):
		x_q,x_a = X_val[i]
		x = np.sum(x_q,axis=1) + np.sum(x_a,axis=1)
		y = Y_val[i].reshape((1,1))
		e_val_loss += [model.evaluate(x, y, batch_size = 1, verbose = 0)]

	train_history['loss'] += [np.mean(e_loss)]
	train_history['val_loss'] += [ np.mean(e_val_loss)]

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

outfile = open( PREFIX + FOOTPRINT + '.output','w')
for i in xrange(len(X_test)):
	x_q,x_a = X_test[i]
	y = Y_test[i].reshape((1,1))
	x = np.sum(x_q,axis=1) + np.sum(x_a,axis=1)
	test = model.predict(x, batch_size = 1, verbose = 0)
	test_c = model.predict_classes(x, batch_size = 1, verbose = 0)
	outfile.write("\t".join([str(test[0][0]),str(test_c[0][0]),str(y[0][0])+'\n']))

print("write experiment log...")
pickle.dump({'train_history' : train_history},open(PREFIX + FOOTPRINT + '.meta', 'w'))
