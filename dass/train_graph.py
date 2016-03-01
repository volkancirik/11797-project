from __future__ import print_function

import numpy as np
import sys
import cPickle as pickle
import json, time, datetime, os

from prepare_data import *
from utils import *
import json, time, datetime, os
import cPickle as pickle

from keras.optimizers import RMSprop

from prepare_data import prepare_ass
from utils import get_parser
from get_model import get_model
'''
MMMT baseline model - basic enc-dec for MT
'''
TR = {'0' : 'train2393.cleanup.xml' , '1' : 'train-less-than-40.manual-edit.xml' }
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
FOOTPRINT = 'M' + str(p.model) + '_U' + str(p.unit) + '_H' + str(HIDDEN_SIZE) + '_L' + str(LAYERS) + '_D' + str(DROPOUT) + '_TR' + p.train

### get data
[X_tr_q,X_tr_a], Y_tr, [X_val_q,X_val_a], Y_val, [X_test_q,X_test_a],Y_test = prepare_ass(train = TR[p.train], mini_batch = True)

print('data loaded...')

DIM = X_tr_q.shape[2]
MAX_Q = X_tr_q.shape[1]
MAX_A = X_tr_a.shape[1]

print('building model...')
model = get_model(p.model, DIM = DIM, HIDDEN_SIZE = HIDDEN_SIZE, LAYERS = LAYERS, UNIT = p.unit, MAX_Q = MAX_Q, MAX_A = MAX_A)
optimizer = RMSprop(clipnorm = 5)
print('compiling model...')
model.compile(optimizer, {'o' : 'binary_crossentropy'})
print("save architecture...")
with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
	json.dump(model.to_json(), outfile)

print("training model with {} parameters...".format(model.get_n_params()))

pat = 0
train_history = {'loss' : [], 'val_loss' : []}
best_val_loss = float('inf')

for iteration in xrange(EPOCH):
	print('_' * 50)
	print('iteration {}/{}'.format(iteration+1,EPOCH))

	eh = model.fit({'q' : X_tr_q,'a' : X_tr_a, 'o' : Y_tr}, batch_size = BATCH_SIZE, nb_epoch=1, verbose = 1, validation_data = ({ 'q' : X_val_q, 'a' : X_val_a, 'o' : Y_val}))

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

pred = model.predict({'q' : X_test_q, 'a' : X_test_a}, batch_size = BATCH_SIZE, verbose = 0)
pred_c = np.greater_equal(pred['o'],0.5)
outfile = open( PREFIX + FOOTPRINT + '.output','w')

for i in xrange(X_test_q.shape[0]):
	outfile.write("\t".join([str(pred['o'][i][0]),str(int(pred_c[i][0])),str(int(Y_test[i][0]))+'\n']))

print("write experiment log...")
pickle.dump({'train_history' : train_history},open(PREFIX + FOOTPRINT + '.meta', 'w'))
