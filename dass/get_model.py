from keras.models import Graph
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout, Merge
from keras.layers import recurrent
from keras.regularizers import l1l2
from keras.layers.embeddings import Embedding

from keras.layers.attention import TimeDistributedAttention, PointerPrediction

RNN_UNIT = {'rnn' : recurrent.SimpleRNN, 'gru' : recurrent.GRU, 'lstm' : recurrent.LSTM}

def dan(DIM = 0, HIDDEN_SIZE = 0 , DROPOUT = 0, LAYERS = 1, UNIT = 'lstm', MAX_Q = 0, MAX_A = 0, FINETUNE = False, embedding_weights = None):
	from keras.layers.averagelayer import Average
	model = Graph()

	if FINETUNE:
		model.add_input(name = 'q', input_shape = (None,), dtype = 'int64')
		model.add_input(name = 'a', input_shape = (None,), dtype = 'int64')
		VOCAB = embedding_weights.shape[0]
		EMB_HIDDEN_SIZE = embedding_weights.shape[1]
		model.add_node(Embedding(VOCAB,EMB_HIDDEN_SIZE,mask_zero=True, weights=[embedding_weights]), name = 'q_e', input = 'q')
		model.add_node(Embedding(VOCAB,EMB_HIDDEN_SIZE,mask_zero=True, weights=[embedding_weights]), name = 'a_e', input = 'a')
		prev_q = 'q_e'
		prev_a = 'a_e'
	else:
		model.add_input(name = 'q', input_shape = (None,DIM))
		model.add_input(name = 'a', input_shape = (None,DIM))
		prev_q = 'q'
		prev_a = 'a'
		EMB_HIDDEN_SIZE = DIM

#	model.add_node(Average(),name = 'avg_a', inputs = [prev_a], merge_mode = 't_ave')
#	model.add_node(Average(), name = 'avg_q', inputs = [prev_q], merge_mode = 't_ave')
	model.add_node(Average(),name = 'avg_a', input = prev_a)
	model.add_node(Average(), name = 'avg_q', input = prev_q)

	model.add_node(Dense(HIDDEN_SIZE, activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'd_0', inputs = ['avg_a','avg_q'], merge_mode = 'concat',  concat_axis = -1)
	model.add_node(Dropout(DROPOUT), name = 'd_0_dr', input = 'd_0')
	prev = 'd_0_dr'
	for layer in xrange(LAYERS-1):
		model.add_node(Dense(HIDDEN_SIZE, activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'd_' + str(layer+1), input = prev)
		model.add_node(Dropout(DROPOUT), name = 'd_' + str(layer+1) + '_dr', input = 'd_' + str(layer+1))
		prev = 'd_' + str(layer+1) + '_dr'

	model.add_node(Dense(1, activation = 'sigmoid'), name = 'sigmoid', input = prev)
	model.add_output(name = 'o', input = 'sigmoid')
	return model



def rnn_0(DIM = 0, HIDDEN_SIZE = 0 , DROPOUT = 0, LAYERS = 1, UNIT = 'lstm', MAX_Q = 0, MAX_A = 0, FINETUNE = False, embedding_weights = None):
	RNN = RNN_UNIT[UNIT]
	model = Graph()

	if FINETUNE:
		model.add_input(name = 'q', input_shape = (None,), dtype = 'int64')
		model.add_input(name = 'a', input_shape = (None,), dtype = 'int64')
		VOCAB = embedding_weights.shape[0]
		EMB_HIDDEN_SIZE = embedding_weights.shape[1]
		model.add_node(Embedding(VOCAB,EMB_HIDDEN_SIZE,mask_zero=True, weights=[embedding_weights]), name = 'q_e', input = 'q')
		model.add_node(Embedding(VOCAB,EMB_HIDDEN_SIZE,mask_zero=True, weights=[embedding_weights]), name = 'a_e', input = 'a')
		prev_q = 'q_e'
		prev_a = 'a_e'
	else:
		model.add_input(name = 'q', input_shape = (None,DIM))
		model.add_input(name = 'a', input_shape = (None,DIM))
		prev_q = 'q'
		prev_a = 'a'

	for layer in xrange(LAYERS -1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), name = 'q_rnn_' + str(layer+1), input = prev_q)
		model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), name = 'a_rnn_' + str(layer+1), input = prev_a)
		prev_q = 'q_rnn_' + str(layer+1)
		prev_a = 'a_rnn_' + str(layer+1)

	model.add_node(RNN(HIDDEN_SIZE, return_sequences=False), name = 'q_rnn_' + str(LAYERS), input = prev_q)
	model.add_node(RNN(HIDDEN_SIZE, return_sequences=False),name = 'a_rnn_' + str(LAYERS), input = prev_a)

	model.add_node(Dense(HIDDEN_SIZE,  activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'dense_0', inputs = ['q_rnn_' + str(LAYERS), 'a_rnn_' + str(LAYERS)], merge_mode = 'concat',  concat_axis = -1 )
	model.add_node(Dropout(DROPOUT), name = 'dropout_0', input = 'dense_0')

	prev_d = 'dropout_0'
	for layer in xrange(LAYERS-1):
		model.add_node(Dense(HIDDEN_SIZE, activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'dense_' + str(layer+1), input = prev_d)
		model.add_node(Dropout(DROPOUT), name = 'dropout_' + str(layer+1), input = 'dense_' + str(layer+1))
		prev_d = 'dropout_' + str(layer+1)

	model.add_node(Dense(1, activation = 'sigmoid'), name = 'sigmoid', input = prev_d)
	model.add_output(name = 'o', input = 'sigmoid')
	return model

def rnn_1(DIM = 0, HIDDEN_SIZE = 0 , DROPOUT = 0, LAYERS = 1, UNIT = 'lstm', MAX_Q = 0, MAX_A = 0, FINETUNE = False, embedding_weights = None):
	RNN = RNN_UNIT[UNIT]
	model = Graph()

	if FINETUNE:
		model.add_input(name = 'q', input_shape = (None,), dtype = 'int64')
		model.add_input(name = 'a', input_shape = (MAX_A,), dtype = 'int64')
		VOCAB = embedding_weights.shape[0]
		EMB_HIDDEN_SIZE = embedding_weights.shape[1]
		model.add_node(Embedding(VOCAB,EMB_HIDDEN_SIZE,mask_zero=True, weights=[embedding_weights], input_length = MAX_Q ), name = 'q_e', input = 'q')
		model.add_node(Embedding(VOCAB,EMB_HIDDEN_SIZE,mask_zero=True, weights=[embedding_weights], input_length = MAX_A ), name = 'a_e', input = 'a')
		prev_q = 'q_e'
		prev_a = 'a_e'
	else:
		model.add_input(name = 'q', input_shape = (None,DIM))
		model.add_input(name = 'a', input_shape = (MAX_A,DIM))
		prev_q = 'q'
		prev_a = 'a'

	for layer in xrange(LAYERS-1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), name = 'q_rnn_' + str(layer), input = prev_q)
		prev_q = 'q_rnn_' + str(layer)

	model.add_node(RNN(HIDDEN_SIZE, return_sequences=False), name = 'q_rnn_' + str(LAYERS), input = prev_q)
	model.add_node(RepeatVector(MAX_A), input = 'q_rnn_' + str(LAYERS), name = 'q_rv')

	for layer in xrange(LAYERS-1):
		if layer == 0:
			model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), name = 'a_rnn_0', inputs = ['q_rv' , prev_a], merge_mode = 'concat', concat_axis = -1)
			prev_a = 'a_rnn_0'
			continue
		model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), name = 'a_rnn_' + str(layer), input = prev_a)
		prev_a = 'a_rnn_' + str(layer)

	if LAYERS == 1:
		model.add_node(RNN(HIDDEN_SIZE, return_sequences=False), name = 'a_rnn_0', inputs = ['q_rv' , prev_a], merge_mode = 'concat', concat_axis = -1)
	else:
		model.add_node(RNN(HIDDEN_SIZE, return_sequences=False),name = 'a_rnn_' + str(LAYERS-1), input = prev_a)

	model.add_node(Dense(HIDDEN_SIZE,  activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'dense_0', input = 'a_rnn_' + str(LAYERS-1))
	model.add_node(Dropout(DROPOUT), name = 'dropout_0', input = 'dense_0')

	prev_d = 'dropout_0'
	for layer in xrange(LAYERS-1):
		model.add_node(Dense(HIDDEN_SIZE, activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'dense_' + str(layer+1), input = prev_d)
		model.add_node(Dropout(DROPOUT), name = 'dropout_' + str(layer+1), input = 'dense_' + str(layer+1))
		prev_d = 'dropout_' + str(layer+1)

	model.add_node(Dense(1, activation = 'sigmoid'), name = 'sigmoid', input = prev_d)
	model.add_output(name = 'o', input = 'sigmoid')
	return model

def rnn_2(DIM = 0, HIDDEN_SIZE = 0 , DROPOUT = 0, LAYERS = 1, UNIT = 'lstm', MAX_Q = 0, MAX_A = 0, FINETUNE = False, embedding_weights = None):
	LAYERS = max(LAYERS,min(2,LAYERS))
	RNN = RNN_UNIT[UNIT]
	model = Graph()

	if FINETUNE:
		model.add_input(name = 'q', input_shape = (None,), dtype = 'int64')
		model.add_input(name = 'a', input_shape = (None,), dtype = 'int64')
		VOCAB = embedding_weights.shape[0]
		EMB_HIDDEN_SIZE = embedding_weights.shape[1]
		model.add_node(Embedding(VOCAB,EMB_HIDDEN_SIZE,mask_zero=True, weights=[embedding_weights]), name = 'q_e', input = 'q')
		model.add_node(Embedding(VOCAB,EMB_HIDDEN_SIZE,mask_zero=True, weights=[embedding_weights]), name = 'a_e', input = 'a')
		prev_q = 'q_e'
		prev_a = 'a_e'

	else:
		model.add_input(name = 'q', input_shape = (None,DIM))
		model.add_input(name = 'a', input_shape = (None,DIM))
		prev_q = 'q'
		prev_a = 'a'

	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='recurrent_context', input = prev_a)

	for layer in xrange(LAYERS - 2):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), name = 'q_rnn_' + str(layer+1), input = prev_q)
		prev_q = 'q_rnn_' + str(layer+1)
	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name = 'encoder_context', input = prev_q)
	model.add_node(TimeDistributedAttention(prev_dim = HIDDEN_SIZE, att_dim = HIDDEN_SIZE, return_sequences = True, prev_context = False), name='attention', inputs=['encoder_context','recurrent_context'], merge_mode = 'join_att')

	prev_a = 'attention'
	for layer in xrange(LAYERS - 2):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), name = 'a_rnn_' + str(layer+1), input = prev_a)
		prev_a = 'a_rnn_' + str(layer+1)
	model.add_node(RNN(HIDDEN_SIZE, return_sequences=False),name = 'a_rnn_' + str(LAYERS), input = prev_a)
	model.add_node(Dense(HIDDEN_SIZE,  activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'dense_0', input = 'a_rnn_' + str(LAYERS))
	model.add_node(Dropout(DROPOUT), name = 'dropout_0', input = 'dense_0')

	prev_d = 'dropout_0'
	for layer in xrange(LAYERS-1):
		model.add_node(Dense(HIDDEN_SIZE, activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'dense_' + str(layer+1), input = prev_d)
		model.add_node(Dropout(DROPOUT), name = 'dropout_' + str(layer+1), input = 'dense_' + str(layer+1))
		prev_d = 'dropout_' + str(layer+1)

	model.add_node(Dense(1, activation = 'sigmoid'), name = 'sigmoid', input = prev_d)
	model.add_output(name = 'o', input = 'sigmoid')
	return model

def get_model(model_id, DIM = 0, HIDDEN_SIZE = 0, LAYERS = 0, UNIT = 'lstm', MAX_Q = 0, MAX_A = 0, FINETUNE = False, embedding_weights = None):
	M = {-1 : dan, 0 : rnn_0, 1 : rnn_1, 2 : rnn_2}
	return M[model_id](DIM = DIM, HIDDEN_SIZE = HIDDEN_SIZE, LAYERS = LAYERS, UNIT = UNIT, MAX_Q = MAX_Q, MAX_A = MAX_A, FINETUNE = FINETUNE, embedding_weights = embedding_weights)
