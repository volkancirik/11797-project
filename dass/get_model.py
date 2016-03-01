from keras.models import Graph
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout, Merge
from keras.layers import recurrent
from keras.regularizers import l1l2

from keras.layers.attention import TimeDistributedAttention, PointerPrediction

RNN_UNIT = {'rnn' : recurrent.SimpleRNN, 'gru' : recurrent.GRU, 'lstm' : recurrent.LSTM}

def rnn_0(DIM = 0, HIDDEN_SIZE = 0 , DROPOUT = 0, LAYERS = 1, UNIT = 'lstm', MAX_Q = 0, MAX_A = 0):
	RNN = RNN_UNIT[UNIT]
	model = Graph()
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

def rnn_1(DIM = 0, HIDDEN_SIZE = 0 , DROPOUT = 0, LAYERS = 1, UNIT = 'lstm', MAX_Q = 0, MAX_A = 0):
	LAYERS = max(LAYERS,min(2,LAYERS))
	RNN = RNN_UNIT[UNIT]
	model = Graph()
	model.add_input(name = 'q', input_shape = (None,DIM))
	model.add_input(name = 'a', input_shape = (MAX_A,DIM))

	prev_q = 'q'

	for layer in xrange(LAYERS -1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), name = 'q_rnn_' + str(layer+1), input = prev_q)
		prev_q = 'q_rnn_' + str(layer+1)
	model.add_node(RNN(HIDDEN_SIZE, return_sequences=False), name = 'q_rnn_' + str(LAYERS), input = prev_q)
	model.add_node(RepeatVector(MAX_A), input = 'q_rnn_' + str(LAYERS), name = 'q_rv')
	model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), name = 'a_rnn_0', inputs = ['q_rv' , 'a'], merge_mode = 'concat', concat_axis = -1)
	prev_a = 'a_rnn_0'
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

def rnn_2(DIM = 0, HIDDEN_SIZE = 0 , DROPOUT = 0, LAYERS = 1, UNIT = 'lstm', MAX_Q = 0, MAX_A = 0):
	LAYERS = max(LAYERS,min(2,LAYERS))
	RNN = RNN_UNIT[UNIT]
	model = Graph()
	model.add_input(name = 'q', input_shape = (None,DIM))
	model.add_input(name = 'a', input_shape = (None,DIM))
	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='recurrent_context', input='a')

	prev_q = 'q'

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

def get_model(model_id, DIM = 0, HIDDEN_SIZE = 0, LAYERS = 0, UNIT = 'lstm', MAX_Q = 0, MAX_A = 0):
	M = {0 : rnn_0, 1 : rnn_1, 2 : rnn_2}
	return M[model_id](DIM = DIM, HIDDEN_SIZE = HIDDEN_SIZE, LAYERS = LAYERS, UNIT = UNIT, MAX_Q = MAX_Q, MAX_A = MAX_A)
