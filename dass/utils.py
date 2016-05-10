import argparse

def get_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', action='store', dest='batch_size',help='batch-size , default 64',type=int,default = 64)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 500',type=int,default = 500)

	parser.add_argument('--patience', action='store', dest='patience',help='# of epochs for patience, default = 20',type=int,default = 20)

	parser.add_argument('--dropout', action='store', dest='dropout',help='dropout rate, default = 0.2',type=float,default = 0.2)

	parser.add_argument('--layers', action='store', dest='layers',help='# of hidden layers, default = 1',type=int,default = 1)

	parser.add_argument('--unit', action='store', dest='unit',help='train with {lstm gru rnn} units,default = lstm', default = 'lstm')

	parser.add_argument('--hidden', action='store', dest='n_hidden',help='hidden size of neural networks, default = 256',type=int,default = 256)

	parser.add_argument('--prefix', action='store', dest='prefix',help='exp log prefix to append exp/{} default = 0',default = '0')

	parser.add_argument('--train', action='store', dest='train',help='train data : {0 : train2393.cleanup.xml , 1 : train-less-than-40.manual-edit.xml} ,default : 0',default = '0')

	parser.add_argument('--model', action='store', dest='model',help='the type of the model {0,1,2}, default = 0', type = int, default = 0)

	parser.add_argument('--finetune', action='store_true', dest='finetune',help='finetune word embeddings')

	parser.add_argument('--bucket', action='store_false', dest='bucket',help='do not use buckets')

	parser.set_defaults(finetune = False)
	parser.set_defaults(buckets = True)

	return parser

def get_parser_test():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', action='store', dest='batch_size',help='batch-size , default 64',type=int,default = 64)

	parser.add_argument('--path', action='store', dest='path',help='path ',default = '')
	parser.add_argument('--dataset', action='store', dest='dataset',help='dataset liveqa - factual, default liveqa ',default = 'liveqa')

	return parser
