import argparse

def get_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', action='store', dest='batch_size',help='batch-size , default 128',type=int,default = 128)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 500',type=int,default = 500)

	parser.add_argument('--patience', action='store', dest='patience',help='# of epochs for patience, default = 20',type=int,default = 20)

	parser.add_argument('--dropout', action='store', dest='dropout',help='dropout rate, default = 0.2',type=float,default = 0.2)

	parser.add_argument('--layers', action='store', dest='layers',help='# of hidden layers, default = 1',type=int,default = 1)

	parser.add_argument('--unit', action='store', dest='unit',help='train with {lstm gru rnn} units,default = lstm', default = 'lstm')

	parser.add_argument('--hidden', action='store', dest='n_hidden',help='hidden size of neural networks, default = 256',type=int,default = 256)

	parser.add_argument('--prefix', action='store', dest='prefix',help='exp log prefix to append exp/{} default = 0',default = '0')

	return parser
