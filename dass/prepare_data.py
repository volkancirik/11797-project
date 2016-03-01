#!/usr/bin/env/python
import sys, gzip, theano
import cPickle as pickle
import numpy as np

UNK='*UNKNOWN*'

def open_file(fname):
	try:
		f = open(fname)
	except:
		print >> sys.stderr, "file %s could not be opened" % (fname)
		quit(0)
	return f

def embed_sentence(s,v_map,unk, mini_batch = False, max_len = 0):

	if not(mini_batch):
		max_len = len(s)

	dim = len(list(v_map[UNK]))
	x = np.zeros((1,max_len, dim), dtype=theano.config.floatX)

	for j,tok in enumerate(s):
		try:
			v = v_map[tok]
		except:
			unk +=1
			v = v_map[UNK]
			pass
		x[0,j,:] = list(v)
	return x,unk

def vectorize(data,v_map, mini_batch = True):

	N = len(data)

	dim = len(list(v_map[UNK]))
	max_q = 0
	max_a = 0
	for i in xrange(N):
		q,a,label,featq,feata = data[i]
		if len(q) > max_q:
			max_q = len(q)
		if len(a) > max_a:
			max_a = len(a)

	if mini_batch:
		X_q = np.zeros((N,max_q,dim), dtype=theano.config.floatX)
		X_a = np.zeros((N,max_a,dim), dtype=theano.config.floatX)
	else:
		X = []

	Y = np.zeros((N,1), dtype = bool)

	unk_q = 0
	unk_a = 0

	for i,(q,a,label,featq,feata) in enumerate(data):
		if mini_batch:
			x_q, unk_q = embed_sentence(q, v_map, unk_q, mini_batch = mini_batch, max_len = max_q)
			x_a, unk_a = embed_sentence(a, v_map, unk_a, mini_batch = mini_batch, max_len = max_a)
			X_q[i,:,:] = x_q
			X_a[i,:,:] = x_a
		else:
			x_q, unk_q = embed_sentence(q, v_map, unk_q)
			x_a, unk_a = embed_sentence(a, v_map, unk_a)
			X += [(x_q,x_a)]
		Y[i] = label
	if mini_batch:
		X = [X_q,X_a]
	return X,Y

def process_target(fname, word_idx, tokenize = False, max_len = -1):

	in_file = open_file(fname)

	V = len(word_idx)
	unk = 0.0
	ntok = 0.0

	S = [line.lower().strip().split() for line in in_file]
	N = len(S)

	Y = np.zeros((N,max_len,V), dtype=np.bool)
	for i,s in enumerate(S):
		ntok += len(s)
		for j,tok in enumerate(s):
			try:
				idx = word_idx[tok]
			except:
				unk += 1
				idx = word_idx[UNK]
				pass
			Y[i,j,idx] = 1
	print >> sys.stderr, "UNK rate for %s is %f" % (fname, unk/ntok)
	return Y


def read_data(fname):

	f = open_file(fname)
	data = []

	fq = True
	lc = 0
	label = 0
	uniq = 0
	featq = []
	feata = []
	q = []
	a = []

	for line in f:
		l = line.strip().split()
		if (l[0][0:2] == '</' or l[0] == '<QApairs') or (lc == 5 and l[0][0] != '<'):
			continue
		if l[0] == '<question>':
			fq = True
			lc = 0
			continue
		if l[0] == '<positive>':
			label = 1
			lc = 0
			continue
		if l[0] == '<negative>':
			label = 0
			lc = 0
			continue
		if lc == 0:
			if fq:
				q = l
				uniq +=1
			else:
				a = l
		else:
			if fq:
				featq += l
			else:
				feata += l
		lc += 1

		if lc == 5:
			if fq:
				fq = False
			else:
				data += [(q,a,label,featq,feata)]
				feata = []
	print >> sys.stderr, 'there are %d unique questions in %s' % (uniq,fname)
	return data

def prepare_ass(prefix = '../data/answer-sentence-selection/', train = 'train2393.cleanup.xml', validation = 'dev-less-than-40.manual-edit.xml', test = 'test-less-than-40.manual-edit.xml', wvec = '../embeddings/word2vec.pkl', mini_batch = False):

	try:
		print >> sys.stderr, "loading word vectors..."
		v_map = pickle.load(gzip.open(wvec, "rb"))
		dim = len(list(v_map[UNK]))
		print >> sys.stderr, "%d dim word vectors are loaded.." % (dim)
	except:
		print >> sys.stderr, "word embedding file %s cannot be read." % (wvec)
		quit(1)

	tr_d = read_data(prefix + train)
	val_d = read_data(prefix + validation)
	te_d = read_data(prefix + test)

	X_tr,Y_tr = vectorize(tr_d, v_map, mini_batch = mini_batch)
	print >> sys.stderr, "vectorized training..."
	X_val,Y_val = vectorize(val_d, v_map, mini_batch = mini_batch)
	print >> sys.stderr, "vectorized validation..."
	X_test,Y_test = vectorize(te_d, v_map, mini_batch = mini_batch)
	print >> sys.stderr, "vectorized test..."

	return X_tr, Y_tr, X_val, Y_val, X_test,Y_test

if __name__ == '__main__':
	prepare_ass()
#	X_tr, Y_tr, X_val, Y_val, X_test,Y_test = prepare_ass()

