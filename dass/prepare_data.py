#!/usr/bin/env/python
import sys, gzip, theano
import cPickle as pickle
import numpy as np
from collections import defaultdict
UNK='*UNKNOWN*'
THRESHOLD = 2
EOQ = '</q>'
EOA = '</a>'
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
		pad = max_len - len(s)
		tok = tok.lower()
		try:
			v = v_map[tok]
		except:
			unk +=1
			v = v_map[UNK]
			pass
		x[0, pad+j ,:] = list(v) ## left padding
#		x[0, j ,:] = list(v)
	return x,unk

def vectorize(data,v_map, mini_batch = True, finetune = False, word_idx = {}):

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
	max_q += 1 ## EO symbol
	max_a += 1
	print >> sys.stderr, "max sequence length for question/answer %d/%d" % (max_q,max_a)

	if mini_batch:
		if finetune:
			X_q = np.zeros((N,max_q), dtype = 'int64')
			X_a = np.zeros((N,max_a), dtype = 'int64')
		else:
			X_q = np.zeros((N,max_q,dim), dtype=theano.config.floatX)
			X_a = np.zeros((N,max_a,dim), dtype=theano.config.floatX)
	else:
		X = []

	Y = np.zeros((N,1), dtype = bool)

	unk_q = 0.0
	unk_a = 0.0
	n_q = 0
	n_a = 0
	length_q = []
	length_a = []
	for i,(q,a,label,featq,feata) in enumerate(data):
		q += [EOQ]
		a += [EOA]
		n_q += len(q)
		n_a += len(a)
		length_q += [len(q)]
		length_a += [len(a)]
		if mini_batch:
			if finetune:
				pad = max_q - len(q) #
				for j,tok in enumerate(q):
					try:
						X_q[i,pad + j] = word_idx[tok]
					except:
						X_q[i,pad + j] = word_idx[UNK]
						unk_q += 1

				pad = max_q - len(a) #
				for j,tok in enumerate(a):
					try:
						X_a[i,pad + j] = word_idx[tok]
					except:
						X_a[i,pad + j] = word_idx[UNK]
						unk_a += 1
			else:
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
	print >> sys.stderr, "UNK for q %f and a %f" % ( unk_q/n_q, unk_a/n_a)
	return X,Y,[length_q,length_a]

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

def read_liveqa(prefix = '../data/qalab-liveqa/dataset/qrels/', train = 'LiveQA2015-ver2.qrels', tokenize = True):
	import nltk

	f = open_file(prefix + train)
	np.random.seed(0)

	data_split = {0: [], 1 : [], 2 : []}
	ref_split = {0: [], 1 : [], 2 : []}

	for i,line in enumerate(f):
		l = line.strip().split('\t')
		if l[2] == '':
			first = " ? ".join(l[3].strip().split("?"))
			second = " . ".join(first.strip().split("."))
			q = " ".join(nltk.word_tokenize(second.strip())).lower().split(' ')
			split_id = np.random.choice([0,0,0,1,2])
			continue
		label = int(l[2]) >= 3

		first = " ? ".join(l[3].strip().split("?"))
		second = " . ".join(first.strip().split("."))
		a = " ".join(nltk.word_tokenize(second.strip())).lower().split(' ')
		data_split[split_id] += [(q,a,label,'','')]
		ref_split[split_id] += [(l[0],'0',l[0]+'_'+l[1]+'_'+str(i),str(int(label)))]

	return data_split[0],data_split[1],data_split[2],(ref_split[0],ref_split[1],ref_split[2])

def print_ref(ref,fp,suffix):
	f = open(fp+suffix,'w')
	for line in ref:
		print >> f, "\t".join(list(line))
	f.close()

def get_embeddings(word_idx, idx_word, wvec = '../embeddings/word2vec.pkl', UNK_vmap = '*UNKNOWN*', expand_vocab = False):
	import gzip, sys
	import cPickle as pickle
	import numpy as np

	try:
		print >> sys.stderr, "loading word vectors..."
		v_map = pickle.load(gzip.open(wvec, "rb"))
		dim = len(list(v_map[UNK_vmap]))
		print >> sys.stderr, "%d dim word vectors are loaded.." % (dim)
	except:
		print >> sys.stderr, "word embedding file %s cannot be read." % (wvec)
		quit(1)

	V_text = len(word_idx)

	if expand_vocab:
		for w in v_map.keys():
			if w not in word_idx:
				V_text += 1
				word_idx[w] = V_text
				idx_word[V_text] = w

	embedding_weights = np.zeros((V_text+1,dim))

	for w in word_idx:
		idx = word_idx[w]
		if w not in v_map:
			w = UNK_vmap
		try:
			embedding_weights[idx,:] = v_map[w]
		except:
			print >> sys.stderr, "something is wrong with following tuple:", idx, idx_word[idx], w
			quit(1)
	return v_map, embedding_weights, word_idx, idx_word

def get_dicts(d_list):

	vocab = defaultdict(int)
	for d in d_list:
		for i in xrange(len(d)):
			q,a,label,featq,feata = d[i]
			for tok in q:
				vocab[tok] +=1
			for tok in a:
				vocab[tok] +=1

	vocab[UNK] = THRESHOLD
	vocab[EOA] = THRESHOLD
	vocab[EOQ] = THRESHOLD

	words = [w for w in vocab]
	for w in words:
		if vocab[w] < THRESHOLD:
			del vocab[w]

	word_idx = dict((c, i) for i, c in enumerate(vocab))
	idx_word = dict((i, c) for i, c in enumerate(vocab))

	V = len(word_idx)
	first_w = idx_word[0]
	idx_word[0] = '*dummy*'
	idx_word[V] = first_w
	word_idx[first_w] = V
	word_idx['*dummy*'] = 0

	return word_idx, idx_word

def prepare_ass(prefix = '../data/answer-sentence-selection/', train = 'train2393.cleanup.xml', validation = 'dev-less-than-40.manual-edit.xml', test = 'test-less-than-40.manual-edit.xml', wvec = '../embeddings/word2vec.pkl', mini_batch = True, mode = '', dataset = 'yahool6', fp = '', finetune = False):

	print >> sys.stderr, "%s is being prepared" % (dataset)
	if mode == 'DEBUG':
		train = 'test-less-than-40.manual-edit.xml'
		wvec = '../embeddings/small.pkl'
	if dataset == 'liveqa':
		tr_d, val_d, te_d, (ref_tr,ref_val,ref_test) = read_liveqa()
		print >> sys.stderr, "writing reference files..."
		print_ref(ref_tr,fp,'.train.TREC.ref')
		print_ref(ref_val,fp,'.val.TREC.ref')
		print_ref(ref_test,fp,'.test.TREC.ref')
	elif dataset == 'yahool6':
		tr_d, val_d, te_d, (ref_tr,ref_val,ref_test) = read_liveqa(prefix = '/usr0/home/vcirik/projects/11797-project/data/yahool6/', train = 'FullOct2007.qrel')
		print >> sys.stderr, "writing reference files..."
		print_ref(ref_tr,fp,'.train.TREC.ref')
		print_ref(ref_val,fp,'.val.TREC.ref')
		print_ref(ref_test,fp,'.test.TREC.ref')
	else:
		tr_d = read_data(prefix + train)
		val_d = read_data(prefix + validation)
		te_d = read_data(prefix + test)

	word_idx, idx_word = get_dicts([tr_d,val_d])
	v_map, embedding_weights, word_idx, idx_word = get_embeddings(word_idx, idx_word, wvec = wvec, UNK_vmap = '*UNKNOWN*')

	X_tr, Y_tr, [tr_length_q,tr_length_a] = vectorize(tr_d, v_map, mini_batch = mini_batch, finetune = finetune, word_idx = word_idx)
	print >> sys.stderr, "vectorized training..."
	X_val,Y_val,[val_length_q,val_length_a] = vectorize(val_d, v_map, mini_batch = mini_batch, finetune = finetune, word_idx = word_idx)
	print >> sys.stderr, "vectorized validation..."
	X_test,Y_test,[test_length_q,test_length_a] = vectorize(te_d, v_map, mini_batch = mini_batch, finetune = finetune, word_idx = word_idx)
	print >> sys.stderr, "vectorized test..."

	return X_tr, Y_tr, X_val, Y_val, X_test,Y_test,[tr_length_a,val_length_a,test_length_a], [word_idx,idx_word], embedding_weights,

if __name__ == '__main__':
	read_liveqa()
#	X_tr, Y_tr, X_val, Y_val, X_test,Y_test = prepare_ass()

	# try:
	# 	print >> sys.stderr, "loading word vectors..."
	# 	v_map = pickle.load(gzip.open(wvec, "rb"))
	# 	dim = len(list(v_map[UNK]))
	# 	print >> sys.stderr, "%d dim word vectors are loaded.." % (dim)
	# except:
	# 	print >> sys.stderr, "word embedding file %s cannot be read." % (wvec)
	# 	quit(1)
