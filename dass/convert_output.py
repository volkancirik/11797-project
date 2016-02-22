#!/usr/bin/env/python
import sys

def open_file(fname):
	try:
		f = open(fname)
	except:
		print >> sys.stderr, "file %s could not be opened" % (fname)
		quit(0)
	return f

def convert_data(outfname, prefix = '../data/answer-sentence-selection/', refname = 'test-less-than-40.manual-edit.xml'):

	f = open_file(outfname)
	OUT = [line.strip().split('\t') for line in f]

	reference = open(outfname + '.TREC.ref','w')
	prediction = open(outfname + '.TREC.pred','w')

	f = open_file(prefix+refname)
	data = []

	fq = True
	lc = 0
	label = 0
	uniq = 0
	featq = []
	feata = []
	q = []
	a = []
	idx = 0

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
				assert(label == int(float(OUT[idx][-1])))
				print >> prediction, "\t".join([str(uniq),'0',str(uniq)+'_'+str(idx),'0',OUT[idx][0],'-'])
				print >> reference, uniq, 0, str(uniq)+'_'+str(idx), label

				idx += 1
				feata = []
	print >> sys.stderr, 'there are %d unique questions in %s and idx %d' % (uniq,refname,idx)
	return data
if __name__ == '__main__':
	convert_data(sys.argv[1])
