#!/usr/bin/env sh

ASS_DATA=http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2
F1=answer-sentence-selection
cd data ; wget $ASS_DATA ; tar -xjf *.tar.*; mv jacana-qa-naacl2013-data-results $F1 ; rm *.bz2 cd -

FQA_DATA=http://cs.umd.edu/~miyyer/data/question_data.tar.gz
F2=factoid-question-answering
cd data/ ; wget $FQA_DATA ; tar xvfz question_data.tar.gz; mv question_data $F2; rm *.tar.gz ; cd -

TRECEVAL=http://trec.nist.gov/trec_eval/trec_eval.8.1.tar.gz
cd bin/ ; wget $TRECEVAL ; tar xvfz *.tar.* ; rm *.tar.gz ; cd trec_eval.8.1 ; make ;
