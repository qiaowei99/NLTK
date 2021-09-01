# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:49:47 2018

@author: admin
"""

##tips:
#use both LSI (sometimes called LSA) and LDA
#lsi_model = models.LsiModel
#lda_model = models.LdaModel

import nltk
from  nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

import gensim
import gensim.corpora as corpora


doc1="From fairest creatures we desire increase,That thereby beauty's rose might never die,"
doc2="But as the riper should by time decease,His tender heir might bear his memory,"
doc3="But thou contracted to thine own bright eyes,Feed'st thy light's flame with self-substantial fuel,Making a famine where abundance lies,Thy self thy foe, to thy sweet self too cruel"
doc4="Thou that art now the world's fresh ornament,And only herald to the gaudy spring,Within thine own bud buriest thy content,And tender churl mak'st waste in niggarding"
doc5="Pity the world, or else this glutton be,To eat the world's due, by the grave and thee."
doc_complete = [doc1, doc2, doc3, doc4, doc5]

##clean data
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

##creat dictionary and matrix
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]



##print original text
print('the original text:',doc_complete,'\n')


##creat LDA model and get the result(compare the result when number of keywords are 2/3)
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=50)
print('the result of LDA MODEL(num_words=2):',ldamodel.print_topics(num_topics=2, num_words=2),'\n')

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=50)
print('the result of LDA MODEL(num_words=3):',ldamodel.print_topics(num_topics=2, num_words=3),'\n')

##creat LSI model and get the result
Lsi = gensim.models.LsiModel
lsimodel = Lsi(doc_term_matrix, num_topics=2, id2word = dictionary)
print('the result of LSI MODEL(num_words=2):',lsimodel.print_topics(num_topics=2, num_words=2),'\n')

Lsi = gensim.models.LsiModel
lsimodel = Lsi(doc_term_matrix, num_topics=2, id2word = dictionary)
print('the result of LSI MODEL(num_words=3):',lsimodel.print_topics(num_topics=2, num_words=3),'\n')


'''
Using one of Shakespeare's sonnets as the original text, the results seems intersting.

When keywords number is 2, the result of LDA/LSI is approaching. The first topic is "world+tender", while the second 
one is "thy+ self". But the weight is different.

When keywords number is 3, the third keyword for two topics in different models vary. 

And personally, I think the LDA model is more accurate than LSI.



'''