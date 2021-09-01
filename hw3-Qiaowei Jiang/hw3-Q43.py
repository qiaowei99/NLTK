# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 00:19:13 2018

@author: admin
"""
import re
from nltk import *
from nltk.corpus import udhr
from nltk.corpus import genesis
from nltk import spearman_correlation
from nltk.corpus import gutenberg

def guess(text):
    #choose the test language 
    #fids = list(udhr.fileids())
    fids = ['French_Francais-Latin1', 'Spanish-Latin1', 'German_Deutsch-Latin1', 'English-Latin1']
    

    #makes a list of all the available languages that use Latin1 encoding
    languages = [fileid for fileid in fids if re.findall('Latin1', fileid)]
    #get rid of the puncutation and list letters of each language
    udhr_corpus = [[list(word.lower()) for word in udhr.words(language) if word.isalpha()] for language in languages]
    udhr_corpus = [[item for sublist in language for item in sublist] for language in udhr_corpus]
  
    #add index to each language
    languages = list(enumerate(languages))
    #determine frequency distributions for all the characters in a language and get the rank list based on freq
    language_freq_dists = [FreqDist(language) for language in udhr_corpus]
    language_ranks = [list(ranks_from_sequence(dist)) for dist in language_freq_dists]
      
    #get rid of the puncutation in the testing text and determine frequency distributions for all the characters 
    text_words = [list(word.lower()) for word in text if word.isalpha()]
    text_words = [item for sublist in text_words for item in sublist]
    fd_text_words = FreqDist(text_words)
    #determine ranking list of characters based on the frequency distribution
    text_ranks = list(ranks_from_sequence(fd_text_words))
    
    #The Spearman correlation coefficient gives a number from -1.0 to 1.0 comparing two rankings
    #A coefficient of 1.0 indicates identical rankings
    #-1.0 indicates exact opposite rankings
    #compare the unknown distribution to the known ones to identify the language
    spearman_numbers = []
    for language in language_ranks:
       number = spearman_correlation(language, text_ranks)
       spearman_numbers.append(number)
    
    zipped = list(zip(languages, spearman_numbers))
    #rank the language, thus the language with highest Spearman correlation coefficient ranks first
    zipped_rank=sorted(zipped,key=lambda x:x[1],reverse=True)

    return(print(zipped_rank[0]))
    


#using text=gutenberg.words('austen-persuasion.txt')  to test the function
text = gutenberg.words('austen-persuasion.txt')  
guess(text)
