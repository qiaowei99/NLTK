# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:43:02 2018

@author: admin
"""
import nltk
##Q9
#a
def load(file):
    f=open(file)
    ff=f.read()
    return ff

text=load('corpus.txt')
#2018-09-22:this is a real surprise?that ''Andy'',John and Alice won 9.99$; 99$ and 999$ in 06.11.2018 King Pro.
pattern1 = r'''(?x)                     #set flag to allow verbose regexps
[\.,;"'?\(\):\-_`\[\]\{\}]              #tokens
'''
text1=nltk.regexp_tokenize(text,pattern1)
print('punctuations:',text1)

#b
#monetary amounts
mon_pattern =r'''(?x)                   #set flag to allow verbose regexps
(?:\d+\.)?\d+\s?\$                      #monetary amounts like 9.99$
'''
mon_text=nltk.regexp_tokenize(text, mon_pattern)
print('monetary amounts:',mon_text)

#dates
date_pattern =r'''(?x)                  #set flag to allow verbose regexps
 \d{4}\-\d{2}\-\d{2}                    #date like 2018-09-22
|  \d{2}.\d{2}.\d{4}                    #date like 06.11.2018
'''
date_text=nltk.regexp_tokenize(text, date_pattern)
print('dates:',date_text)

#names of people and organizations
mame_pattern =r'''(?x)                  #set flag to allow verbose regexps
 [A-Z][a-z]+(?:\s[A-Z][a-z]+)?          # names of people and organizations
'''
name_text=nltk.regexp_tokenize(text, mame_pattern)
print('names:',name_text)





