#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Methods to load wikipedia into strings and build a vocabulary of syllables from it.

"""

__author__      = ["Karen Ullrich","Jonathan Raiman"]
__email__       = "karen.ullrich@ofai.at"

#-------------------------------------------------------
# Libraries
#-------------------------------------------------------

import numpy as np
import hyphen # syllable extraction , pip install pyhyphen
from re import findall
from os import listdir
from os.path import join
import random as r
import codecs
import cPickle as pickle

#--------------------------------------
# Methods
#--------------------------------------

## Stream Wiki article-wise (type: unicode)

class Wiki_articles:

    def __init__(self, dir = None, seed =None):

        if dir is None: #Hack for my homedir
            dir = '/home/karen/projects/LSTM_RNN_PoetryGenerator/data/text/'
        
        files =[]
        subdirs = [ s for s in listdir(dir)]

        for subdir in subdirs:
            tmp = [join(dir,subdir,f) for f in listdir(join(dir,subdir))]
            files.extend(tmp)
        self.files = files

    def __call__(self):
        r.shuffle(self.files)
        for file in self.files:
            f = codecs.open(file, "r", "utf-8")
            for line in f:
                if '<doc ' in line:
                    article = []
                elif '</doc>' in line:
                    yield ''.join(article)
                else:
                    article.append(line)

## Builds a vocabulary for the RNN from unicode(!)
class Vocab:
    __slots__ = ["syllable2index", "index2syllable", "unknown"]
    
    def __init__(self, syllable2index = None):

        self.syllable_extractor = hyphen.Hyphenator('de_DE')
        self.syllable2index = {}
        self.index2syllable = []
        
        # load old dictionary if path given
        if syllable2index is not None:
            self.syllable2index = pickle.load( open(syllable2index, "rb"))
            self.index2syllable = ["" for x in range(len(self.syllable2index))]
            for syllable in self.syllable2index.keys():
                self.index2syllable[self.syllable2index[syllable][0]] = syllable

        # add unknown syllable:
        self.add_syllables(u'UNKNOWN')# for now its in German. unknown is a word that wont occur → perfect but if we switch to English we need to change that
        self.unknown = [0,0]
                
    def add_syllables(self, text):
        for word in findall(r"(?u)\w+|[ ,.;!?'%#-]", text):
            if len(word) < 100:
                syllables = self.syllable_extractor.syllables(word)
                if syllables == []: # for one-syllable-words and puctuation
                    syllables = [word]
                for syllable in syllables:
                    if syllable not in self.syllable2index:
                        self.syllable2index[syllable] = [len(self.syllable2index),1] # [idx ,count]
                        self.index2syllable.append(syllable)
                    else: 
                        self.syllable2index[syllable][1] += 1
            else:
                print word, ' could not be included into library of vocabulary'

    def __call__(self, line):
        """
        Convert from numerical representation to words
        and vice-versa.
        """
        if type(line) is np.ndarray:
            return "".join([self.index2syllable[syllable] for syllable in line])
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    return "".join([self.index2syllable[syllable] for syllable in line])
            indices = np.zeros(len(line), dtype=np.int32)
        else:
            line = findall(r"[\w']+|[.%'#,!?; ]", line)
            indices = np.zeros(len(line), dtype=np.int32)
        
        for i, syllable in enumerate(line):
            indices[i] = self.syllable2index.get(syllable, self.unknown)[0]
            
        return indices
    
    @property
    def size(self):
        return len(self.index2syllable)
    
    def __len__(self):
        return len(self.index2syllable)

    def restrict_vocabs(self,max_syllables):
        """
        Restricts the vocabulary to 'max_syllables' available syllables.
        """
        # find threshold 
        num_syllables = len(self.syllable2index)
        if num_syllables <= max_syllables: return

        # sort syllables by occurence
        # probably not hte most clever way to do that
        threshold = np.argsort(np.transpose(self.syllable2index.values())[1])[max_syllables]
        # keep the 'max_syllables' most occuerent syllables
        for syllable in self.syllable2index:
            if syllable != u'UNKNOWN':
                if self.syllable2index[syllable][1] < threshold:
                    self.index2syllable.pop(syllable2index[syllable][0])
                    self.syllable2index,pop(syllable)

        # I am aware that the current implementation may yield more than max_syllables 
        # but that seems reasonable to me.

    def show_occurence(self,ylim=10):
        ''' Shows an n-occurence plot

        '''
        import pylab as plt
        plt.plot(np.sort(np.transpose(self.syllable2index.values()))[1][::-1])
        plt.ylim([0,ylim])
        plt.show()

    def save(self, outdir=None):
        '''Saves vocabulary dictionary to file

        '''
        pickle.dump(self.syllable2index, 
            open( join(outdir,'vocabulary'), "wb" ))

def pad_into_matrix(rows, padding = 0):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    lengths = [i for i in map(len, rows)]
    width = max(lengths)
    height = len(rows)
    mat = np.empty([height, width], dtype=rows[0].dtype)
    mat.fill(padding)
    for i, row in enumerate(rows):
        mat[i, 0:len(row)] = row
    return mat, list(lengths)


if __name__=="__main__":
    '''
    Go through the wiki corpus and create (and save) a 
    library of all occurring syllables
    '''
    counter = 0
    articles = Wiki_articles()
    vocabulary = Vocab()
    for article in articles():
        counter +=1
        print 'article ', counter
        vocabulary.add_syllables(article)

    vocabulary.save(outdir='/home/karen/projects/LSTM_RNN_PoetryGenerator/data/')
