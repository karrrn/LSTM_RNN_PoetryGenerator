#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Methods to load wikipedia into strings and build a vocabulary of syllables from it.

"""

__author__      = "Karen Ullrich"
__email__       = "karen.ullrich@ofai.at"

#-------------------------------------------------------
# Libraries
#-------------------------------------------------------

import hyphen # syllable extraction , pip install pyhyphen
from re import findall


#--------------------------------------
# Methods
#--------------------------------------

## Load Wiki into str. or several strs

#TODO

## Converts lsits of words into a vocabulary dictonary (one vocabulary is a syllable)
class Vocab:
    __slots__ = ["word2index", "index2word", "unknown"]
    
    def __init__(self, index2syllable = None):
        self.punctuation = ['.',',',';','?',' '] # maybe better to ignore that for poetry
        self.syllable_extractor = hyphen.Hyphenator('de_DE')
        self.syllable2index = {}
        self.index2syllable = []
        
        # add unknown syllable:
        self.add_syllables("**UNKNOWN**")# for now its in german so unknown is a word that wont occur → perfect but if we switch to english we need to change that
        self.unknown = 0
        
        if index2syllable is not None:
            self.add_syllables(index2syllable)
                
    def add_syllables(self, text):
        for word in findall(r"[\w']+|[.%'#,!?; ]", text):
            syllables = self.syllable_extractor.syllables(unicode(word))
            if syllables == []: # for one-syllable-words and puctuation
                syllables = [word]
            for syllable in syllables:
                if syllable not in self.syllable2index:
                    self.syllable2index[syllable] = [len(self.syllable2index),1] # [idx ,count]
                    self.index2syllable.append(syllable)
                else: 
                    self.syllable2index[syllable][1] += 1

                       
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
            indices[i] = self.syllable2index.get(syllable, self.unknown)
            
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
            if self.syllable2index[syllable][1] < threshold:
                self.index2syllable.pop(syllable2index[syllable][0])
                self.syllable2index,pop(syllable)

        if 'UNKNOWN' not in self.syllable2index:
            self.add_syllables("**UNKNOWN**")

        # I am aware that the current implementation may yield more than max_syllables but that seems resonable to me.


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