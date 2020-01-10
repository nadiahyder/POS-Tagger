#!/usr/bin/python

# David Bamman
# 2/14/14
#
# Python port of train_hmm.pl:

# Noah A. Smith
# 2/21/08
# Code for maximum likelihood estimation of a bigram HMM from
# column-formatted training data.

# Usage:  train_hmm.py tags text > hmm-file

# The training data should consist of one line per sequence, with
# states or symbols separated by whitespace and no trailing whitespace.
# The initial and final states should not be mentioned; they are
# implied.
# The output format is the HMM file format as described in viterbi.pl.

# This is the original train_hmm given to us.

import sys, re
from itertools import izip
from collections import defaultdict

TAG_FILE = sys.argv[1]
TOKEN_FILE = sys.argv[2]

vocab = {}
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

emissions = {}
transitions = {}
transitionsTotal = defaultdict(int)
emissionsTotal = defaultdict(int)

# change the size of training data?
# file.readlines([sizehint])
with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
    for tagString, tokenString in izip(tagFile.readlines(100000), tokenFile.readlines(1000000)):

    #for tagString, tokenString in izip(tagFile, tokenFile):

        tags = re.split("\s+", tagString.rstrip())
        tokens = re.split("\s+", tokenString.rstrip())
        pairs = zip(tags, tokens)  # [slice(5000)]

        prevtag = INIT_STATE

        for (tag, token) in pairs:

            # this block is a little trick to help with out-of-vocabulary (OOV)
            # words.  the first time we see *any* word token, we pretend it
            # is an OOV.  this lets our model decide the rate at which new
            # words of each POS-type should be expected (e.g., high for nouns,
            # low for determiners).

            if token not in vocab:
                vocab[token] = 1
                token = OOV_WORD

            if tag not in emissions:
                emissions[tag] = defaultdict(int)
            if prevtag not in transitions:
                transitions[prevtag] = defaultdict(int)

            # increment the emission/transition observation
            emissions[tag][token] += 1
            emissionsTotal[tag] += 1

            transitions[prevtag][tag] += 1
            transitionsTotal[prevtag] += 1

            prevtag = tag

        # don't forget the stop probability for each sentence
        if prevtag not in transitions:
            transitions[prevtag] = defaultdict(int)

        transitions[prevtag][FINAL_STATE] += 1
        transitionsTotal[prevtag] += 1

for prevtag in transitions:
    for tag in transitions[prevtag]:
        print "trans %s %s %s" % (prevtag, tag, float(transitions[prevtag][tag]) / transitionsTotal[prevtag])

for tag in emissions:
    for token in emissions[tag]:
        print "emit %s %s %s " % (tag, token, float(emissions[tag][token]) / emissionsTotal[tag])