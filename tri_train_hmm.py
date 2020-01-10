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

# This is a modified HMM for handling trigrams.

import sys, re
from itertools import izip
from collections import defaultdict

TAG_FILE = sys.argv[1]
TOKEN_FILE = sys.argv[2]

vocab = {}
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

transitions = {}
transitionsTotal = defaultdict(int)
emissions = {}
emissionsTotal = defaultdict(int)
triTransitions = {}
triTransitionsTotal = defaultdict(int)

with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
    # for tagString, tokenString in izip(tagFile.readlines(60000), tokenFile.readlines(60000)):
    for tagString, tokenString in izip(tagFile, tokenFile):

        tags = re.split("\s+", tagString.rstrip())
        tokens = re.split("\s+", tokenString.rstrip())
        pairs = zip(tags, tokens)

        # changed this
        prevtag = INIT_STATE
        prevprevtag = INIT_STATE #both uninitialized

        for (tag, token) in pairs:

            # deal with OOV words

            if token not in vocab:
                vocab[token] = 1
                token = OOV_WORD

            if tag not in emissions:
                emissions[tag] = defaultdict(int)
            if prevtag not in transitions:
                transitions[prevtag] = defaultdict(int)
            if (prevprevtag, prevtag) not in triTransitions:
                triTransitions[(prevprevtag, prevtag)] = defaultdict(int)

            # increment the emission/transition observation
            emissions[tag][token] += 1
            emissionsTotal[tag] += 1

            transitions[prevtag][tag] += 1
            transitionsTotal[prevtag] += 1

            # trigram model
            triTransitions[(prevprevtag,prevtag)][tag] += 1
            triTransitionsTotal[(prevprevtag,prevtag)] += 1

            prevprevtag = prevtag
            prevtag = tag

        # don't forget the stop probability for each sentence
        if prevtag not in transitions:
            transitions[prevtag] = defaultdict(int)

        if (prevprevtag, prevtag) not in triTransitions:
            triTransitions[(prevprevtag,prevtag)] = defaultdict(int)

        transitions[prevtag][FINAL_STATE] += 1
        transitionsTotal[prevtag] += 1

        triTransitions[(prevprevtag, prevtag)][FINAL_STATE] += 1
        triTransitionsTotal[(prevprevtag, prevtag)] += 1

for prevtag in transitions:
    for tag in transitions[prevtag]:
        print "trans %s %s %s" % (prevtag, tag, float(transitions[prevtag][tag]) / transitionsTotal[prevtag])

for (prevprevtag, prevtag) in triTransitions:
    for tag in triTransitions[(prevprevtag, prevtag)]:
        print "tritrans %s %s %s %s" % (prevprevtag, prevtag, tag, float(triTransitions[(prevprevtag, prevtag)][tag]) / triTransitionsTotal[(prevprevtag,prevtag)])

for tag in emissions:
    for token in emissions[tag]:
        print "emit %s %s %s " % (tag, token, float(emissions[tag][token]) / emissionsTotal[tag])
