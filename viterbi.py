#!/usr/bin/python
# Python implementation of viterbi algorithm

# Usage:  viterbi.py hmm-file < text > tags

# Nadia Hyder
# 5/8/19

import sys
import itertools
import math


TAG_FILE=sys.argv[1]
testFile = 'ptb.22.txt'

INIT_STATE = "init"
FINAL_STATE = "final"
OOV_SYMBOL = "OOV"

transitionProb = {}
emissionProb = {}
vocabulary = {}
tags = set()

# read in the HMM and store the probabilities as log probabilities
with open(TAG_FILE) as hmm:
    for line in hmm:
        words = line.split()
        if words[0] == 'trans':
            transitionProb[tuple([words[1], words[2]])] = math.log(float(words[3])) # tuple[prevTag, curTag]
            tags.update([words[1], words[2]])
        elif words[0] == 'emit':
            emissionProb[tuple([words[1], words[2]])] = math.log(float(words[3]))  # tuple[tag, token]
            vocabulary[words[2]] = 1
            tags.update(words[1])
        else:
            pass

# read test file and run Viterbi alg
with open(testFile) as testFile:
    for line in testFile.read().splitlines():
        tokenArr = line.split(' ')
        n = len(tokenArr)

        bckptr = {}
        pi = {(0, INIT_STATE): 0.0} # base case of the recursive equations

        for i, token in enumerate(tokenArr):
            i = i + 1

            # if a word isn't in the vocabulary, rename with OOV symbol
            if token not in vocabulary:
                token = OOV_SYMBOL


            for currTag, prevTag in itertools.product(tags, tags):
                if (prevTag, currTag) in transitionProb and (currTag, token) in emissionProb and (i - 1, prevTag) in pi:
                    score = pi[(i - 1, prevTag)] + emissionProb[(currTag, token)] + transitionProb[(prevTag, currTag)]
                    if (i, currTag) not in pi or score > pi[(i, currTag)]:
                        pi[(i, currTag)] = score
                        bckptr[(i, currTag)] = prevTag

        foundgoal = 0
        tag = INIT_STATE
        goal = float('-Inf')


        # find max scoring last tag
        for prevTag in tags:
            if (prevTag, FINAL_STATE) in transitionProb and (n, prevTag) in pi:
                score = pi[(n, prevTag)] + transitionProb[(prevTag, FINAL_STATE)]  # no emission prob here
                if not foundgoal or score > goal:
                    goal = score # finding max probability final tag
                    foundgoal = 1
                    tag = prevTag

        # backtrace
        if foundgoal:
            finalTags = []
            for j in xrange(n, 1, -1):
                finalTags.append(bckptr[(j, tag)])
                tag = bckptr[(j, tag)]

            finalTags.reverse()
            print ' '.join(finalTags)
        else:
            print ' '.join([])  # If the HMM fails to recognize a sequence, a blank line will be written
