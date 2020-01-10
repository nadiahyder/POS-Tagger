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

import sys
import itertools
import math
from collections import defaultdict, deque

TAG_FILE = sys.argv[1]
testFile = 'btb.test.txt'
INIT_STATE = "init"
FINAL_STATE = "final"
OOV_SYMBOL = "OOV"

transitionProb = defaultdict(lambda: 0)
triTransitionProb = defaultdict(lambda: 0)
emissionProb = defaultdict(lambda: 0)
tags = set()
vocabulary = defaultdict(lambda: 0)
unigramCounts = defaultdict(lambda: 0)

# read in the HMM and store the probabilities as log probabilities
with open(TAG_FILE) as hmm:
    for line in hmm:
        split = line.split()
        if split[0] == 'trans':
            transitionProb[(split[1], split[2])] = math.log(float(split[3])) # tuple[prevTag, curTag]
            tags.update([split[1], split[2]])

            unigramCounts[split[1]]+=1 # update unigram
            unigramCounts[split[2]]+=1

        elif split[0] == 'emit':
            emissionProb[(split[1], split[2])] = math.log(float(split[3]))  # tuple[tag, token]
            vocabulary[split[2]] = 1
            tags.update(split[1])  # add the tag to tags list

        elif split[0] == 'tritrans':
            triTransitionProb[(split[1],split[2],split[3])] = math.log(float(split[4])) # tuple[prevprev, prev, tag]
            unigramCounts[split[1]]+=1
            unigramCounts[split[2]]+=1
            unigramCounts[split[3]]+=1

        else:
            pass


# use a backoff model
def getProbability(prev2, prev1, current):
    score = 0

    uniCount = unigramCounts[current]

    if (prev2, prev1, current) in triTransitionProb:
        score += triTransitionProb[(prev2, prev1, current)]
    else:
        if (prev1, current) in transitionProb:
            score += transitionProb[(prev1, current)]
        else:
            score += math.log(uniCount + 1)
            score -= math.log(sum(unigramCounts.values()) + len(unigramCounts))

    return score


# read test file and run Viterbi alg
with open(testFile) as testFile:
    tagged = []
    for line in testFile.read().splitlines():
        tokenArr = line.split(' ')
        n = len(tokenArr)  # sentence length

        bckptr = {}
        pi = {(0, INIT_STATE, INIT_STATE): 0.0}  # base case

        for i, token in enumerate(tokenArr):  # look at each word
            i = i+1

            # if a word isn't in the vocabulary, rename with OOV symbol
            if token not in vocabulary:
                token = OOV_SYMBOL

            for currTag, prevtag, prevprevtag in itertools.product(tags, tags, tags):
                if (currTag, token) in emissionProb and (i-1, prevprevtag, prevtag) in pi:
                    score = pi[(i - 1, prevprevtag, prevtag)] + emissionProb[(currTag, token)] + getProbability(prevprevtag, prevtag, currTag)

                    if (i, prevtag, currTag) not in pi or score > pi[(i, prevtag, currTag)]: #fixme are the prevtag and currtags correct
                        pi[(i, prevtag, currTag)] = score
                        bckptr[(i, prevtag, currTag)] = prevprevtag
                    else:
                        pass
                else:
                    pass

        # account for end of sentence
        foundgoal = False
        curMax = INIT_STATE
        prevMax = INIT_STATE
        goal = float("-inf")

        for previous, tg in itertools.product(tags, tags):
            if(n, previous, tg) in pi:
                score = pi[(n, previous, tg)] + getProbability(previous, tg, FINAL_STATE)
                if not foundgoal or score > goal:
                    goal = score
                    foundgoal = True
                    prevMax = previous
                    curMax = tg
            else:
                pass

        # backtrace
        if foundgoal:
            finalTags = []

            current = curMax
            previous = prevMax

            for i in xrange(n, 1, -1):

                copyCurrent = current
                copyPrevious = previous

                current = previous
                previous = bckptr[(i, copyPrevious, copyCurrent)]
                finalTags.append(current)
            finalTags.reverse()
            print ' '.join(finalTags)
            pass
        else:
            print ' '.join([])

