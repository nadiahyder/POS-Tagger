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
testFile = 'ptb.22.txt'
INIT_STATE = "init"
FINAL_STATE = "final"
OOV_SYMBOL = "OOV"

transitionProb = defaultdict(lambda: 0)
triTransitionProb = defaultdict(lambda: 0)
emissionProb =defaultdict(lambda: 0)
vocabulary = defaultdict(lambda: 0)
tags = set()
trigramCounts = defaultdict(lambda: 0) # to keep track of word occurrences
bigramCounts = defaultdict(lambda: 0) # to keep track of word occurrences
unigramCounts = defaultdict(lambda: 0) # to keep track of word occurrences
total = 0
lambdas = [0.0,0.0,0.0]

# read in the HMM and store the probabilities as log probabilities
with open(TAG_FILE) as hmm:
    for line in hmm:
        split = line.split()
        if split[0] == 'trans':
            transitionProb[(split[1], split[2])] = math.log(float(split[3])) # tuple[prevTag, curTag]
            tags.update([split[1], split[2]])

            bigramCounts[(split[1], split[2])]+=1 # update bigram for tag sequence

            unigramCounts[split[1]]+=1 # update unigram
            unigramCounts[split[2]]+=1

            total +=2

        elif split[0] == 'emit':
            emissionProb[(split[1], split[2])] = math.log(float(split[3]))  # tuple[tag, token]
            vocabulary[split[2]] = 1 # fixme what is this for
            tags.update(split[1])  # add the tag to tags list

        elif split[0] == 'tritrans':
            triTransitionProb[(split[1],split[2],split[3])] = math.log(float(split[4])) # tuple[prevprev, prev, tag]
            trigramCounts[(split[1], split[2], split[3])]+=1
            bigramCounts[(split[1], split[2])] += 1
            bigramCounts[(split[2], split[3])] += 1
            unigramCounts[split[1]]+=1
            unigramCounts[split[2]]+=1
            unigramCounts[split[3]]+=1
            total += 3
        else:
            pass

#perform deleted interpolation, looking at all possible trigrams
# lambda1 = lambda2 = lambda3 = 0.0
# for cur, prevone, prevtwo in itertools.product(tags, tags, tags):
#     triCount = trigramCounts[(prevtwo, prevone, cur)]
#     if (triCount > 0):
#         try:
#             tri = float(triCount - 1) / (bigramCounts[(prevtwo, prevone)] - 1)  #tri will always be 0 bc triCount = 1 for all
#         except ZeroDivisionError:
#             tri = 0
#         try:
#             bi = float(bigramCounts[(prevone, cur)] - 1) / (unigramCounts[prevone] - 1)
#         except ZeroDivisionError:
#             bi = 0
#         try:
#             uni = float(unigramCounts[cur] - 1) / (sum(unigramCounts.values()) - 1)
#         except ZeroDivisionError:
#             uni = 0
#
#         maximum = max(tri, bi, uni)
#
#         if maximum == tri:
#             lambda3 += triCount
#         elif maximum == bi:
#             lambda2 += triCount
#         elif maximum == uni:
#             lambda1 += triCount
#
# # normalize lambdas
# weights = [lambda1, lambda2, lambda3]
# try:
#     lambdas = [float(lambda1 / sum(weights)), float(lambda2 / sum(weights)), float(lambda3/sum(weights))]
# except ZeroDivisionError:
#     lambdas = [0,0,0]
#
# def getTransitionProb(prev2, prev1, current):
#     triCount = trigramCounts[tuple([prev2, prev1, current])]
#
#     try:
#         tri = float(triCount) / (bigramCounts[(prev2, prev1)])  # emissionProb[(currTag, token)]
#     except ZeroDivisionError:
#         tri = 0
#     try:
#         bi = float(bigramCounts[(prev1, current)]) / (unigramCounts[prev1])
#     except ZeroDivisionError:
#         bi = 0
#     try:
#         uni = float(unigramCounts[current]) / (total)
#     except ZeroDivisionError:
#         uni = 0
#
#     #print math.log(float(lambdas[2]*tri + lambdas[1]*bi + lambdas[0]*uni))
#     return math.log(float(lambdas[2]*tri + lambdas[1]*bi + lambdas[0]*uni))


# use a backoff model
def getProbability(prev2, prev1, current):
    score = 0
    # triCount = trigramCounts[(prev2, prev1, current)]
    # biCount = bigramCounts[(prev1, current)]
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
        n = len(tokenArr) # sentence length

        bckptr = {}
        pi = {(0, INIT_STATE, INIT_STATE): 0.0} # base case of the recursive equations

        for i, token in enumerate(tokenArr): # look at each word
            i = i+1

            # if a word isn't in the vocabulary, rename with OOV symbol
            if token not in vocabulary:
                token = OOV_SYMBOL

            for currTag, prevtag, prevprevtag in itertools.product(tags, tags, tags):
                if (currTag, token) in emissionProb and (i-1, prevprevtag, prevtag) in pi:
                    score = pi[(i - 1, prevprevtag, prevtag)] + emissionProb[(currTag, token)] + getProbability(prevprevtag, prevtag, currTag)

                    if (i, prevtag, currTag) not in pi or score > pi[(i, prevtag, currTag)]:
                        pi[(i, prevtag, currTag)] = score
                        bckptr[(i, prevtag, currTag)] = prevprevtag
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

