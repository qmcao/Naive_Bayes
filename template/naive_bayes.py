# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    ##TODO:
    i = 0
    for email in X:
        for word in email:
            if y[i] == 1:
                if word in pos_vocab:
                    pos_vocab[word] += 1
                else:
                    pos_vocab[word] = 1
            else:
                if word in neg_vocab:
                    neg_vocab[word] += 1
                else:
                    neg_vocab[word] = 1
        i +=1
    
    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    ##TODO:
    i = 0
    
    for email in X:
        for j in range(len(email) - 1):
            bigram_word = email[j] + email[j + 1]
            if y[i] == 1:
                if bigram_word in pos_vocab:
                    pos_vocab[bigram_word] += 1
                else:
                    pos_vocab[bigram_word] = 1
            else:
                if bigram_word in neg_vocab:
                    neg_vocab[bigram_word] += 1
                else:
                    neg_vocab[bigram_word] = 1
    return dict(pos_vocab), dict(neg_vocab)


# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)
    dev_labels = []
    
    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels)
    
    #print(pos_vocab)

    
    # First we need to calculate N(Y = y)
    # N(Y = 1) = pos_toks =  the total number of word tokens that exist in e-mails that are labeled Y=1
    # N(Y = 0) = pos_toks =  the total number of word tokens that exist in e-mails that are labeled Y=0
    pos_toks, neg_toks = total_tokens(pos_vocab, neg_vocab) 
    
    prio_ham, prio_spam = pos_prior, 1 - pos_prior
    
    
    for email in dev_set:
        log_p_ham = np.log(prio_ham)
        log_p_spam = np.log(prio_spam)
        for word in email:      
            pos_count = 0
            neg_count = 0
            
            if word in pos_vocab: # check if word exist in training set
                pos_count = pos_vocab[word]
            
            if word in neg_vocab:
                neg_count = neg_vocab[word]
            
            log_p_ham += np.log(likelihood_prob_ham(pos_count, laplace, pos_toks, len(pos_vocab)))
            log_p_spam += np.log(likelihood_prob_spam(neg_count, laplace, neg_toks, len(neg_vocab)))
        
    
        if log_p_ham > log_p_spam:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
        

    

    return dev_labels


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None

    dev_labels = []
    
    pos_vocab_bi, neg_vocab_bi = create_word_maps_bi(train_set, train_labels)
    pos_vocab_uni, neg_vocab_uni = create_word_maps_uni(train_set, train_labels)
    
    #N(Y = y) for unigram word ie, sum of counts of each bigram word
    pos_toks_uni, neg_toks_uni = total_tokens(pos_vocab_uni, neg_vocab_uni)
    
    #N(Y = y) for bigram word ie, sum of counts of each unigram word
    pos_toks_bi, neg_toks_bi = total_tokens(pos_vocab_bi, neg_vocab_bi)
    
    pos_toks_bi += pos_toks_uni
    neg_toks_bi += neg_toks_uni
    
    #prior probability
    prio_ham, prio_spam = pos_prior, 1 - pos_prior
    
    ham_distinct_word_uni = len(pos_vocab_uni)
    spam_distinct_word_uni = len (neg_vocab_uni)
    
    ham_distinct_word_bi = len(pos_vocab_bi) + ham_distinct_word_uni
    spam_distinct_word_bi = len(neg_vocab_bi) + spam_distinct_word_uni
    
    first_term_pos = (1 + bigram_lambda) * np.log(pos_prior)
    first_term_neg = (1 + bigram_lambda) * np.log(1 - pos_prior)
    
    for email in dev_set:
        ham_sum = 0
        spam_sum = 0
        for j in range(len(email) - 1):
            # get bigram and unigram word
            bigram_word = email[j] + email[j + 1]
            uni_word = email[j]
            
            #count for bigram word given label
            pos_count_bi = 0
            neg_count_bi = 0
            
            #count for unigram word given label
            pos_count_uni = 0
            neg_count_uni = 0
            
            # if bigram word exist in training set
            if bigram_word in pos_vocab_bi:
                pos_count_bi += pos_vocab_bi[bigram_word]
            if bigram_word in neg_vocab_bi:
                neg_count_bi += neg_vocab_bi[bigram_word]
            
            # if unigram word exist in training set
            if uni_word in pos_vocab_uni:
                pos_count_uni += pos_vocab_uni[uni_word]
            if uni_word in neg_vocab_uni:
                neg_count_uni += neg_vocab_uni[uni_word]
            
        
            ham_sum -=( np.log(likelihood_prob_ham(pos_count_uni, unigram_laplace, pos_toks_uni, ham_distinct_word_uni)) -
                   np.log(likelihood_prob_ham(pos_count_bi, bigram_laplace, pos_toks_bi, ham_distinct_word_bi)))
        
            spam_sum -= (np.log(likelihood_prob_spam(neg_count_uni, unigram_laplace, neg_toks_uni, spam_distinct_word_uni)) -
                     np.log(likelihood_prob_spam(neg_count_bi, bigram_laplace, neg_toks_bi, spam_distinct_word_bi)))
        
        ham_sum *= - bigram_lambda
        spam_sum *= - bigram_lambda
        
        ham_sum += first_term_pos
        spam_sum += first_term_neg


        if ham_sum > spam_sum:
            dev_labels.append(1)
        else:
            dev_labels.append(0)            
            
    
    

    return dev_labels


def total_tokens(pos_vocab, neg_vocab):

    pos_token = 0
    neg_token = 0    
    for word in pos_vocab:
        pos_token += pos_vocab[word]
    
    for word in neg_vocab:
        neg_token += neg_vocab[word]
        
    return pos_token, neg_token

def likelihood_prob_ham(pos_count, laplace, total_tokens, distict_pos_wordcount):
    '''
    Calculate the likelihood of each word in an ham email
    '''
    p = (pos_count + laplace) / (total_tokens + laplace * distict_pos_wordcount)
    return p


def likelihood_prob_spam(neg_count, laplace, total_tokens, distict_neg_wordcount):
    '''
    Calculate the likelihood of each word in an spam email
    '''
    p = (neg_count + laplace) / (total_tokens + laplace * distict_neg_wordcount)
    return p

