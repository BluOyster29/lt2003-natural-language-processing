from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk import bigrams, trigrams
import re, math, numpy

"""First I want to generate a bigram model and unigram model"""

def tokenize(text):
    words = re.findall(r"<[a-z]+>|"
                       r"[A-Z]?[a-z]+", text) 
    return words

def gen_probability(ngrams):
    
    n_gram_freq_smoothed = {}
    n_gram_freq = {}
    
    for i in ngrams:
        
        if i not in n_gram_freq:
            n_gram_freq.update({i : 1})
        else:
            n_gram_freq[i] += 1
    
    
    
    for i in ngrams:
        
        if i not in n_gram_freq_smoothed:
            n_gram_freq_smoothed.update({i : 1})
        else:
            n_gram_freq_smoothed[i] += 1

    for keys in n_gram_freq_smoothed:
        n_gram_freq_smoothed[keys] += 1
     
    n_gram_freq_smoothed.update({'it' : 1})
    n_gram_prob = {}

    return n_gram_freq_smoothed

def gen_probs(dictio):

    n_gram_prob = {}
    
    for key, val in dictio.items():
        n_gram_prob.update({key : val /sum(dictio.values())})

    return n_gram_prob

def gen_logs(prob_dict):
    
    logs = {}

    for key, value in prob_dict.items():
        logs.update({key : value *  (math.log(value, 2))})

    return -sum(logs.values())

def training_set(tokens):
    probabilities = gen_probability(tokens)
    probs = gen_probs(probabilities)
    return probs


def test_set(text):
    #difining test set 

    test_set = {}
    get_train = training_set(unigrams)
    
    get_probs = gen_probability(text)
    print(get_train)
    #print(gen_probs(get_probs))
    
if __name__ == "__main__":
    text = r"<start> how much wood would a woodchuck chuck if a woodchuck could chuck wood a woodchuck would chuck as much wood as a woodchuck could chuck if a woodchuck could chuck wood <end>"
    unigrams = tokenize(text)
    text_Q2A = r"<start> would a woodchuck chuck wood if it could chuck wood <end>"
    text_Q2B = r"<start> wood a woodchuck chuck would if it could chuck would <end>"


    text_1_bi = [0.5,0.3,0.6,1,0.3,0.5,0.2,0.3,1,1,0.5,0.4]
    text_2_bi = [0.5,0.5,0.4,1,0.3,0.16,0.3,1,1,0.16,0.3]

    text_1_uni = [0.0434,0.0652,0.1304,0.1304,0.1304,0.1086,0.0652,0.0217,0.0869,0.1304,0.1086,0.0434]
    text_2_uni =  [0.0434,0.1086,0.1304,0.1304,0.1304,0.0652,0.0652,0.0217,0.0869,0.1304,0.0652,0.0434]
    
    text_logs = [math.log(i, 2) for i in text_1_bi]
    text_logs_2 = [math.log(i, 2) for i in text_2_bi]

    text_log_uni_1 = [math.log(i, 2) for i in text_1_uni]
    text_log_uni_2 = [math.log(i, 2) for i in text_2_uni]
    
    
    
    print(-sum((text_logs)))
    print(-sum((text_logs_2)))
    print(-sum(text_log_uni_1))
    print(-sum(text_log_uni_2))

    print(numpy.prod(text_1_bi)**((-1)/12))
    print(numpy.prod(text_2_bi)**((-1)/12))
    print(numpy.prod(text_1_uni)**((-1)/12))
    print(numpy.prod(text_2_uni)**((-1)/12))
    
    unigrams = tokenize(text)
    bigrams = list(bigrams(unigrams))
    unigram_frequencies = gen_probability(unigrams)
    bigram_frequencies = gen_probability(bigrams)
    unigram_probabilities = gen_probs(unigram_frequencies)
    bigram_probabilities = gen_probs(bigram_frequencies)
    #print(unigram_frequencies)
    #Entropy of unigram probs print(gen_logs(unigram_probabilities))
    #Entropy of bigram probs print(gen_logs(bigram_probabilities))
    #print(unigram_probabilities)
    unigrams_test = gen_probability(tokenize(text_Q2A))
    #print(unigrams_test)



"""1. Creating the word dictionary [Coding only: save code as problem1.py or problem1.java] The first step in 
   building an n-gram model is to create a dictionary that maps words to java map or python dictionary (which 
   we’ll use to access the elements corresponding to that word in a vector or matrix of counts or probabilities).
   You’ll create this dictionary from the given data files (Select one file for training purpose) for all unique words. 
   You’ll need to split the sentences (consider each line) into a list of words and convert each word to lowercase, before
   storing it to the dictionary."""