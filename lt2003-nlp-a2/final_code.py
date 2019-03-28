from nltk import bigrams, trigrams
from numpy import prod
from nltk import word_tokenize, sent_tokenize
import re
import math

def pre_process_text(file_path):

    """Function that will process the text files by removing digits, special characters 
       etc."""

    #read the train file 
    with open(file_path, "r", encoding="utf-8") as train_file:
        sent = []
        words = []
        for line in train_file:
            words.append(re.sub(r"\d+|" r"[\.\,\"\'\(\)\?\%\!\?\&\@\#\€\$\∞\§\|\[\]\©\:\;\\\/]|"  r"[\t\n]", "", line).lower())
            sent.append([re.sub(r"\d+|" r"[\.\,\"\'\(\)\?\%\!\?\&\@\#\€\$\∞\§\|\[\]\©\:\;\\\/]|"  r"[\t\n]", "", line).lower()])
            
    word = "".join(words)
    tokenized_sent = []
    for sentence in sent:
        for token in sentence:
            tokenized_sent.append(word_tokenize(token))
    return word_tokenize(word), tokenized_sent #function returns whole tokenized text (for training) and sentence tokenised (for testing)

def gen_unigram_frequency(text):
    unigram_freq = {}

    for i in text:
        if i in unigram_freq:
            unigram_freq[i] += 1
        else: 
            unigram_freq[i] = 1
    for i in unigram_freq:
        unigram_freq[i] += 1

    for i in unigram_freq:
        unigram_freq[i] += 1

    return unigram_freq

def gen_ngram_frequency(text):
    text_bigrams = bigrams(text)
    text_trigrams = trigrams(text)
    '''Loop that adds unseen unigram frequencies
    unigram_freq = {}

    for i in text:
        if i in unigram_freq:
            unigram_freq[i] += 1
        else: 
            unigram_freq[i] = 1
    for i in unigram_freq:
        unigram_freq[i] += 1
    '''
    
    bigram_freq = {}
    for i in text_bigrams:
        if i in bigram_freq:
            bigram_freq[i] += 1
        else:
            bigram_freq[i] = 1
    
    trigram_freq = {}
    for i in text_trigrams:
        if i in bigram_freq:
            trigram_freq[i] += 1
        else:
            trigram_freq[i] = 1

    return bigram_freq, trigram_freq

def gen_top_frequencies(ngram_freq):
    top_frequencies = {}

    for key,value in ngram_freq.items():
        if value > 5:
            top_frequencies[key] = value

    return top_frequencies

def smooth_frequencies(training_ngram_freq, test_tokens_freq):
    
    #plus one smoothing training dictionary
    for key in training_ngram_freq[1]:
        training_ngram_freq[1][key] += 1
    
    
    #adding unseen bigrams to training freqs
    for key, value in test_tokens_freq[1].items():
        if key not in training_ngram_freq[1]:
            training_ngram_freq[1][key] = value
    
    
    return training_ngram_freq

def testing_model():
    #list_of_probs = {}
    #for g in range(len(lang_name)):    
    #    for b in range(1,101):
    #        
    #        list_of_probs[lang_name[g]," sentence " + str(b)] = []
    list_of_probs = {}
    for x in range(len(lang_name)):
        training_models = gen_training_models(lang_name, lang_path)[0][x]#bigram or trigram[1]#language[1])#Language name[0])
        #print(lang_name[x])
        
        
        for i in range(len(lang_name)):
            test_models = gen_training_models(lang_name, test_path)[0][i]#generating test models 
            final_test_model = smooth_frequencies(training_models, test_models) #this is the test and training models smoothed together. '''
            #print(final_test_model)
    
            test_file_bigrams = []
            test_file_trigrams = []
            for sentence in pre_process_text(test_path[i])[1]:
                test_file_bigrams.append(list(bigrams(sentence)))
    
            for sentence in pre_process_text(test_path[i])[1]:
                test_file_trigrams.append(list(trigrams(sentence)))
            
            
            
            
            num = 1
            for sentences in test_file_bigrams:
                probs = []
                for ngram in sentences:
                    if ngram in final_test_model[1]:
                        probs.append(math.log(final_test_model[1][ngram]/sum(final_test_model[1].values()),2))
                    else:
                        probs.append(math.log(1/sum(final_test_model[1].values()),2))
                


                if (lang_name[x], " sentence " + str(num)) in list_of_probs:
                    list_of_probs[lang_name[x], " sentence " + str(num)].append((lang_name[i],-sum(probs)))
                else:
                    list_of_probs[lang_name[x], " sentence " + str(num)] = [(lang_name[i],-sum(probs))]
                    
                num += 1
        accuracy_tester(list_of_probs)
        

        #return lang_name[0], list_of_probs'''
    
def accuracy_tester(sentence_probs):
    probsa = {}
    correct = 0
    incorrect = 0
    for key, item in sentence_probs.items():
        for probs in item:
            probsa[probs[0]] = probs[1]
        
        maximum = max(probsa.items(), key=lambda k: k[1])

        if maximum[0] == key[0]:
            
            correct += 1

        else:
            
            
            incorrect += 1
        
        
    print(correct, incorrect)
    '''
    maximum = max(prob.items(), key=lambda k: k[1])
    

    if maximum[0] == train_lang:
        print('correct')
    
    else:
        print('incorrect')'''

        

def gen_training_models(lang_name, lang_path):

    list_of_bi_models = []
    list_of_tri_models = []

    for i in range(len(lang_name)):
        bi_model = (lang_name[i], gen_ngram_frequency(pre_process_text(lang_path[i])[0])[0])
        list_of_bi_models.append(bi_model)
        tri_model = gen_ngram_frequency(pre_process_text(lang_path[i])[0])[1]
        list_of_tri_models.append((lang_name[i], tri_model))
    
    return list_of_bi_models, list_of_tri_models

if __name__ == "__main__":
    
    lang_name = ["french", "english", "dutch", "italian", "german","spanish"]
    lang_path = ["train/french/french.txt", "test/english/english.txt", "test/dutch/dutch.txt", "test/italian/italian.txt", "test/germany/germany.txt", "test/spanish/spanish.txt" ]
    test_path = ["test/french/french.txt", "test/english/english.txt", "test/dutch/dutch.txt", "test/italian/italian.txt", "test/germany/germany.txt", "test/spanish/spanish.txt"]
    lang_path = ["practice_train/french/french.txt", "practice_train/english/english.txt", "practice_train/dutch/dutch.txt", "practice_train/italian/italian.txt", "practice_train/germany/germany.txt", "practice_train/spanish/spanish.txt" ]
    #test_path = ["practice_test/french/french.txt", "practice_test/english/english.txt", "practice_test/dutch/dutch.txt", "practice_test/italian/italian.txt", "practice_test/germany/germany.txt", "practice_test/spanish/spanish.txt"]
    
    
    
    
    training_models = gen_training_models(lang_name, lang_path)[0][0]#bigram or trigram[1]#language[1])#Language name[0])
    test_models = gen_training_models(lang_name, test_path)[0][1]#generating test models 
    final_test_model = smooth_frequencies(training_models, test_models) #this is the test and training models smoothed together. 
    

    testing_model()

    

    '''
    for x in range(len(lang_name)):
        training_models = gen_training_models(lang_name, lang_path)[0][x]#bigram or trigram[1]#language[1])#Language name[0])
        print(lang_name[x])
        for i in range(len(lang_name)):
            test_models = gen_training_models(lang_name, test_path)[0][i]#generating test models 
            final_test_model = smooth_frequencies(training_models, test_models) #this is the test and training models smoothed together. 
            print(testing_model(i, final_test_model, pre_process_text(test_path[i])[1]))'''
    
    
    #prob_list = []
    #for i in range(len(lang_name)):
    #    prob_list.append(testing_model(i, final_test_model, pre_process_text(test_path[i])[1]))
    #    accuracy_tester(prob_list)
    #
    #print(prob_list)

    





    '''
    def testing(test_file):
    
    test_file_bigrams = []
    
    for i in test_file:
            test_file_bigrams.append(list(bigrams(i)))
    test_file_trigrams = []
    for i in test_file:
            test_file_trigrams.append(list(trigrams(i)))
    
    
    for sentence in test_file_bigrams:
        probs = []
        for ngram in sentence:
            if ngram in testing_set[1]:
                probs.append(math.log(testing_set[1][ngram]/sum(testing_set[1].values()),2))
            else:
                probs.append(math.log(1/sum(testing_set[1].values()),2))
        
        return [lang_name[0],-sum(probs)]'''
        
