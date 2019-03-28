from nltk import bigrams
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

def gen_bigrams(tokens):

    """I created my own bigram generator function because I forgot I could use nltk...."""
    
    
    bigrams = []
    gram_1 = 0
    gram_2 = 1

    #loop appends bigram tuple to a list
    for i in range(len(tokens)):
        if gram_1 == len(tokens) - 1:
            break
        else:
            bigrams.append((tokens[gram_1], tokens[gram_2]))
            gram_1 += 1
            gram_2 += 1
    
    #Returns a list of tuples, the tuples contains the bigrams 
    return bigrams

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

def gen_bigram_frequency(text):
    text_bigrams = bigrams(text)
    
    unigram_freq = {}

    for i in text:
        if i in unigram_freq:
            unigram_freq[i] += 1
        else: 
            unigram_freq[i] = 1
    for i in unigram_freq:
        unigram_freq[i] += 1

    bigram_freq = {}
    for i in text_bigrams:
        if i in bigram_freq:
            bigram_freq[i] += 1
        else:
            bigram_freq[i] = 1
    #print(bigram_freq)
    for i in bigram_freq:
        bigram_freq[i] += 1
    probs = {}
    return bigram_freq

'''^The above are all processing^-----'''

def test_set(text):
    
    training_set = bigram_frequency_model
    
    for i in text:
        test_bigrams = gen_bigrams(i)
      
    
    
    test_freqs = {}
    #print(training_set)
    for i in test_bigrams: 
        
        if i in test_freqs:
            test_freqs[i] += 1
        else:
            test_freqs[i] = 1
    #print(test_freqs)
    
    for key, value in test_freqs.items():
        if key not in training_set:
            training_set[key] = value
    
    
    #print(training_set)
    return training_set

'''------Everything above works------'''

def gen_perplexity(training_set, test):
    probs = []
    test_bigrams = list(bigrams(test))
    
    #print(training_set)
    for bigram in test_bigrams:

        if bigram in training_set:
            probs.append(training_set[bigram])
        else:
            print(bigram)
            pass
    #print(training_set)    
    #print(training_set)
       
    
    
    #print(training_set)
    #print(probs)
    print(len(training_set))
    print(prod(probs)**((-1)/len(training_set)))

def gen_probability(bigram_freq): #unigram_freq in arg 1 for bigram prob (maybe wrong one)
    
    unigram_freq = gen_unigram_frequency(training_tokens[0])

    probs = {}
    for key, value in bigram_freq.items():
        probs[key] = value/sum(bigram_freq.values())
    #This look makes a dictionary with the equation that you are not sure if is the right one. 
    '''
    probs2 = {}
    for key, value in bigram_freq.items():
        thing = key[0]
        
        if key[0] in unigram_freq:
            probs2[key] = (value / unigram_freq[thing] )'''
    '''unigram_freq[thing]9'''
    return bigram_probabilies
    
'''------^Not sure if we use these guys^----'''

def testing_model(lang_name, test_file, test_sent):
    
    correct = training_lang
    
    training_model = test_set(test_file) #training model has the bigram frequencies of training plus the test bigram freqs. everything has been smoothed with +1 smoothing
    
    top_freq = {}
    for key, value in training_model.items():
        if value > 5:
            top_freq[key] = value
            
    

    test = []
    sent_bigrams = []
    
    for sentence in test_sent:
        test.append(gen_bigrams(sentence))
    
    for sentence in test:
        probs = []
        for bigram in sentence:
            #print(bigram)
            if bigram not in training_model:
                training_model[bigram] = 1

            else:
                training_model[bigram] += 1

    
    final_model = {}
    
    
    
    sentence_probs = []
    for sentence in test:
        #print(sentence)
        probs = []
        for bigram in sentence: 
            probs.append(math.log2(training_model[bigram]/len(training_model))) #sum(training_model.values())))
            
        
        sentence_probs.append(-sum(probs))
    #print(max(sentence_probs))
    #return lang_name, max(sentence_probs)
        
    
    print((lang_name, sentence_probs))



if __name__ == "__main__":
    
    lang_name = ["french", "english", "dutch", "italian", "german","spanish"]
    lang_path = ["train/french/french.txt", "test/english/english copy.txt", "test/dutch/dutch.txt", "test/italian/italian.txt", "test/germany/germany.txt", "test/spanish/spanish.txt" ]
    test_path = ["test/french/french copy.txt", "test/english/english copy.txt", "test/dutch/dutch copy.txt", "test/italian/italian copy.txt", "test/germany/germany copy.txt", "test/spanish/spanish copy.txt"]
    language_models = {}
    count = 0
    for i in lang_path
    
    '''
    pount = 0
    for i in lang_path:
        count = 0
        training_lang = lang_name[pount]
        training_tokens = pre_process_text(i)[0]
        bigram_frequency_model = gen_bigram_frequency(training_tokens)
        print(training_lang)
        pount += 1
        for i in test_path:
            test_tokens = pre_process_text(i)[1]
            train_test_bigram_freqs = test_set(test_tokens)
            testing_model(lang_name[count], train_test_bigram_freqs, test_tokens)
            count += 1'''

    '''
    pount = 0
    for i in test_path:
        count = 0
        test_lang = lang_name[pount]
        print(test_lang)
        test_tokens = pre_process_text(i)[1]
        pount += 1
        for i in lang_path:
            training_lang = lang_name[count]
            training_tokens = pre_process_text(i)[0]
            bigram_frequency_model = gen_bigram_frequency(training_tokens)
            train_test_bigram_freqs = test_set(test_tokens)
            testing_model(training_lang, train_test_bigram_freqs, test_tokens)
       
            count += 1'''
    '''for i in lang_path:
        training_tokens = pre_process_text(i)[0]
        bigram_frequency_model = gen_bigram_frequency(training_tokens)
        test_tokens = pre_process_text(i)[1]
        train_test_bigram_freqs = test_set(test_tokens)
        testing_model(lang_name[count], train_test_bigram_freqs, test_tokens)
        count += 1'''

        



    
    '''
    tokenized_file = pre_process_text(lang_path[1])[0]
    #print(tokenized_file)
    train_bigram_freq = gen_bigram_frequency(tokenized_file)
    
    test_set_1 = pre_process_text(test_path[0])[0]
    test_set_1_sent = pre_process_text(test_path[0])[1]
    test_set_2 = pre_process_text(test_path[1])[0]
    test_set_2_sent = pre_process_text(test_path[1])[1]
    test_set_3 = pre_process_text(test_path[2])[0]
    test_set_3_sent = pre_process_text(test_path[2])[1]
    test_set_4 =pre_process_text(test_path[3])[0]
    test_set_4_sent = pre_process_text(test_path[3])[1]
    test_set_5 = pre_process_text(test_path[4])[0]
    test_set_5_sent = pre_process_text(test_path[4])[1]
    test_set_6 = pre_process_text(test_path[5])[0]
    test_set_6_sent = pre_process_text(test_path[5])[1]


    training_1 = test_set(test_set_1)
    training_2 = test_set(test_set_2)
    training_3 = test_set(test_set_3)
    training_4 = test_set(test_set_4)
    training_5 = test_set(test_set_5)
    training_6 = test_set(test_set_6)

    print(testing_model(lang_name[0], test_set_1, test_set_1_sent))
    print(testing_model(lang_name[1],test_set_2, test_set_2_sent))
    print(testing_model(lang_name[2], test_set_3, test_set_3_sent))
    print(testing_model(lang_name[3],test_set_4, test_set_4_sent))
    print(testing_model(lang_name[4], test_set_5, test_set_5_sent))
    print(testing_model(lang_name[5], test_set_6, test_set_6_sent))
    '''

    
    '''------CODE GRAVE YARD-----'''
     
    ''' TES SET THAT WORKS!!
    wood_train = ['hello', 'i', 'am', 'Rob','how', 'are', 'you']
    wood_test1 = ['Guten', 'tag', 'ich', 'bin', 'Rob', 'wie', 'gehts']
    wood_test2 = ['Bonjour', 'Je', 'M"apelle', 'Rob', 'ca', 'va'] 
    wood_test3 = ['hello', 'i', 'am', 'Rob', 'how', 'are', 'you', 'doing']

    train_unigram_freq = gen_unigram_frequency(wood_train)
    train_bigram_freq = gen_bigram_frequency(wood_train)
    
    training_1 = test_set(wood_test1)
    
    training_2 = test_set(wood_test2)
    
    training_3 = test_set(wood_test3)

    testing_model(training_1, wood_test1)
    testing_model(training_2, wood_test2) 
    testing_model(training_3, wood_test3)
   

    '''



    '''
    train = ['i', 'am', 'Rob', 'and', 'i', 'am', 'happy', 'and', 'i', 'am', '21']
    test = ['i', 'am', 'Gerge', 'and', 'i', 'am', 'happy']
    test_2 = ['i', 'and', 'am', 'Sam', 'am', 'i', 'and', 'happy']
    #print(gen_bigram_frequency(train))
    train_unigram_freq = gen_unigram_frequency(wood_train)
    #print(train_unigram_freq) training unigram frequencies
    train_bigram_freq = gen_bigram_frequency(wood_train)
    #print(train_bigram_freq) training set unigram frequencies 
    
    training = test_set(wod_1)
    
    #print(training) #This is updated probabilities with smoothed bigrams
    training_2 = test_set(wod_2)'''
    

    
    

    '''
    wood_train = ['hello', 'i', 'am', 'Rob','how', 'are', 'you']
    wood_train2 = ['Guten', 'tag', 'wie', 'gehts', 'deiner','mushy']
    wood_train3 = ['ca','va', 'Je', 'suis', 'Rob', 'putan','de','merd']
    wood_test1 = [['Guten', 'tag', 'ich', 'bin', 'Rob', 'wie', 'gehts']]
    wood_test2 = [['Bonjour', 'Je', 'M"apelle', 'Rob', 'ca', 'va']] 
    wood_test3 = [['hello', 'i', 'am', 'Rob', 'how', 'are', 'you', 'doing']]

    train_unigram_freq = gen_unigram_frequency(wood_train)
    train_bigram_freq = gen_bigram_frequency(wood_train)
    
    training_1 = test_set(wood_test1)
   
    print(testing_model('german', training_1, wood_test1))
    
    '''