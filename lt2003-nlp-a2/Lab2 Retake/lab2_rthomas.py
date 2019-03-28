from nltk import word_tokenize, sent_tokenize, bigrams, trigrams, FreqDist
import re, math, operator

"""
Lab2:Language Identification
Student: Robert Thomas 
Status: Resubmission
"""

def pre_process_text(file_path):

    """Function that will process the text files by removing digits, special characters 
       etc."""
    sent = [] 
    word = []
    with open(file_path, "r", encoding="utf-8") as train_file:
        for line in train_file:
            x = re.sub(r"[\d\[\]\(\)\.\,\:\;\!\*%&\?€#@£∞§\|{}$]*", "", line).lower()
            word += x.split()
            sent += [x.split()]
    return word, sent #, tokenized_sent #function returns whole tokenized text (for training) and sentence tokenised (for testing)

def gen_training_bigram_model(lang_path):
    
    """This function creates 6 dictionaries to be used as the training sets. The function outputs both the 
    bigram and trigram models. This is done by iterating through each language name in the list, generating 
    the test, bigrams and trigrams, then placing them in a dictionary with the top bi/trigram relative frequencies"""
    
    bi_training_sets = [] #List will contain the a 2-tuple, the language name and dictionary with bigram relative frequencies
    bi_training_tokens = {} #This counts the total number of bigrams

    for i in range(len(lang_path)):

        """Loop for creating the relative bigram frequncies"""

        bi_training = {}
        test_file = pre_process_text(lang_path[i])[0] #list of tokenised words
        text_bigrams = list(bigrams(test_file)) #built in nltk bigram function for turning list of words into bigrams
        bi_freq_dist = dict(FreqDist(text_bigrams)) #built in function creates frequency distribuition of bigrams
        bi_total_tok = len(text_bigrams) # storing the total number of bigrams 
        bi_training_tokens[lang_name[i]] = bi_total_tok #adding total to a dictionary 
        
        for key, value in bi_freq_dist.items():
            #loop for finding more frequent bigrams
            if value > 20:
                bi_training[key] = value/bi_total_tok
            else:
                continue
        bi_training_sets.append((lang_name[i], bi_training))#language name and bigram frequncy added to list
    #print(bi_training_tokens)
    return bi_training_sets, bi_training_tokens #Returns list of 2-tuple and dictionary

def gen_training_trigram_model(file_path):

    """This function is identical to the previous except it creates the trigram training set.
    All steps are identical and variables perform the same action"""

    tri_training_sets = []
    tri_training_tokens = {}
    for i in range(len(lang_path)):
        tri_training = {}
        test_file = pre_process_text(lang_path[i])[0]
        text_trigrams = list(trigrams(test_file))
        tri_freq_dist = dict(FreqDist(text_trigrams))
        tri_total_tok = len(text_trigrams)
        tri_training_tokens[lang_name[i]] = tri_total_tok
        for key, value in tri_freq_dist.items():
            if value > 1:
                tri_training[key] = value/tri_total_tok
            else:
                continue
        tri_training_sets.append((lang_name[i], tri_training))
    return tri_training_sets, tri_training_tokens

def gen_testing_bigram_model(bigram_training_set, test_path):
    
    """This function tests the file based on the bigram language model. To do this 
    the function generates a test file and appends to a list the probabilities of
    each bigram in the test sentence based on the probability from the language model.
    The function returns the accuracy of the testing as well as the, per language, accuracies
    and stats as well as the overall accuracy of the experiment"""
    
    correct = 0 #recording global accuracy
    incorrect = 0
    testing_accuracy = {}
    bi_stats = {} #dictionary to record total number of bigrams
    for i in range(len(test_path)):
        local_correct = 0 #for storing accuracy line by ine 
        local_incorrect = 0
        test_file = pre_process_text(test_path[i])[1] #generating test file
        total_bigrams = sum(dict(FreqDist(pre_process_text(test_path[i])[0])).values()) 
        #total bigrams will be the sum of the frquency distrubtion of all bigrams
        
        for line in test_file:
            model_probs = {} #to store all possible probabilities per training language 
            line_bigrams = list(bigrams(line)) #generates bigrams by line
            
            for num in range(0,6): 

                '''This loop goes through each bigram in the line and appends the probabilities 
                to a list. For bigrams that are not in the training set I have given them a very
                small probability so that we can avoid dividing by zero.'''

                probs = list()
                for token in line_bigrams:
                    if token not in bigram_training_set[num][1].keys():
                        probs += [math.log2(0.00000000000000000000000000000000000001)]
                    else:
                        probs += [math.log2(bigram_training_set[num][1][token])]
                model_probs[bigram_training_set[num][0]] = sum(probs) #have the joint probability of the line for each training language

            training_language = max(model_probs, key=model_probs.get) #program coosing language with highest probability           
            actual_language = lang_name[i] #storing the actual language of the test sentence
            
            if actual_language == training_language: #statements for recording accuracies
                correct += 1  #global accuracy
                local_correct += 1 #line by line, 'local' accuracy
            else:
                local_incorrect += 1
                incorrect += 1

        bi_stats[lang_name[i]] = total_bigrams #counting total bigrams, to be used in generating statistics
        testing_accuracy[lang_name[i]] = (local_correct,local_incorrect) #recording local accuracies to be used later  
    bigram_accuracy = correct/(correct+incorrect)  #global accuracy of the bigram language model experiment

    '''The function returns the total accuracy of the experiment in the form of an integer, a dictionary that contains
    the bigram counts for each language and a dictionary with the local accuries for each language'''

    return bigram_accuracy, bi_stats, testing_accuracy
  
def gen_testing_trigram_model(trigram_training_set, test_path):
    
    '''This function is identical to testing_bigram_model, however we are using the trigram
    model and generating trigrams for testing''' 
    
    tri_stats = {}
    correct = 0
    incorrect = 0
    testing_accuracy = {}
    for b in range(len(test_path)):
        local_correct = 0
        local_incorrect = 0
        test_file = pre_process_text(test_path[b])[1]   
        total_trigrams = sum(dict(FreqDist(pre_process_text(test_path[b])[0])).values())  
        
        for line in test_file:
            line_trigrams = list(trigrams(line)) #built in nltk function generates trigrams           
            tri_model_probs = {}
            for bum in range(0,6):
                probs = list()
                for token in line_trigrams:
                    if token not in trigram_training_set[bum][1].keys():
                        probs += [math.log2(0.000000000000000000000000001)]
                    else:
                        probs += [math.log2(trigram_training_set[bum][1][token])]                
                tri_model_probs[trigram_training_set[bum][0]] = sum(probs)
            training_language = max(tri_model_probs, key=tri_model_probs.get)           
            actual_language = lang_name[b]
            if actual_language == training_language:
                correct += 1
                local_correct += 1
            else:
                incorrect += 1   
                local_incorrect += 1
        testing_accuracy[lang_name[b]] = (local_correct, local_incorrect)
        tri_stats[lang_name[b]] = total_trigrams       
    trigram_accuracy = correct/(correct+incorrect)    

    '''The function returns the same data as the previous function however it is the trigram
    experiment'''

    return trigram_accuracy, tri_stats, testing_accuracy

def gen_corpus_statistics(count_train_bigrams, count_test_bigrams, count_train_trigrams, count_test_trigrams, bi_model_accuracy, tri_model_accuracy):
    
    '''This function prints to the screen the corpus statistics for each language model'''
    
    print(line_break + "\n\nCorpus Statistics\n\n" + line_break)
    for i in lang_name:
        print("\nTraining Language: " + i)
        total_bigrams = count_train_bigrams[i] #total bigrams in the train and test set
        total_test_bigrams = count_test_bigrams[i]
        total_trigrams = count_train_trigrams[i] #total trigrams 
        total_test_trigrams = count_test_trigrams[i]
        print("\nTotal Lines: " + str(bi_model_accuracy[i][0] +bi_model_accuracy[i][1]) + "\nTotal Training Bigrams: " + str(total_bigrams)+ "\nTotal Test Bigrams: "+ str(total_test_bigrams) + "\nTotal Training Trigrams:  " +str(total_trigrams) + "\nTotal Test Trigrams: " + str(total_test_trigrams))
        #counts how many lines by adding the correctly guessed and the incorrectly guessed
    print("\n") #for aesthetic

def gen_bigram_model_statistics(bigram_test_accuracy, count_train_bigrams, count_test_bigrams, lang_model_accuracy):
    
    '''This function prints the results of the bigram experiment for each language.''' 
    
    print(line_break + "\n\nBigram Model Statistics\n\n" + line_break)
    
    for i in lang_name: #loop iterates through each language and prints statistics

        print("\nTraining Language: " + i)
        print("Total Lines Guessed Correctly: " + str(lang_model_accuracy[i][0])) #takes the local accuracies from experiment 
        print("Total Lines Guessed Incorrectly: " + str(lang_model_accuracy[i][1]))
        print("Total Accuracy: " + str(lang_model_accuracy[i][0] / (lang_model_accuracy[i][0] +lang_model_accuracy[i][1])* 100) + "%")
        #uses the global accuracy stats from the experiment to show the total accuracy of the experiment
    print("\nTotal Bigram Model Accuracy: " + str(bigram_test_accuracy * 100) + "%\n")

def gen_trigram_model_statistics(trigram_test_accuracy, count_train_trigrams, count_test_trigrams, lang_model_accuracy):

    '''again, identical to the bigram_model_statistics just with trigrams'''

    print(line_break + "\n\nTrigram Model Statistics\n\n" + line_break)
    for i in lang_name:
        print("\nTraining Language: " + i)
        print("Total Lines Guessed Correctly: " + str(lang_model_accuracy[i][0]))
        print("Total Lines Guessed Incorrectly: " + str(lang_model_accuracy[i][1]))
        print("Total Accuracy: " + str(lang_model_accuracy[i][0] / (lang_model_accuracy[i][0] +lang_model_accuracy[i][1])* 100) + "%")
    print("\nTotal Trigram Accuracy: " + str(trigram_test_accuracy * 100) + "%\n")

def commence_testing(lang_names, training_sets, test_sets):
    
    '''This is where all the functions are called'''

    
    print("\n\n" +line_break + "\n\nLab2: Language Identification\nStudent: Robert Rhys Thomas\nDeadline: Resubmission")
    bigram_training_set, total_train_bi_counts = gen_training_bigram_model(lang_path)
    trigram_training_set, total_train_tri_counts = gen_training_trigram_model(lang_path)
    bigram_test_accuracy, count_bigram_test, bi_lang_accuracy = gen_testing_bigram_model(bigram_training_set, test_path)
    trigram_test_accuracy, count_trigram_test,tri_lang_accuracy = gen_testing_trigram_model(trigram_training_set, test_path)
    gen_corpus_statistics(total_train_bi_counts, count_bigram_test, total_train_tri_counts, count_trigram_test,bi_lang_accuracy, tri_lang_accuracy)
    gen_bigram_model_statistics(bigram_test_accuracy, total_train_bi_counts, count_bigram_test, bi_lang_accuracy)
    gen_trigram_model_statistics(trigram_test_accuracy, total_train_tri_counts, count_trigram_test,tri_lang_accuracy)

if __name__ == "__main__":
    line_break = "-" * 38
    lang_name = ["french", "english", "dutch", "italian", "german","spanish"]
    lang_path = ["train/french/french.txt", "train/english/english.txt", "train/dutch/dutch.txt", "train/italian/italian.txt", "train/germany/germany.txt", "train/spanish/spanish.txt" ]
    test_path = ["test/french/french.txt", "test/english/english.txt", "test/dutch/dutch.txt", "test/italian/italian.txt", "test/germany/germany.txt", "test/spanish/spanish.txt"]
    commence_testing(lang_name, lang_path, test_path)
    


    