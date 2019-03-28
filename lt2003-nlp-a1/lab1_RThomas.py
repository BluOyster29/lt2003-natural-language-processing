
#module imports
import difflib, nltk, re, matplotlib.pyplot as plt
from nltk import bigrams

#constants 

"""-------------------Corpus Functions---------------------"""

def get_corpus_text(nr_files=199):
    """Returns the raw corpus as a long string.
    'nr_files' says how much of the corpus is returned;
    default is 199, which is the whole corpus.
    """
    fileids = nltk.corpus.treebank_raw.fileids()[:nr_files]
    corpus_text = nltk.corpus.treebank_raw.raw(fileids)
    # Get rid of the ".START" text in the beginning of each file:
    corpus_text = corpus_text.replace(".START", "")
    return corpus_text

def fix_treebank_tokens(tokens):
    """Replace tokens so that they are similar to the raw corpus text."""
    return [token.replace("''", '"').replace("``", '"').replace(r"\/", "/")
            for token in tokens]

def get_gold_tokens(nr_files=199):
    """Returns the gold corpus as a list of strings.
    'nr_files' says how much of the corpus is returned;
    default is 199, which is the whole corpus.
    """
    fileids = nltk.corpus.treebank_chunk.fileids()[:nr_files]
    gold_tokens = nltk.corpus.treebank_chunk.words(fileids)
    return fix_treebank_tokens(gold_tokens)

"""-------------------Tokeniser Here----------------------"""

def tokenize_corpus(text):
    
    """This function searches through the texts and appends to a list strings 
    that conform to the rules of the regular expression"""
    
    words = re.findall(r"Nov\,|Rep\.|Ms\.|Mr\.|Mrs\.|Dr\.|Inc\.|Ltd\.|Corp\.?|Mass\.|Co\.|Nov\.|Feb\.|Jr\.|Lt\.|Gov\.|" #All awkward ones that were causing problems
                       r"[A-Z]{4,10}|" #Caps words but not to be confused with acronyms
                       r"[a-z]+(?=n\'t)|n\'t|" #n'ts
                       r"(?=\'s|\'re)\'s|\'re|" #'res and 's
                       r"[0-9a-z]+[\/\-][a-z]+\-[a-z]+|" #Double Hyphenes 
                       r"[A-Z]?[0-9a-z]+[\/\-][A-Z]?[a-z0-9]+|" #hypheness
                       r"[0-9]{4}s|" #Dates
                       r"[A-Z]?[a-z]+(?=\-)?|" # Words
                       r"[A-Z]{3}s?|" #3 letter acronyms
                       r"[A-Z]\.[A-Z]\.|" # Acronyms
                       r"[0-9]+\,[0-9]+\,[0-9]+|"
                       r"[\d]+[:\.\,]?[\d]+|" #Simple Numbers
                       
                       r"[\(\)\:\{\}\%\$\.\,\&\-\"\'][\-]?" #Punctuation
                       , text) 

    return words
    
"""-------------------Evaluation Functions----------------"""

def evaluate_tokenization(test_tokens, gold_tokens):
    """Finds the chunks where test_tokens differs from gold_tokens.
    Prints the errors and calculates similarity measures.
    """
    import difflib
    matcher = difflib.SequenceMatcher()
    matcher.set_seqs(test_tokens, gold_tokens)
    error_chunks = true_positives = false_positives = false_negatives = 0
    #print(" Token%30s  |  %-30sToken" % ("Error", "Correct"))
    #print("-" * 38 + "+" + "-" * 38)
    
    for difftype, test_from, test_to, gold_from, gold_to in matcher.get_opcodes():
        if difftype == "equal":
            true_positives += test_to - test_from
        else:
            false_positives += test_to - test_from
            false_negatives += gold_to - gold_from
            error_chunks += 1
            test_chunk = " ".join(test_tokens[test_from:test_to])
            gold_chunk = " ".join(gold_tokens[gold_from:gold_to])
            #print("%6d%30s  |  %-30s%d" % (test_from, test_chunk, gold_chunk, gold_from))
    precision = 1.0 * true_positives / (true_positives + false_positives)
    recall = 1.0 * true_positives / (true_positives + false_negatives)
    fscore = 2.0 * precision * recall / (precision + recall)
    print()
    print("Test size: %5d tokens" % len(test_tokens))
    print("Gold size: %5d tokens" % len(gold_tokens))
    print("Nr errors: %5d chunks" % error_chunks)
    print("Precision: %5.2f %%" % (100 * precision))
    print("Recall:    %5.2f %%" % (100 * recall))
    print("F-score:   %5.2f %%" % (100 * fscore))
    print()

"""-------------------Question_functions------------------"""

#Question 1
def type_token(tokens): 

    """ This function provides the type/token ratio, which is the ratio of duplicate tokens total number of tokens"""
    
    types = []
    for word in tokens:
        if word not in types:
            types.append(word)
        else:
            continue

    total_tokens = len(tokens)
    total_types = len(types)
    lexical_density = round((total_types/total_tokens) *100)

    return total_tokens, total_types, lexical_density #Return the total tokens, types and the lexical density(bonus output)

#Question 2.    
def average_token_length(tokens):
    
    """This function provides the average token length by creating a list of the 
        lengths of all the tokens and then returning the average length"""

    token_length = []
    number_of_tokens = len(tokens)
    for word in tokens:
        token_length.append(len(word))
    return (sum(token_length)/number_of_tokens) #average token length(i think)

 #Question 3.   

#Question 3
def longest_word_length(tokens):

    """Function to find the longest word length and returning all words 
       with this length"""

    token_length = []
    words_with_length = []
    number_of_tokens = len(tokens)
    for word in tokens:
        token_length.append(len(word)) #list of all token lengths
    longest_word_length = max(token_length) #variable that takes the largest integer from token_length

    for word in tokens:
        if len(word) == longest_word_length:
            words_with_length.append(word) #loop that finds all tokens that has the same length as the longest word
    
    return longest_word_length, words_with_length #outpus integer (longest word length) and string(words that have that length)

#Question 4
def number_hapax(tokens):
    
    """Funtion to find number of unique words that only appear once in the test tokens"""
    
    word_freq = {}
    hapax = []
    
    #loop to populate dictionary with word frequencies 
    for word in tokens:
        if word not in word_freq:
            word_freq.update({word : 1})
        else:
            word_freq[word] += 1

    #If the word frequency is one then it is a hapax word, this is then added to a dictionary
    for word, freq in word_freq.items():
        if freq == 1:
            hapax.append(word)

    #Calculation to return the ratio of hapax to total
    num_hapax = len(hapax)
    total_tokens = len(tokens)
    ratio = round((num_hapax/total_tokens) * 100)
    return [num_hapax, total_tokens, ratio] #A list with the number of hapax, total tokens and hapax to token ratio
    
#Question 5.
def most_frequent_words(tokens):
    
    """function calculates the most frequent word in the corpus"""

    word_freq = {}
    top_words = {}
    total_tokens = len(tokens)

    #Dictionary that counts word frequencies 
    for word in tokens:
        if word not in word_freq:
            word_freq.update({word : 1})
        else:
            word_freq[word] += 1

    #loop that generates the top ten frequencies from the word frequency dictionary 
    for w in sorted(word_freq, key=word_freq.get, reverse=True)[:10]: #code taken from: https://stackoverflow.com/a/3177911
            top_words.update({w : word_freq[w]})
    
    ratio = []
    
    #Finall returns the top word and it's average frequency over the entire corpus
    for key, value in top_words.items():
        ratio.append([key, "%" + str(round(value/total_tokens * 100, 2))])
    
    return ratio



    #return top_words, ratio
#Question 6a.
def slicing_corpus(tokens):
    
    """Function cuts the corpus into ten equal slices"""
    
    slic_length = round(len(tokens)/10)
    token_slice = []
    for i in range(10):
        slices = []
        for word in tokens[:slic_length]:
            slices.append(word)
        token_slice.append(slices)
        del tokens[:slic_length]
    
    #Returns a list of lists that contain each equal slice's tokens 
    return token_slice

#Question 6b 
def hapaxes_in_sliced_corpus(corpus_slices):
    
    """Function that generates hapaxes per equal slice of the corpus"""
    
    hapax_per_slice = []
    percent_hapax_per_slice = []
    ratio = []
    slice_num = 1

    #The loop goes through each slice and generates a new list with the number of hapaxes and the average frequency
    for i in corpus_slices:
        hapax_per_slice.append(number_hapax(i)[0])
        percent = ("%" + str(number_hapax(i)[2]))
        percent_hapax_per_slice.append(percent)
        ratio.append([number_hapax(i)[0], percent])
        slice_num += 1

    #Returns a list of lists that contains lists the frequency and average 
    return ratio

#Question6C
def hapax_rolling_slice(tokens):
    
    """This function is taking each equal slice, return the hapaxes in the slice and then 
    incrementing the slice by the next one eventualy have a slice that contains the entire corpus"""
    test_tokens = tokenize_corpus(corpus_text)
    corpus_slices = slicing_corpus(test_tokens)
    hapax_per_incrementing_slice = []
    
    #Counters for corpus slices 
    slice_number = 0
    next_slice = 1
    incrementing_slices = corpus_slices[slice_number]
    hapax_per_incrementing_slice.append(number_hapax(incrementing_slices))

    """This loop goes through each slice, generates hapaxes and averages then increments the slice
       with the next slice"""
    for i in range(len(corpus_slices) - 1):
        slice_number+= 1
        incrementing_slices += corpus_slices[slice_number]
        next_slice += 1
        hapax_per_incrementing_slice.append(number_hapax(incrementing_slices))
    
    #Returns a lit that contains the number of hapaxes and average per slice
    return hapax_per_incrementing_slice
   
#Question 7
def hapax_graph(corpus_slice_list):
    
    """Function that takes data from previous functions and presents in the form of a graph,
       The template for the graph comes from the matplotlib documentation, reference is as follows:
       https://matplotlib.org/gallery/api/two_scales.html"""

    
    data_1 = []
    data_2 = []
    
    #Loop to populate data with the output from question 6
    for i in corpus_slice_list:
        data_1.append(i[0])
        data_2.append(i[2])
    
    t = [1,2,3,4,5,6,7,8,9,10] #Corpora Slice Number
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Slice Number')
    ax1.set_ylabel('Number of Hapaxes', color=color)
    ax1.plot(t, data_1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Percentage of Hapax-Token', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data_2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped"""
    plt.draw() #draws the graph, will be activated at the end of the script

#Question 8
def gen_bigrams(tokens):

    """I created my own bigram generator function because I forgot I could use nltk...."""
    
    test_tokens = tokenize_corpus(corpus_text)
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

#Question 9
def gen_trigrams(tokens):

    """Again I created my own trigram generator, I'm sure there must be an easier way to do it, good practice for me"""

    trigrams = []
    gram_1 = 0
    gram_2 = 1
    gram_3 = 2

    #Loop same as before, creates tuples of trigrams and appends to a list
    for i in range(len(tokens)):
        if gram_1 == len(tokens) - 2:
            break
        else:
            trigrams.append((tokens[gram_1], tokens[gram_2], tokens[gram_3]))
            gram_1 += 1
            gram_2 += 1
            gram_3 += 1
    
    #List of tuples that contains trigrams
    return trigrams

def unique_ngrams(list_of_tuples):

    """This takes a list of tuples and creates a dictionary to record the
    frequency of each ngram"""

    unique_ngrams = {}
    ngram_freq = {}
    ngram_types = {}

    #This loop populates a dictionary with the total ngrams in the token

    for i in list_of_tuples:
        if i not in ngram_freq:
            ngram_freq.update({i : 1})
        else:
            ngram_freq[i] += 1
    total_ngrams = len(ngram_freq.keys())
    
    #Generating dictionary of Unique ngrams
    for key, value in ngram_freq.items():
        if value == 1:
            unique_ngrams.update({key : value})
        else:
            ngram_types.update({key : value})
    
    #calculating total unique ngrams        
    total_unique_ngrams = sum(unique_ngrams.values())
    
    #calculating average
    ratio = total_unique_ngrams / total_ngrams

    #returning tuple of the total unique ngrams and their average frequency in the corpus
    return total_ngrams, total_unique_ngrams, ratio
        
"""-------------------Corpus_Statistics-------------------"""

def corpus_statistics(tokens):
    
    """Finally function that prints all the information to the terminal"""
    
    line_break = ("-" * 38 + "\n")
    
    print(line_break)
    print("Assignment 1: WordNet (deadline: 2018-11-28)\n")
    print("Name 1: Robert")
    print("Name 2: Thomas\n")
    print(line_break)
    print("Part1: Corpus Evaluation\n")
    evaluate_tokenization(tokens, gold_tokens) 
    print(line_break)
    print("Part 2: Corpus Statistics\n") 
    #Question 1
    print("Size of Corpus: " + str(type_token(tokens)[0]))
    print("Total Types: " + str(type_token(tokens)[1]))
    print("Lexical Density: " + "%" + str(type_token(tokens)[2]) + "\n")
    print(line_break)
    #Question 2
    print("Average Word Length: " + str("{:.1f}".format(average_token_length(tokens))))
    #Question 3
    print("Longest Word Length: " + str(longest_word_length(tokens)[0]) + " characters")
    print("Longest Words: " + str(longest_word_length(tokens)[1]) + "\n")
    print(line_break)
    #Question 4
    print("Number of Hapax Words: " + str(number_hapax(tokens)[0]))
    print("Percentage of Hapax Words: " + "%" + str(number_hapax(tokens)[2]) + "\n")
    print(line_break)
    #Question 5
    print("Top Ten Words in Corpus\n\n" + "".join("Word: " + "'" + i[0] + "'" + "\n" + "Percentage: " + i[1] + "\n\n" for i in most_frequent_words(tokens)))
    print(line_break)
    #Question 6
    corpus_slice = slicing_corpus(tokens)
    print("Hapex/Percentage per slice\n\n" + "".join("Number of Hapaxes: " + "'" + str(i[0]) + "'" + "\n" + "Percentage: " + i[1] + "\n\n" for i in hapaxes_in_sliced_corpus(corpus_slice)))
    print(line_break)
    #Question 7
    test_tokens = tokenize_corpus(corpus_text)
    rolling_hapax = hapax_rolling_slice(test_tokens) 
    print("Hapex/Percentage per incrementing slice\n\n" + "".join("Number of Hapaxes per slice: " + "'" + str(i[0]) + "'" + "\n" + "Percentage: " + "%" + str(i[2]) + "\n\n" for i in hapax_rolling_slice(test_tokens)))
    hapax_graph(rolling_hapax)
    print(line_break)
    #Question 8
    bigrams = gen_bigrams(test_tokens)
    print("Number of unique Bigrams: " + str(unique_ngrams(bigrams)[0]))
    print("Percentage of Unique Bigrams over total: " + "%" + str(unique_ngrams(bigrams)[1]))
    #Question 9
    trigrams = gen_trigrams(test_tokens)
    print("Number of unique Trigrams: " + str(unique_ngrams(trigrams)[0]))
    print("Percentage of Unique Trigrams over total: " + "%" + str(unique_ngrams(trigrams)[1]) + "\n")
    print(line_break)
    print("End\n\n")

"""-------------------main_namespace----------------------"""    

if __name__ == '__main__':
    matcher = difflib.SequenceMatcher()
    nr_files = 199
    corpus_text = get_corpus_text(nr_files) #Creating the corpus text 
    gold_tokens = get_gold_tokens(nr_files) #Gold standard for tokens
    test_tokens = tokenize_corpus(corpus_text) #List of tokens created based on regex
    #corpus_statistics(test_tokens) 
    #evaluate_tokenization(test_tokens, gold_tokens)
    #plt.show()
    bigrams = gen_bigrams(test_tokens)
    print(unique_ngrams(bigrams))

    trigrams = gen_trigrams(test_tokens)
    print(unique_ngrams(trigrams))
   
   
    
    