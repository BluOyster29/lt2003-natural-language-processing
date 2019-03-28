
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
    total_ngrams = sum(ngram_freq.values())
    
    #Generating dictionary of Unique ngrams
    for key, value in ngram_freq.items():
        if value == 1:
            unique_ngrams.update({key : value})
        else:
            ngram_types.update({key : value})
    
    #calculating total unique ngrams        
    total_unique_ngrams = sum(unique_ngrams.values())
    
    #calculating average
    ratio = round(100 * total_unique_ngrams / total_ngrams)

    #returning tuple of the total unique ngrams and their average frequency in the corpus
    return  total_unique_ngrams, ratio