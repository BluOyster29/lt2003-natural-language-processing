def testing_model():
    
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
            
            num = 0
            for sentences in test_file_bigrams:
                probs = []
                for ngram in sentences:
                    if ngram in final_test_model[1]:
                        probs.append(math.log(final_test_model[1][ngram]/sum(final_test_model[1].values()),2))
                    else:
                        probs.append(math.log(1/sum(final_test_model[1].values()),2))
                num += 1
                list_of_probs = ("sentence " + str(num), lang_name[x], [lang_name[i],-sum(probs)])
                print(list_of_probs)