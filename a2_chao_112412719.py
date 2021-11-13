import sys
import gzip
import json #for reading json encoded files into opjects
import re #regular expressions
import numpy as np
import torch
import torch.nn as nn  #pytorch
from collections import Counter
from pprint import pprint

# sys.stdout = open('a2_chao_112412719_OUTPUT.txt', 'w')

# # ## # #
# PART 1 #
# # ## # #

unique_sense_count = 0
unique_sense_map = {}

# PART 1.1
def loadData(filename):
    global unique_sense_count
    file = open(filename, "r", encoding='utf-8')
    data = {}   

    word_counts = {}
    
    for line in file:
        line_id, line_sense, line_context = re.split(r'\t', line)
        word = line_id.split('.')[0]
        line_context = line_context.lower()
        line_context = line_context.split(' ')

        if(line_sense in unique_sense_map):
            line_sense = unique_sense_map[line_sense]
        else:
            unique_sense_map[line_sense] = unique_sense_count
            line_sense = unique_sense_count
            unique_sense_count = unique_sense_count + 1
        
        head_index = -1
        for i in range(0, len(line_context)):
            context_word = re.sub(r'/[^/]*/[^/<]*', '', line_context[i])
            if("<head>" in context_word):
                head_index = i
                context_word = context_word[6:-7]
            line_context[i] = context_word
            if(context_word in word_counts):
                word_counts[context_word] = word_counts[context_word] + 1
            else:
                word_counts[context_word] = 1

        if(word in data):
            data[word] = data[word] + [[line_id, line_sense, line_context, head_index]]
        else:
            data[word] = [[line_id, line_sense, line_context, head_index]]
        
    

    word_counts_popular = sorted(word_counts, key=word_counts.get)
    word_counts_popular.reverse()
    word_counts_dict = {}
    for w in word_counts_popular:
        word_counts_dict[w] = word_counts[w]
    
    return data, word_counts_dict

# PART 1.2
def extractOneHot(context, frequent_words, head_index):
    before_word_hot = [0] * len(frequent_words)
    after_word_hot = [0] * len(frequent_words)
    
    if(head_index > 0):
        before_word = context[head_index-1]
        if(before_word in frequent_words):
            before_word_hot[frequent_words.index(before_word)] = 1
    if(head_index < len(context)-1):
        after_word = context[head_index+1]
        if(after_word in frequent_words):
            after_word_hot[frequent_words.index(after_word)] = 1
    return before_word_hot + after_word_hot

# PART 1.3
def normalizedCrossLoss(ypred, ytrue):
    sum = 0
    for i in range(0, len(ytrue)):
        sum += torch.log(ypred[i][ytrue[i]])
    return -1 / len(ytrue) * sum


class MultiClassLogReg(nn.Module):
    def __init__(self, num_feats, num_classes, learn_rate = 0.01, device = torch.device("cpu") ):
        #DONT EDIT
        super(MultiClassLogReg, self).__init__()
        self.linear = nn.Linear(num_feats+1, num_classes) #add 1 to features for intercept

    def forward(self, X):
        #DONT EDIT
        #This is where the model itself is defined.
        #For logistic regression the model takes in X and returns
        #a probability (a value between 0 and 1)

        newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1) #add intercept
        return self.linear(newX) #logistic function on the linear output



# # ## # #
# PART 2 #
# # ## # #

def getCoOccMatrix(docs, wordToIndex):
    side_length = len(wordToIndex) + 1
    matrix = np.zeros((side_length, side_length))

    for doc in docs:
        for i in range(0, len(doc)):
            word0 = doc[i]
            index0 = len(wordToIndex)
            if(word0 in wordToIndex):
                index0 = wordToIndex[word0]
            for j in range(0, len(doc)):
                if(i != j):
                    word1 = doc[j]
                    index1 = len(wordToIndex)
                    if(word1 in wordToIndex):
                        index1 = wordToIndex[word1]
                    matrix[index0][index1] = matrix[index0][index1]+1
    for i in range(0, len(matrix)):
        matrix[i][i] = matrix[i][i]/2
    return matrix

def standardize(A):
    A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
    return A

def getPCAEmb(matrix, vocab):
    matrix = torch.from_numpy(standardize(matrix))
    U, S, V = torch.svd(matrix)
    U = U.numpy()
    wordEmbeddings = {}
    for i in range(0, len(vocab)):
        wordEmbeddings[vocab[i]] = U[i][:50]
    wordEmbeddings["<OOV>"] = U[len(vocab)][:50]
    return wordEmbeddings

def euclideanDistance(A, B):
    return np.sqrt(np.sum(np.square(A-B)))



# # ## # #
# PART 3 #
# # ## # #
def getEmbeddingFeature(context, head_index, embedding):
    two_before_feature = np.zeros((50))
    one_before_feature = np.zeros((50))
    one_after_feature = np.zeros((50))
    two_after_feature = np.zeros((50))
    if(head_index == 0):
        one_after_word = context[head_index+1] if context[head_index+1] in embedding else "<OOV>"
        two_after_word = context[head_index+2] if context[head_index+2] in embedding else "<OOV>"
        one_after_feature = embedding[one_after_word]
        two_after_feature = embedding[two_after_word]
    elif(head_index == 1):
        one_before_word = context[head_index-1] if context[head_index-1] in embedding else "<OOV>"
        one_after_word = context[head_index+1] if context[head_index+1] in embedding else "<OOV>"
        two_after_word = context[head_index+2] if context[head_index+2] in embedding else "<OOV>"
        one_before_feature = embedding[one_before_word]
        one_after_feature = embedding[one_after_word]
        two_after_feature = embedding[two_after_word]
    elif(head_index == len(context)-1):
        two_before_word = context[head_index-2] if context[head_index-2] in embedding else "<OOV>"
        one_before_word = context[head_index-1] if context[head_index-1] in embedding else "<OOV>"
        two_before_feature = embedding[two_before_word]
        one_before_feature = embedding[one_before_word]
    elif(head_index == len(context)-2):
        two_before_word = context[head_index-2] if context[head_index-2] in embedding else "<OOV>"
        one_before_word = context[head_index-1] if context[head_index-1] in embedding else "<OOV>"
        one_after_word = context[head_index+1] if context[head_index+1] in embedding else "<OOV>"
        two_before_feature = embedding[two_before_word]
        one_before_feature = embedding[one_before_word]
        one_after_feature = embedding[one_after_word]
    else:
        two_before_word = context[head_index-2] if context[head_index-2] in embedding else "<OOV>"
        one_before_word = context[head_index-1] if context[head_index-1] in embedding else "<OOV>"
        one_after_word = context[head_index+1] if context[head_index+1] in embedding else "<OOV>"
        two_after_word = context[head_index+2] if context[head_index+2] in embedding else "<OOV>"
        two_before_feature = embedding[two_before_word]
        one_before_feature = embedding[one_before_word]
        one_after_feature = embedding[one_after_word]
        two_after_feature = embedding[two_after_word]
    final = []
    final = np.append(final, two_before_feature)
    final = np.append(final, one_before_feature)
    final = np.append(final, one_after_feature)
    final = np.append(final, two_after_feature)
    return(final)


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("USAGE: python3 a1_lastname_id.py onesec_train.tsv onesec_test.tsv")
        sys.exit(1)
    filename_train = sys.argv[1]
    filename_test = sys.argv[2]

    train_data, train_word_counts = loadData(filename_train)
    frequent_words = list(train_word_counts)[0:2000]
    
    test_data, test_word_counts = loadData(filename_test)

    # # ## # #
    # PART 1 #
    # # ## # #
    print("")
    print("[TESTING UNIGRAM WSD MODELS]")
    for word in train_data:

        #extract features:
        train_onehots = []
        test_onehots = []
        for d in train_data[word]:
            train_onehots.append(extractOneHot(d[2], frequent_words, d[3]))
        for d in test_data[word]:
            test_onehots.append(extractOneHot(d[2], frequent_words, d[3]))
        

        train_x = torch.from_numpy(np.array(train_onehots))
        train_x = train_x.type(torch.FloatTensor)
        train_y = torch.from_numpy(np.array([d[1] for d in train_data[word]]))
        min_train, max_train = torch.min(train_y), torch.max(train_y)
        train_y = torch.sub(train_y, min_train)
        train_y = train_y.type(torch.LongTensor)


        test_x = torch.from_numpy(np.array(test_onehots))
        test_x = test_x.type(torch.FloatTensor)
        test_y = torch.from_numpy(np.array([d[1] for d in test_data[word]]))
        test_y = torch.sub(test_y, min_train)
        test_y = test_y.type(torch.LongTensor)

        #Model setup:
        learning_rate, epochs = 1.0, 300
        model = MultiClassLogReg(len(frequent_words)*2, int(max_train - min_train + 1))
        sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
        lossfunc = nn.CrossEntropyLoss()

        # # training loop:
        for i in range(epochs):
            model.train()
            sgd.zero_grad()
            #forward pass:
            ypred = model(train_x)
            loss = lossfunc(ypred, train_y)
            #backward:
            loss.backward()
            sgd.step()

        with torch.no_grad():
            ytestpred_prob = model(test_x).numpy()
            ytestpred_class = [np.where(prob == max(prob))[0][0] for prob in ytestpred_prob]
            print(word)
            if(word=="process"):
                for i in range(0, len(test_data["process"])):
                    if(test_data["process"][i][0] == "process.NOUN.000018"):
                        print("  predictions for process.NOUN.000018: ",  ytestpred_prob[i])
                    if(test_data["process"][i][0] == "process.NOUN.000024"):
                        print("  predictions for process.NOUN.000024: ",  ytestpred_prob[i])
                        break
            if(word=="machine"):
                for i in range(0, len(test_data["machine"])):
                    if(test_data["machine"][i][0] == "machine.NOUN.000004"):
                        print("  predictions for machine.NOUN.000004: ",  ytestpred_prob[i])
                    if(test_data["machine"][i][0] == "machine.NOUN.000008"):
                        print("  predictions for machine.NOUN.000008: ",  ytestpred_prob[i])
                        break
            if(word=="language"):
                for i in range(0, len(test_data["language"])):
                    if(test_data["language"][i][0] == "language.NOUN.000008"):
                        print("  predictions for language.NOUN.000008: ",  ytestpred_prob[i])
                    if(test_data["language"][i][0] == "language.NOUN.000014"):
                        print("  predictions for language.NOUN.000014: ",  ytestpred_prob[i])
                        break
            count = 0
            for i in range(0, len(ytestpred_class)):
                if(ytestpred_class[i] == test_y[i]):
                    count += 1
            print("  correct: ", count, " out of ", len(test_y))

    # # ## # #
    # PART 2 #
    # # ## # #
    combined_train_contexts = []
    wordToIndex = {}

    for word in train_data:
        for d in train_data[word]:
            combined_train_contexts.append(d[2])

    for i in range(0, len(frequent_words)):
        wordToIndex[frequent_words[i]] = i

    matrix = getCoOccMatrix(combined_train_contexts, wordToIndex)

    wordEmbeddings = getPCAEmb(matrix, frequent_words)

    print("")
    print("[DISTANCE BT WORDS]")
    print("('language', 'process'): ", euclideanDistance(wordEmbeddings['language'], wordEmbeddings['process']))
    print("('machine', 'process'): ", euclideanDistance(wordEmbeddings['machine'], wordEmbeddings['process']))
    print("('language', 'speak'): ", euclideanDistance(wordEmbeddings['language'], wordEmbeddings['speak']))
    print("('word', 'words'): ", euclideanDistance(wordEmbeddings['word'], wordEmbeddings['words']))
    print("('word', 'the'): ", euclideanDistance(wordEmbeddings['word'], wordEmbeddings['the']))

    # # ## # #
    # PART 3 #
    # # ## # #
    print("")
    print("[TESTING WSD WITH EMBEDDINGS]")
    for word in train_data:

        #extract features:
        train_embeddings = []
        test_embeddings = []
        for d in train_data[word]:
            train_embeddings.append(getEmbeddingFeature(d[2], d[3], wordEmbeddings))
        for d in test_data[word]:
            test_embeddings.append(getEmbeddingFeature(d[2], d[3], wordEmbeddings))
     
        train_x = torch.from_numpy(np.array(train_embeddings))
        train_x = train_x.type(torch.FloatTensor)
        train_y = torch.from_numpy(np.array([d[1] for d in train_data[word]]))
        min_train, max_train = torch.min(train_y), torch.max(train_y)
        train_y = torch.sub(train_y, min_train)
        train_y = train_y.type(torch.LongTensor)


        test_x = torch.from_numpy(np.array(test_embeddings))
        test_x = test_x.type(torch.FloatTensor)
        test_y = torch.from_numpy(np.array([d[1] for d in test_data[word]]))
        test_y = torch.sub(test_y, min_train)
        test_y = test_y.type(torch.LongTensor)

        #Model setup:
        learning_rate, epochs = 1.0, 300
        model = MultiClassLogReg(len(train_embeddings[0]), int(max_train - min_train + 1))
        sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
        lossfunc = nn.CrossEntropyLoss()

        # # training loop:
        for i in range(epochs):
            model.train()
            sgd.zero_grad()
            #forward pass:
            ypred = model(train_x)
            loss = lossfunc(ypred, train_y)
            #backward:
            loss.backward()
            sgd.step()

        with torch.no_grad():
            ytestpred_prob = model(test_x).numpy()
            ytestpred_class = [np.where(prob == max(prob))[0][0] for prob in ytestpred_prob]
            print(word)
            if(word=="process"):
                for i in range(0, len(test_data["process"])):
                    if(test_data["process"][i][0] == "process.NOUN.000018"):
                        print("  predictions for process.NOUN.000018: ",  ytestpred_prob[i])
                    if(test_data["process"][i][0] == "process.NOUN.000024"):
                        print("  predictions for process.NOUN.000024: ",  ytestpred_prob[i])
                        break
            if(word=="machine"):
                for i in range(0, len(test_data["machine"])):
                    if(test_data["machine"][i][0] == "machine.NOUN.000004"):
                        print("  predictions for machine.NOUN.000004: ",  ytestpred_prob[i])
                    if(test_data["machine"][i][0] == "machine.NOUN.000008"):
                        print("  predictions for machine.NOUN.000008: ",  ytestpred_prob[i])
                        break
            if(word=="language"):
                for i in range(0, len(test_data["language"])):
                    if(test_data["language"][i][0] == "language.NOUN.000008"):
                        print("  predictions for language.NOUN.000008: ",  ytestpred_prob[i])
                    if(test_data["language"][i][0] == "language.NOUN.000014"):
                        print("  predictions for language.NOUN.000014: ",  ytestpred_prob[i])
                        break
            count = 0
            for i in range(0, len(ytestpred_class)):
                if(ytestpred_class[i] == test_y[i]):
                    count += 1
            print("  correct: ", count, " out of ", len(test_y))