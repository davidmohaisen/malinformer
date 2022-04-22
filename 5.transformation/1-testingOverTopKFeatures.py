import os, time, pickle, re, sys, logging
import multiprocessing as mp
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# mpl = mp.log_to_stderr()
# mpl.setLevel(logging.INFO)

def logCleaner(pickleFile):
    f_str = pickle.load(open(pickleFile, "rb")).replace("\n", "").replace("\n", "")
    return f_str

def sklearnFitting(args):
    [corpus, outPath, fam, startTime, k ,vocab, type] = args
    print("Family: ", fam, "::: K = ", k, "Corpus length: ", len(corpus), "::: Length of vocab", len(vocab))

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), lowercase=True, ngram_range=(5, 20), vocabulary=vocab)
    Fitvectorizer = vectorizer.transform(corpus)
    Fitvectorizer = Fitvectorizer.toarray()
    with open(outPath + fam + '-' + str(k) + '-frequency-'+type+'.pickle', 'wb') as handle:
        pickle.dump(Fitvectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    Fitvectorizer = []
    del Fitvectorizer
    # print("Cycle ", k, "completed for ", fam, " family.")

    features = vectorizer.vocabulary_
    with open(outPath + fam + '-' + str(k) + '-features-'+type+'.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Time taken for family ", fam, "at K = ", k,  "is: ", time.time() - startTime)

def famCorpusCreator(fam_malw, f_strDir):
    fam_corpus = {}
    for family in fam_malw:
        allLogs = fam_malw[family]
        print(len(allLogs), "Malws in family: ", family)
        corpus = []
        for fp in allLogs:
            if os.path.isfile(f_strDir + fp + ".log.pickle.pickle"):
                corpus.append(logCleaner(f_strDir + fp + ".log.pickle.pickle"))
            elif os.path.isfile(f_strDir + "VirusShare_" + fp + ".log.pickle.pickle"):
                corpus.append(logCleaner(f_strDir + "VirusShare_" + fp + ".log.pickle.pickle"))
        fam_corpus[family] = corpus
    return fam_corpus

def ArgumentsCreator(families, topFeatDir, fam_cnt, testFam_corpus, trainFam_corpus, outPath, startTime):
    Arguments = []
    for k in range(500, 20500, 500):
        vocab = np.asarray([])
        for family in families:
            vocab_pre = np.asarray(list(pickle.load(
                open(topFeatDir + family + "-" + str(fam_cnt[family]) + "-finalConsideredFeatures.pickle",
                     "rb"))[:k]))
            vocab = np.append(vocab, vocab_pre)
        print("Vocab data structure: ", type(vocab))
        vocab = np.unique(vocab)
        for fam in families:
            corpus = testFam_corpus[fam]
            Arguments.append([corpus, outPath, fam, startTime, k, vocab, "Test"])
            corpus = trainFam_corpus[fam]
            Arguments.append([corpus, outPath, fam, startTime, k, vocab, "Train"])
    return Arguments

def main():
    testF_strDir = "../../data/testDataset/f_str-Pickles/"
    trainF_strDir = "../../data/f_str-Pickles/"
    outPath = "../../data/testDataset/transformedOnTopKFeats/"
    topFeatDir = "../../data/featureAnalysis/topNfeats-R2/"

    testFam_malw = pickle.load(open('../../data/testFamily_MalwareList.pickle', 'rb'))
    trainFam_malw = pickle.load(open('../../data/trainFamily_MalwareList.pickle', 'rb'))

    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    startTime = time.time()

    fam_cnt = {"dofloo": 20500, "elknot": 20500, "gafgyt": 20500, "ganiw": 20500, "generica": 20500, "local": 20500, "mirai": 20500, "setag": 20500, "tsunami": 20500, "xorddos": 20500}
    families = ["dofloo", "elknot", "gafgyt", "ganiw", "generica", "local", "mirai", "setag", "tsunami", "xorddos"]

    testFam_corpus = famCorpusCreator(testFam_malw, testF_strDir)
    trainFam_corpus = famCorpusCreator(trainFam_malw, trainF_strDir)

    Arguments = ArgumentsCreator(families, topFeatDir, fam_cnt, testFam_corpus, trainFam_corpus, outPath, startTime)

    print("Length of Arguments is: ", len(Arguments))
    p = mp.Pool(processes=58)
    corpus = p.map(sklearnFitting, Arguments)
    p.close()
    p.join()



if __name__=="__main__":
    main()
