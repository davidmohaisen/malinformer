import os, time, pickle, re, sys, logging
import multiprocessing as mp
from sklearn.feature_extraction.text import CountVectorizer

# mpl = mp.log_to_stderr()
# mpl.setLevel(logging.INFO)

def logCleaner(pickleFile):
    f_str = pickle.load(open(pickleFile, "rb")).replace("\n", "").replace("\n", "")
    return f_str

def TestRunner(argsTest):
    [Testdir, outPath, famPre, famInner, vocab, startTime, Testfam_malw] = argsTest
    allLogs = Testfam_malw[famInner]
    print(len(allLogs), "Malws in family: ", famInner)

    # Arguments = []
    corpus = []
    for fp in allLogs:
        if os.path.isfile(Testdir + "cyberIoCs/" + fp + ".log.pickle"):
            # Arguments.append(Testdir + "cyberIoCs/" + fp + ".log.pickle")
            corpus.append(logCleaner(Testdir + "cyberIoCs/" + fp + ".log.pickle"))
        elif os.path.isfile(Testdir + "VirusShare/" + "VirusShare_" + fp + ".log.pickle"):
            # Arguments.append(Testdir + "VirusShare/" + "VirusShare_" + fp + ".log.pickle")
            corpus.append(logCleaner(Testdir + "VirusShare/" + "VirusShare_" + fp + ".log.pickle"))
    # print(famInner, len(Arguments))
    # if len(Arguments) >= 55:
    #     p = mp.Pool(processes=55)
    # else:
    #     p = mp.Pool(processes=len(Arguments))
    # corpus = p.map(logCleaner, Arguments)
    # p.close()
    # p.join()
    print("Corpus length: ", len(corpus), "::: Length of vocab", len(vocab))

    # Arguments = []

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), lowercase=True, ngram_range=(5, 20),
                                 vocabulary=vocab)

    Fitvectorizer = vectorizer.transform(corpus)
    Fitvectorizer = Fitvectorizer.toarray()
    with open(outPath + famPre + "-" + famInner + '-frequency-Test.pickle', 'wb') as handle:
        pickle.dump(Fitvectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    Fitvectorizer = []
    del Fitvectorizer
    print("Cycle completed for ", famPre + "-" + famInner, " family.")

    features = vectorizer.get_feature_names()
    with open(outPath + famPre + "-" + famInner + '-features-Test.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Time take for family (--TEST--)", famPre + "-" + famInner, "is: ", time.time() - startTime)

def TrainRunner(argsTrain):
    [Traindir, outPath, famPre, famInner, vocab, startTime, Trainfam_malw] = argsTrain
    allLogs = Trainfam_malw[famInner]
    print(len(allLogs), "Malws in family: ", famInner)

    # Arguments = []
    corpus = []
    for fp in allLogs:
        if os.path.isfile(Traindir + fp + ".log.pickle"):
            # Arguments.append(Traindir + fp + ".log.pickle")
            corpus.append(logCleaner(Traindir + fp + ".log.pickle"))
        elif os.path.isfile(Traindir + "VirusShare_" + fp + ".log.pickle"):
            # Arguments.append(Traindir + "VirusShare_" + fp + ".log.pickle")
            corpus.append(logCleaner(Traindir + "VirusShare_" + fp + ".log.pickle"))
    # print(famInner, len(Arguments))
    # if len(Arguments) >= 55:
    #     p = mp.Pool(processes=55)
    # else:
    #     p = mp.Pool(processes=len(Arguments))
    # corpus = p.map(logCleaner, Arguments)
    # p.close()
    # p.join()

    print("Corpus length: ", len(corpus), "::: Length of vocab", len(vocab))

    # Arguments = []

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), lowercase=True, ngram_range=(5, 20),
                                 vocabulary=vocab)

    Fitvectorizer = vectorizer.transform(corpus)
    Fitvectorizer = Fitvectorizer.toarray()
    with open(outPath + famPre + "-" + famInner + '-frequency-Train.pickle', 'wb') as handle:
        pickle.dump(Fitvectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    Fitvectorizer = []
    del Fitvectorizer
    print("Cycle completed for ", famPre + "-" + famInner, " family.")

    features = vectorizer.get_feature_names()
    with open(outPath + famPre + "-" + famInner + '-features-Train.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Time take for family (--TEST--)", famPre + "-" + famInner, "is: ", time.time() - startTime)



def main():
    Testdir = "../../../data/testDataset/f_str-Pickles/"
    Traindir = "../../../data/f_str-Pickles/"
    topFeatDir = "../../../data/featureAnalysis/topNfeats/withoutThreshold_rest/"
    outPath = "../../../data/byFamilyTopFeatOccurrence_test-train/"

    # disasms = set(os.listdir(dir))
    # disasms = set([itx.replace(".pickle", "") for itx in list(disasms)])
    Testfam_malw = pickle.load(open('../../../data/testDataset/testFamily_MalwareList.pickle', 'rb'))
    Trainfam_malw = pickle.load(open('../../../data/trainFamily_MalwareList.pickle', 'rb'))

    fam_cnt = {"dofloo": 20000, "elknot": 20001, "gafgyt": 20000, "ganiw": 20001, "generica": 20000, "local": 20000, "mirai": 20000, "setag": 20000, "tsunami": 20000, "xorddos": 20000}

    startTime = time.time()

    ArgumentsTrain, ArgumentsTest = [], []
    for famPre in Testfam_malw:
        cnt = fam_cnt[famPre]
        vocab = pickle.load(open(topFeatDir+famPre+"-"+str(cnt)+"-finalConsideredFeatures-laters.pickle", "rb"))

        for famInner in Testfam_malw:
            # TrainRunner(Traindir, outPath, famPre, famInner, vocab, startTime, Trainfam_malw)
            ArgumentsTrain.append([Traindir, outPath, famPre, famInner, vocab, startTime, Trainfam_malw])
            # TestRunner(Testdir, outPath, famPre, famInner, vocab, startTime, Testfam_malw)
            ArgumentsTest.append([Testdir, outPath, famPre, famInner, vocab, startTime, Testfam_malw])


    print("Train: Length of Arguments: ", len(ArgumentsTrain))
    p = mp.Pool(processes=len(ArgumentsTrain))
    corpus = p.map(TrainRunner, ArgumentsTrain)
    p.close()
    p.join()

    print("Training Complete")
    print("Test: Length of Arguments: ", len(ArgumentsTest))

    p = mp.Pool(processes=len(ArgumentsTest))
    corpus = p.map(TestRunner, ArgumentsTest)
    p.close()
    p.join()

    print("Testing Complete")




if __name__=="__main__":
    main()
