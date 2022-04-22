import os, time, pickle, re, sys, logging
import multiprocessing as mp
from sklearn.feature_extraction.text import CountVectorizer

# mpl = mp.log_to_stderr()
# mpl.setLevel(logging.INFO)

def logCleaner(pickleFile):
    f_str = pickle.load(open(pickleFile, "rb")).replace("\n", "").replace("\n", "")
    return f_str


def main():
    f_strDir = "../data/dynamic/f_str-Pickles/"
    outPath = "../data/dynamic/transFormedOverTrainingVocab/"
    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    disasms = set(os.listdir(f_strDir))
    disasms = set([itx.replace(".pickle", "") for itx in list(disasms)])
    fam_malw = pickle.load(open('../data/fam_malw_MalwareList.pickle', 'rb'))

    startTime = time.time()

    vocab = pickle.load(open("../../data/overallNgramOccurrences/elknot-features.pickle", "rb"))
    print("Vocab data structure: ", type(vocab))

    for fam in fam_malw:
        allLogs = fam_malw[fam]
        print(len(allLogs), "Malws in family: ", fam)


        # Arguments = []
        corpus = []
        for fp in allLogs:
            if os.path.isfile(f_strDir+fp+".log.pickle"):
                # Arguments.append(f_strDir+fp+".log.pickle")
                corpus.append(logCleaner(f_strDir+fp+".log.pickle"))
            elif os.path.isfile(f_strDir+"VirusShare_"+fp+".log.pickle"):
                # Arguments.append(f_strDir+"VirusShare_"+fp+".log.pickle")
                corpus.append(logCleaner(f_strDir+"VirusShare_"+fp+".log.pickle"))
        # print(fam, len(Arguments))
        # p = mp.Pool(processes=45)
        # corpus = p.map(logCleaner, Arguments)
        # p.close()
        # p.join()
        print("Corpus length: ", len(corpus), "::: Length of vocab", len(vocab))

        # with open(outPath + fam + '-malwareOrder.pickle', 'wb') as handle:
        #     pickle.dump(Arguments, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Arguments = []


        vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), lowercase=True, ngram_range=(5,20), vocabulary=vocab)
        for i in range(0, len(corpus), 100):
            Fitvectorizer = vectorizer.transform(corpus[i:i+100])
            Fitvectorizer = Fitvectorizer.toarray()
            with open(outPath + fam + '-frequency'+str(i)+'.pickle', 'wb') as handle:
                pickle.dump(Fitvectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            Fitvectorizer = []
            del Fitvectorizer
            print("Cycle ", i, "completed for ", fam, " family.")

        features = vectorizer.get_feature_names()
        with open(outPath + fam + '-features.pickle', 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Time take for family ", fam, "is: ", time.time() - startTime)


if __name__=="__main__":
    main()
