import os, time, pickle, re, sys, logging
import multiprocessing as mp
from sklearn.feature_extraction.text import CountVectorizer

mpl = mp.log_to_stderr()
mpl.setLevel(logging.INFO)

def logCleaner(pickleFile):
    f_str = pickle.load(open(pickleFile, "rb")).replace("\n", "").replace("\n", "")
    return f_str


def main():
    startTime = time.time()

    fstrDir = "../../data/f_str-Pickles/"
    outPath = '../../data/perFamilyTransformation/'
    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    disasms = set(os.listdir(fstrDir))
    disasms = set([itx.replace(".pickle", "") for itx in list(disasms)])
    fam_malw = pickle.load(open('../../data/trainFamily_MalwareList.pickle', 'rb'))
    
    for fam in fam_malw:
        allLogs = fam_malw[fam]
        print(len(allLogs), "Malws in family: ", fam)

        Arguments = []
        for fp in allLogs:
            if os.path.isfile(fstrDir + fp + ".log.pickle.pickle"):
                Arguments.append(fstrDir +fp+".log.pickle.pickle")
            elif os.path.isfile(fstrDir + "VirusShare_"+fp+".log.pickle.pickle"):
                Arguments.append(fstrDir + "VirusShare_"+fp+".log.pickle.pickle")
        print(fam, len(Arguments))
        p = mp.Pool(processes=60)
        corpus = p.map(logCleaner, Arguments)
        p.close()
        p.join()
        vocab = pickle.load(open("../../data/modularizedVocab/" + fam + '-vocab.pickle', 'rb'))
        print("Corpus length: ", len(corpus), "::: Length of vocab", len(vocab))

        Arguments = []
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), lowercase=True, ngram_range=(5,20), vocabulary=vocab)
        for i in range(0, len(corpus), 100):
            Fitvectorizer = vectorizer.transform(corpus[i:i+100])
            Fitvectorizer = Fitvectorizer.toarray()
            with open(outPath + fam + '-frequency'+str(i)+'.pickle', 'wb') as handle:
                pickle.dump(Fitvectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            Fitvectorizer = []
            del Fitvectorizer
            print("Cycle ", i, "completed for ", fam, " family.")


if __name__=="__main__":
    main()
