import os, time, pickle, re, sys, logging
import multiprocessing as mp
from sklearn.feature_extraction.text import CountVectorizer
from itertools import islice

# mpl = mp.log_to_stderr()
# mpl.setLevel(logging.INFO)

def logCleaner(picklePath):
    f_str = pickle.load(open(picklePath, "rb")).replace("\n", "")
    return f_str

def main():
    fstPickleDir = "../../data/f_str-Pickles/"
    outPath = '../../data/modularizedVocab/'
    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    disasms = set(os.listdir(fstPickleDir))
    disasms = set([itx.replace(".pickle", "") for itx in list(disasms)])
    fam_malw = pickle.load(open('../../data/trainFamily_MalwareList.pickle', 'rb'))

    startTime = time.time()

    overallCorpus = []
    for fam in fam_malw:
        allLogs = fam_malw[fam]
        print(len(allLogs), "Malws in family: ", fam)

        Arguments = []
        for fp in allLogs:
            if os.path.isfile(fstPickleDir + fp + ".log.pickle.pickle"):
                Arguments.append(fstPickleDir + fp + ".log.pickle.pickle")
            elif os.path.isfile(fstPickleDir + "VirusShare_" + fp + ".log.pickle.pickle"):
                Arguments.append(fstPickleDir + "VirusShare_" + fp + ".log.pickle.pickle")
        print(fam, len(Arguments))
        p = mp.Pool(processes=60)
        corpus = p.map(logCleaner, Arguments)
        p.close()
        p.join()

        print("Corpus length: ", len(corpus))
        print("ngram Count for ", [len(itm.rsplit("/")) for itm in corpus])

        vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), lowercase=True, ngram_range=(5,20), min_df=2)
        vectorizer.fit(corpus)
        vocab = vectorizer.vocabulary_

        print("Length of vocab: ", len(vocab))

        with open(outPath+str(fam)+"-vocab.pickle", 'wb') as handle:
            pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Time taken is ", time.time() - startTime)

        if len(overallCorpus) == 0:
            overallCorpus = corpus
        else:
            overallCorpus += corpus
        
        try:
            del corpus
            del vocab
            del vectorizer
            del Arguments
        except:
            pass

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), lowercase=True, ngram_range=(5,20), min_df=2)
    vectorizer.fit(overallCorpus)
    overallVocab = vectorizer.vocabulary_
    with open('../../data/overallVocab.pickle', 'wb') as handle:
        pickle.dump(overallVocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Number of malware and size of vocab ", len(overallCorpus), len(overallVocab))


if __name__=="__main__":
    main()