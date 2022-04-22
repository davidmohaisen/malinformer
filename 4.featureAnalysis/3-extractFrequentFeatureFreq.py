import os, time, pickle, re, sys, logging
import multiprocessing as mp
from sklearn.feature_extraction.text import CountVectorizer

# mpl = mp.log_to_stderr()
# mpl.setLevel(logging.INFO)

def freqFeatureOccurrenceByColumnIndex(transformedpath, fam, freqFeatIndices, outPath):
    freqs = pickle.load(open(transformedpath + fam + '-frequency0'+'.pickle', 'rb'))[:,freqFeatIndices]
    
    with open(outPath+fam+"-frequenFeatureOccurrences.pickle", 'wb') as handle:
        pickle.dump(freqs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del freqs


def main():
    startTime = time.time()
    families = pickle.load(open('../../data/trainFamily_MalwareList.pickle', 'rb')).keys()
    transformedpath = '../../data/perFamilyTransformation/'
    frequentFeatPath = '../../data/featureAnalysis/frequentFeatsPerFamily/'

    outPath = '../../data/featureAnalysis/frequentFeatOccurrence/'
    if not os.path.isdir(outPath):
        os.mkdir(outPath)
        
    for fam in families:
        freqFeats = pickle.load(open(frequentFeatPath+str(fam)+"-featName_featIndex.pickle", 'rb'))
        freqFeatIndices = sorted(freqFeats.values())
        freqFeatureOccurrenceByColumnIndex(transformedpath, fam, freqFeatIndices, outPath)
        print("Completed for ", fam, "Time taken: ", time.time() - startTime)

    
    

    


if __name__=="__main__":
    main()
