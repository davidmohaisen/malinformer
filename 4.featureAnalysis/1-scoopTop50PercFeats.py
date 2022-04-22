import os, time, pickle
import numpy as np

# freq of family vocab ::: countvectorizer, vocab = fam_vocab
# <50% -- remove feats
# mututal info (fam1 vs fam2 ... fam10)
def sumMatrixByColumn(transformedpath, fam):
    freqs = pickle.load(open(transformedpath + fam + '-frequency0'+'.pickle', 'rb'))
    freqs[freqs > 0] = 1
    freqs[freqs < 0] = 0
    sumFreqs = freqs.sum(axis=0)
    del freqs
    return sumFreqs
    
def InverserFreqAnalyzer(transformedpath, fam, families, famFeatName_featIndex):
    featNames = set(famFeatName_featIndex.keys())
    invFam_freq = {}
    for invFam in families:
        if invFam == fam:
            continue
        invFeats_index = pickle.load(open("../../data/modularizedVocab/" + invFam + '-vocab.pickle', 'rb'))
        # index_feats = dict((v,k) for k,v in feats_index.items())
        intersectingFeats = set(invFeats_index.keys()).intersection(featNames)
        
        sumInvFreqs = sumMatrixByColumn(transformedpath, invFam)
        for invFeats in intersectingFeats:
            invFr = sumInvFreqs[invFeats_index[invFeats]]
            if invFeats not in invFam_freq:
                invFam_freq[invFeats] = invFr
            else:
                invFam_freq[invFeats] = max(invFam_freq[invFeats], invFr)
    
    return invFam_freq

def main():
    families = pickle.load(open('../../data/trainFamily_MalwareList.pickle', 'rb')).keys()
    transformedpath = '../../data/perFamilyTransformation/'
    outPath = '../../data/featureAnalysis/'
    if not os.path.isdir(outPath):
        os.mkdir(outPath)
    outPath = '../../data/featureAnalysis/frequentFeatsPerFamily/'
    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    inverAnalysisOutPath = '../../data/featureAnalysis/inverseFreqAnalysis/'
    if not os.path.isdir(inverAnalysisOutPath):
        os.mkdir(inverAnalysisOutPath)

    startTime = time.time()

    fam_SamplePresence = {}
    fam_InvSamplePresence = {}

    for fam in families:
        sumFreqs = sumMatrixByColumn(transformedpath, fam)#freqs.sum(axis=0)
        # fam_SamplePresence[fam] = sumFreqs

        feats_index = pickle.load(open("../../data/modularizedVocab/" + fam + '-vocab.pickle', 'rb'))
        index_feats = dict((v,k) for k,v in feats_index.items())
        del feats_index
        # print(sumFreqs)
        # print(fam, freqs.shape, len(feats), sumFreqs.shape) 
        # print(np.where(sumFreqs > 49))
        # print(type(feats), type(np.where(sumFreqs > 49)[0]), type(sumFreqs))
        # reducedFeats = feats[np.where(sumFreqs > 49)[0]]
        indices = np.where(sumFreqs > 49)[0]
        famFeatName_featIndex = {}#[{index_feats[idx]: idx} for idx in indices]
        feat_freq = {}
        for idx in indices:
            famFeatName_featIndex[index_feats[idx]] = idx
            feat_freq[index_feats[idx]] = sumFreqs[idx]

        with open(outPath+str(fam)+"-featName_featIndex.pickle", 'wb') as handle:
            pickle.dump(famFeatName_featIndex, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("Family: ", fam, ". Length of sumFreq array: ", len(sumFreqs),
              ". Length of reduced features array", len(famFeatName_featIndex))
        print("Time taken: ", time.time() - startTime)

        invFeat_freq = InverserFreqAnalyzer(transformedpath, fam, families, famFeatName_featIndex)
        with open(inverAnalysisOutPath+fam+"-FeatFrequency.pickle", 'wb') as handle:
            pickle.dump(feat_freq, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(inverAnalysisOutPath+fam+"-FeatInverseFrequency.pickle", 'wb') as handle:
            pickle.dump(invFeat_freq, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fam_SamplePresence[fam] = feat_freq
        fam_InvSamplePresence[fam] = invFeat_freq
        del feat_freq
        del invFeat_freq

    with open(inverAnalysisOutPath+"familyFeat-Frequency.pickle", 'wb') as handle:
        pickle.dump(fam_SamplePresence, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(inverAnalysisOutPath+"familyFeat-InverseFrequency.pickle", 'wb') as handle:
        pickle.dump(fam_InvSamplePresence, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(inverAnalysisOutPath+"familyFeat-Frequency_InverseFrequency.pickle", 'wb') as handle:
        pickle.dump([fam_SamplePresence, fam_InvSamplePresence], handle, protocol=pickle.HIGHEST_PROTOCOL)
    



        # sum it and see if it is < 50. If yes, remove


if __name__=="__main__":
    main()
