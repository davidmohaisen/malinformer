import os, pickle, json, time
import pandas as pd
import numpy as np
import  logging
import multiprocessing as mp

mpl = mp.log_to_stderr()
mpl.setLevel(logging.INFO)

def indicesByFIF(uniqueFIFs, fIF_perSample, FeatFreqSum):
    uniqueFIFs_sortedIndiceList = {}
    for i in range(len(uniqueFIFs)):
        certainFIFindices = np.where(fIF_perSample == uniqueFIFs[i])[0]
        featFreqSumAtCertainFIF = FeatFreqSum[certainFIFindices]

        # sorting by frequency sum
        tmp = featFreqSumAtCertainFIF.argsort()
        certainFIFindices = certainFIFindices[tmp]
        if uniqueFIFs[i] > 1:
            if 1 in uniqueFIFs_sortedIndiceList:
                uniqueFIFs_sortedIndiceList[1] = uniqueFIFs_sortedIndiceList[1] + len(certainFIFindices)
            else:
                uniqueFIFs_sortedIndiceList[1] = len(certainFIFindices)

        if uniqueFIFs[i] > 2:
            if 2 in uniqueFIFs_sortedIndiceList:
                uniqueFIFs_sortedIndiceList[2] = uniqueFIFs_sortedIndiceList[2] + len(certainFIFindices)
            else:
                uniqueFIFs_sortedIndiceList[2] = len(certainFIFindices)
        del tmp
    return uniqueFIFs_sortedIndiceList

def mapFIFtoFeatures(args):
    [direc, family, N] = args

    fIF_perSample = open(direc + family + "-Freq_InverseFreq.csv", "r").read().replace("\n", "").rsplit(",")
    fIF_perSample = [float(itm.rsplit(":")[-1]) for itm in fIF_perSample if float(itm.rsplit(":")[-1]) > 1]
    fIF_perSample = np.array(fIF_perSample)

    FeatFreqMatrix = pickle.load(
        open(direc + family + "/" + family + "-" + family + "-reducedFeatsFrequency.pickle", "rb"))
    FeatFreqMatrix[FeatFreqMatrix > 1] = 1
    FeatFreqSum = FeatFreqMatrix.sum(axis=0)

    features = np.array(list(pickle.load(open(direc + family + "/" + family + "-" + family + "-reducedFeats.pickle", "rb"))))

    # remove the higher n-grams for every threshold
    discovered = np.zeros(100, dtype=int)
    consideredFeatsIndices = np.asarray([], dtype=int)

    uniqueFIFs = np.sort(np.unique(fIF_perSample))[::-1]
    print(uniqueFIFs)

    uniqueFIFs_sortedIndiceList = indicesByFIF(uniqueFIFs, fIF_perSample, FeatFreqSum)
    with open("./uniqueFIFs_FeatureCount-byFam.txt", 'a') as fooy:
        fooy.write(family+":"+str(uniqueFIFs_sortedIndiceList)+"\n")

    return 0

def InverseFreq_CountByFamily(direc):
    freqIFbyFamily = pd.read_csv(direc + "InverseFreq_Count-byFamily.csv", sep=',', index_col=False)
    freqIFbyFamily = freqIFbyFamily.sort_values(by=['IF'], ascending=False)
    return freqIFbyFamily

def main(N):
    direc = "../../data/featureAnalysis/"
    freqIFbyFamily = InverseFreq_CountByFamily(direc)
    headers =  list(freqIFbyFamily.columns.values)[1:]
    # print(freqIFbyFamily)
    print(headers)
    familyList = headers[1:]
    fIF = list(freqIFbyFamily['IF'])
    # print(fIF)

    Arguments = []
    for family in familyList:
        Arguments.append([direc, family, N])

    p = mp.Pool(processes=10)
    corpus = p.map(mapFIFtoFeatures, Arguments)
    p.close()
    p.join()
    print("************************************ ALHAMDULILLAH!!! *****************************************")


if __name__=="__main__":
    N = 10000
    main(N)