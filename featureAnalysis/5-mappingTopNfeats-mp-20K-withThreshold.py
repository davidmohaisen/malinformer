# F/IF: {features 1 ... n}
# Top N: ngram - minimum size such that the F/IF doesn't change
# Feature - freq within family map:

# Set a threshold based on N
# If there are multiple features with the same score. Let say we want top 100 feats. But, F/IF threshold gives us
#     150 features.
#           So, if there are features that are present in the same malware sets, we can consider the longest n-gram
#                 and ignore the rest.
import os, pickle, json, time
import pandas as pd
import numpy as np
import  logging
import multiprocessing as mp

mpl = mp.log_to_stderr()
mpl.setLevel(logging.INFO)

def discoverability(certainFIFindices, FeatFreqMatrix, discovered, consideredFeatsIndices, deleteAtIndices):
    for t in range(len(certainFIFindices)):
        if certainFIFindices[t] in consideredFeatsIndices:
            continue
        if certainFIFindices[t] in deleteAtIndices:
            continue
        presenceInSamples = FeatFreqMatrix[:,certainFIFindices[t]]
        logicResult = np.logical_xor(discovered, presenceInSamples)
        logicResult = np.logical_and(presenceInSamples, logicResult)
        if np.sum(logicResult) >= 1:
            consideredFeatsIndices = np.append(consideredFeatsIndices, certainFIFindices[t])
            discovered = np.logical_or(discovered, presenceInSamples)
        if np.sum(discovered) == 100:
            return consideredFeatsIndices, discovered
    return consideredFeatsIndices, discovered


def consideredFeaturesExtractor_FromUniqueFIFs(uniqueFIFs_sortedIndiceList, uniqueFIFs_sortedFeatsList, family,
                                               consideredFeatsIndices, discovered, FeatFreqMatrix, start_time, uniqueFIFs, N):
    for uniqueFIF in uniqueFIFs:
        certainFIFindices = uniqueFIFs_sortedIndiceList[uniqueFIF]
        famFeatsAtCertainFIF = uniqueFIFs_sortedFeatsList[uniqueFIF]

        deleteAtIndices = set()
        for k1 in range(len(famFeatsAtCertainFIF)):
            for k2 in range(len(famFeatsAtCertainFIF)):
                if k1 == k2:
                    continue
                # if famFeatsAtCertainFIF[k2] in famFeatsAtCertainFIF[k1]:
                if famFeatsAtCertainFIF[k1].startswith(famFeatsAtCertainFIF[k2]):
                    deleteAtIndices = np.append(deleteAtIndices, k1)
                    break

        if len(certainFIFindices) > len(deleteAtIndices):
            if uniqueFIF <2.0:
                with open(family+"-run-Threshold.log", 'a') as toof:
                    toof.write(", ".join(["Finale ::::", family, "::: F/IF: ", str(uniqueFIF), "Number of top features selected: ", str(len(consideredFeatsIndices)), "Discovered Samples: ", str(np.sum(discovered)) , "Delete length: ", str(len(deleteAtIndices)), "Time taken: ", str(time.time() - start_time)])+"\n")
                return consideredFeatsIndices, discovered, 1
            consideredFeatsIndices, discovered = discoverability(certainFIFindices, FeatFreqMatrix, discovered,
                                                                 consideredFeatsIndices, deleteAtIndices)

            if np.sum(discovered) == 100:
                # print("Family: ", family, "100 samples covered. From ", len(certainFIFindices), " features (certainFIFindices), Length of consideredFeatsIndices: ", len(consideredFeatsIndices), "uniqueFIFs: ", uniqueFIFs[i])
                # print(family, "::: F/IF: ", uniqueFIF, "Number of top features selected: ", len(consideredFeatsIndices), "Discovered Samples: ", np.sum(discovered) , "Delete length: ", len(deleteAtIndices), "Time taken: ", time.time() - start_time)
                with open(family+"-run-Threshold.log", 'a') as toof:
                    toof.write(", ".join([family, "::: F/IF: ", str(uniqueFIF), "Number of top features selected: ", str(len(consideredFeatsIndices)), "Discovered Samples: ", str(np.sum(discovered)) , "Delete length: ", str(len(deleteAtIndices)), "Time taken: ", str(time.time() - start_time)])+"\n")
                return consideredFeatsIndices, discovered, 0
        else:
            continue
    return consideredFeatsIndices, discovered, 0

def indicesByFIF(uniqueFIFs, fIF_perSample, FeatFreqSum, features):
    uniqueFIFs_sortedIndiceList, uniqueFIFs_sortedFeatsList = {}, {}
    for i in range(len(uniqueFIFs)):
        certainFIFindices = np.where(fIF_perSample == uniqueFIFs[i])[0]
        featFreqSumAtCertainFIF = FeatFreqSum[certainFIFindices] #Freq of features at certainFIFindices
        famFeatsAtCertainFIF = features[certainFIFindices]

        # sorting by frequency sum
        tmp = featFreqSumAtCertainFIF.argsort()
        certainFIFindices = certainFIFindices[tmp]
        uniqueFIFs_sortedIndiceList[uniqueFIFs[i]] = certainFIFindices
        uniqueFIFs_sortedFeatsList[uniqueFIFs[i]] = famFeatsAtCertainFIF
        del tmp
    return uniqueFIFs_sortedIndiceList, uniqueFIFs_sortedFeatsList

def mapFIFtoFeatures(args):
    [direc, family, N] = args

    fIF_perSample = open(direc + family + "-Freq_InverseFreq.csv", "r").read().replace("\n", "").rsplit(",")
    fIF_perSample = [float(itm.rsplit(":")[-1]) for itm in fIF_perSample]
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
    # print(uniqueFIFs)
    print("*****************")
    print(len(uniqueFIFs))
    print("-------------")

    uniqueFIFs_sortedIndiceList, uniqueFIFs_sortedFeatsList = indicesByFIF(uniqueFIFs, fIF_perSample, FeatFreqSum, features)
    start_time = time.time()

    lengthArray = list(range(500, N, 500))
    while N > len(consideredFeatsIndices):
        if np.sum(discovered) == 100:
            discovered = np.zeros(100, dtype=int)
        consideredFeatsIndices, discovered, flag = consideredFeaturesExtractor_FromUniqueFIFs(uniqueFIFs_sortedIndiceList,
                                                                                        uniqueFIFs_sortedFeatsList,
                                                                                        family,
                                                                                        consideredFeatsIndices,
                                                                                        discovered,
                                                                                        FeatFreqMatrix, start_time, uniqueFIFs, N)

        if flag != 0:
            print("Family: ", family, "COMPLETED!!!!!")
            finalConsideredFeatures = features[consideredFeatsIndices]
            with open(direc + "topNfeats/threshold/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeatures-laters.pickle', 'wb') as handle:
                pickle.dump(finalConsideredFeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(direc + "topNfeats/threshold/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeaturesIndices-laters.pickle', 'wb') as handle:
                pickle.dump(consideredFeatsIndices, handle, protocol=pickle.HIGHEST_PROTOCOL)
            break
        if len(consideredFeatsIndices) >= lengthArray[0]:
            print("Number of features considered", len(consideredFeatsIndices), "Module Number: ", lengthArray[0], "Family: ", family)
            lengthArray.remove(lengthArray[0])
            consideredFeatsIndices = consideredFeatsIndices[:len(consideredFeatsIndices)]
            finalConsideredFeatures = features[consideredFeatsIndices]
            with open(family + "-run-Threshold.log", 'a') as toof:
                toof.write("Finished f1or "+ str(len(consideredFeatsIndices))+"\n")
            with open(direc + "topNfeats/threshold/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeatures-laters.pickle', 'wb') as handle:
                pickle.dump(finalConsideredFeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(direc + "topNfeats/threshold/" + family + "-" + str(len(consideredFeatsIndices)) + '-consideredFeaturesIndices-laters.pickle', 'wb') as handle:
                pickle.dump(consideredFeatsIndices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # consideredFeatsIndices = consideredFeatsIndices
    finalConsideredFeatures = features[consideredFeatsIndices]
    print("Final Writing to file beginning: ", family, "Length of considered features: ", len(consideredFeatsIndices))
    with open(direc + "topNfeats/threshold/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeatures-laters.pickle', 'wb') as handle:
        pickle.dump(finalConsideredFeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(direc + "topNfeats/threshold/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeaturesIndices-laters.pickle', 'wb') as handle:
        pickle.dump(consideredFeatsIndices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0

# def InverseFreq_CountByFamily(direc):
#     freqIFbyFamily = pd.read_csv(direc + "InverseFreq_Count-byFamily.csv", sep=',', index_col=False)
#     freqIFbyFamily = freqIFbyFamily.sort_values(by=['IF'], ascending=False)
#     return freqIFbyFamily

def main(N):
    direc = "../../data/featureAnalysis/"
    # freqIFbyFamily = InverseFreq_CountByFamily(direc)
    # headers =  list(freqIFbyFamily.columns.values)[1:]
    # # print(freqIFbyFamily)
    # print(headers)
    # familyList = headers[1:]
    # fIF = list(freqIFbyFamily['IF'])
    # print(fIF)
    familyList = ['mirai', 'xorddos', 'local', 'tsunami', 'generica', "dofloo", "elknot", "gafgyt", "ganiw", "setag"]

    Arguments = []
    for family in familyList:
        Arguments.append([direc, family, N])

    p = mp.Pool(processes=len(familyList))
    corpus = p.map(mapFIFtoFeatures, Arguments)
    p.close()
    p.join()
    print("************************************ ALHAMDULILLAH!!! *****************************************")


if __name__=="__main__":
    N = 20500
    main(N)