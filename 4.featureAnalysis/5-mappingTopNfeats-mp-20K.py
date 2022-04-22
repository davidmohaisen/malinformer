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
from collections import Counter

# mpl = mp.log_to_stderr()
# mpl.setLevel(logging.INFO)

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

def wentBeyondThreshold(uniqueFIFs, uniqueFIFs_sortedIndiceList, deleteAtIndices, consideredFeatsIndices, N, family):
    for uniqueFIF in uniqueFIFs:
        if uniqueFIF > 1:
            continue
        certainFIFindices = uniqueFIFs_sortedIndiceList[uniqueFIF]
        for itm in certainFIFindices:
            if itm in consideredFeatsIndices:
                continue
            elif itm in deleteAtIndices:
                continue
            else:
                consideredFeatsIndices = np.append(consideredFeatsIndices, itm)
                print(family, ": unique FIF value <= 1: ", uniqueFIF, ". Features with unique FFIF = ", len(uniqueFIFs), " being probed =", len(certainFIFindices), ". Found features: ", len(consideredFeatsIndices))
            if len(consideredFeatsIndices) >= N:
                return consideredFeatsIndices

def consideredFeaturesExtractor_FromUniqueFIFs(uniqueFIFs_sortedIndiceList, uniqueFIFs_sortedFeatsList, family,
                                               consideredFeatsIndices, discovered, FeatFreqMatrix, start_time, uniqueFIFs, alreadyDones, doneCount, N, outPath):
    for uniqueFIF in uniqueFIFs:
        print("Running for: ", uniqueFIF)
        if uniqueFIF <= 1.0:
            consideredFeatsIndices = wentBeyondThreshold(uniqueFIFs, uniqueFIFs_sortedIndiceList, deleteAtIndices, consideredFeatsIndices, N, family)
            return consideredFeatsIndices, np.asarray([])
        certainFIFindices = uniqueFIFs_sortedIndiceList[uniqueFIF]
        famFeatsAtCertainFIF = uniqueFIFs_sortedFeatsList[uniqueFIF]

        deleteAtIndices = alreadyDones
        for k1 in range(len(famFeatsAtCertainFIF)):
            for k2 in range(len(famFeatsAtCertainFIF)):
                if k1 == k2:
                    continue
                # if famFeatsAtCertainFIF[k2] in famFeatsAtCertainFIF[k1]:
                if famFeatsAtCertainFIF[k1].startswith(famFeatsAtCertainFIF[k2]):
                    if certainFIFindices[k2] not in deleteAtIndices:
                        deleteAtIndices = np.append(deleteAtIndices, certainFIFindices[k2])
                    break

        if len(set(certainFIFindices) - set(deleteAtIndices)) > 0:
            if uniqueFIF <2:
                print(family, ": unique FIF value < 2: ", uniqueFIF, ". Features with unique FFIF = ", len(uniqueFIFs), " being probed =", len(certainFIFindices), ". Found features: ", len(consideredFeatsIndices))
                ctr1, ctr2 = 0, 0
                for itm in certainFIFindices:
                    ctr1+=1
                    if itm not in deleteAtIndices:
                        ctr2+=1
                        consideredFeatsIndices = np.append(consideredFeatsIndices, itm)
                        # print(family, ": unique FIF value < 2: ", uniqueFIF, ". Features with unique FFIF = ", len(uniqueFIFs), " being probed =", len(certainFIFindices), ". Found features: ", len(consideredFeatsIndices), "Inserted ", ctr2, "out of ", ctr1)
                #     preConsFeats = [itm for itm in certainFIFindices if itm not in deleteAtIndices]
                # consideredFeatsIndices = np.asarray(list(consideredFeatsIndices)+preConsFeats)
                if len(consideredFeatsIndices) < N:
                    continue
                else:
                    return consideredFeatsIndices[:N], np.asarray([])
            print(family, ": Current discoverability round: ", uniqueFIF, ". Total unique FFIFs are: ", len(uniqueFIFs), ". Found features: ", len(consideredFeatsIndices))
            consideredFeatsIndices, discovered = discoverability(certainFIFindices, FeatFreqMatrix, discovered,
                                                                 consideredFeatsIndices, deleteAtIndices)

            if np.sum(discovered) == 100:
                # print("Family: ", family, "100 samples covered. From ", len(certainFIFindices), " features (certainFIFindices), Length of consideredFeatsIndices: ", len(consideredFeatsIndices), "uniqueFIFs: ", uniqueFIFs[i])
                # print(family, "::: F/IF: ", uniqueFIF, "Number of top features selected: ", len(consideredFeatsIndices), "Discovered Samples: ", np.sum(discovered) , "Delete length: ", len(deleteAtIndices), "Time taken: ", time.time() - start_time)
                with open(outPath+"Logs/"+family+"-run.log", 'a') as toof:
                    toof.write(", ".join([family, "::: F/IF: ", str(uniqueFIF), "Number of top features selected: ", str(len(consideredFeatsIndices)), "Discovered Samples: ", str(np.sum(discovered)) , "Delete length: ", str(len(deleteAtIndices)), "Time taken: ", str(time.time() - start_time)])+"\n")
                return consideredFeatsIndices, discovered
            else:
                continue
        else:
            continue
    return consideredFeatsIndices, discovered

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

def FIFmapper(family):
    frequentFeatPath = '../../data/featureAnalysis/frequentFeatsPerFamily/'
    inverseFreqAnalysisPath = '../../data/featureAnalysis/inverseFreqAnalysis/'
    
    try:
        [FIFlist, features] = pickle.load(open(inverseFreqAnalysisPath + family+'FIFlist_CorrespondingFeats.pickle', 'rb'))
    except:
        frequentFeats_indice = pickle.load(open(frequentFeatPath+str(family)+"-featName_featIndex.pickle", 'rb'))
        index_feats = dict((v,k) for k,v in frequentFeats_indice.items())
        
        feat_fif = pickle.load(open(inverseFreqAnalysisPath+family+"-Feat-FIF.pickle", 'rb'))
        FIFlist, features = [], []
        freqFeatIndices = sorted(frequentFeats_indice.values())
        for idx in freqFeatIndices:
            featname = index_feats[idx]
            FIFlist.append(feat_fif[featname])
            features.append(featname)
        with open(inverseFreqAnalysisPath + family+'FIFlist_CorrespondingFeats.pickle', 'wb') as handle:
            pickle.dump([FIFlist, features], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(len(FIFlist), len(features), "Maxim F/IF", max(FIFlist))
    return FIFlist, features


def mapFIFtoFeatures(args):
    [direc, family, N, doneCount, outPath] = args

    fIF_perSample, features = FIFmapper(family)
    fIF_perSample = [round(val, 1) for val in fIF_perSample]
    fIF_perSample, features = np.asarray(fIF_perSample), np.asarray(features)

    print(Counter(fIF_perSample))

    # print("########################")
    # print(type(FIFlist), len(FIFlist), type(features), len(features))
    # # print(type(features), len(features))
    # print("---------------------")
    # print("F/IF: ", FIFlist[:10])
    # print("Features: ", features[:10])
    # print("*****************************")

    FeatFreqMatrix = pickle.load(
        open(direc + family + "-frequenFeatureOccurrences.pickle", "rb"))
    FeatFreqMatrix[FeatFreqMatrix > 1] = 1
    FeatFreqSum = FeatFreqMatrix.sum(axis=0)

    # remove the lower n-grams for every threshold

    uniqueFIFs = np.sort(np.unique(fIF_perSample))[::-1]
    # print(uniqueFIFs)
    print("*****************")
    print(family, " : ", len(uniqueFIFs))
    print("-------------")

    try:
        alreadyDones = pickle.load(open(outPath + family + "-" + str(len(doneCount)) + '-consideredFeaturesIndices.pickle', 'rb'))
    except:
        alreadyDones = np.asarray([], dtype=int)

    discovered = np.zeros(100, dtype=int)
    consideredFeatsIndices = alreadyDones
    # consideredFeatsIndices = np.asarray([], dtype=int)

    uniqueFIFs_sortedIndiceList, uniqueFIFs_sortedFeatsList = indicesByFIF(uniqueFIFs, fIF_perSample, FeatFreqSum, features)
    start_time = time.time()

    startPoint = ((int(doneCount/500))*500)+500

    lengthArray = list(range(500, N, 500))
    while N > len(consideredFeatsIndices):
        if np.sum(discovered) == 100:
            discovered = np.zeros(100, dtype=int)
        consideredFeatsIndices, discovered = consideredFeaturesExtractor_FromUniqueFIFs(uniqueFIFs_sortedIndiceList,
                                                                                        uniqueFIFs_sortedFeatsList,
                                                                                        family,
                                                                                        consideredFeatsIndices,
                                                                                        discovered,
                                                                                        FeatFreqMatrix, start_time, uniqueFIFs, alreadyDones, doneCount, N, outPath)
        if len(consideredFeatsIndices) >= lengthArray[0]:
            print(lengthArray[0], "completed. ", family, ": Number of features considered", len(consideredFeatsIndices), "Module Number: ", lengthArray[0])
            lengthArray.remove(lengthArray[0])
            # consideredFeatsIndices = consideredFeatsIndices#[:len(consideredFeatsIndices)]
            # print(type(consideredFeatsIndices), type(features), len(features), max(consideredFeatsIndices))
            # print(consideredFeatsIndices[:10])
            # print(family, " : ", lengthArray[0], "finding features. ")
            finalConsideredFeatures = features[consideredFeatsIndices]
            # print(family, " : ", lengthArray[0], "features found. ")
            with open(outPath+"Logs/"+family + "-run.log", 'a') as toof:
                toof.write("Finished for "+ str(len(consideredFeatsIndices))+"\n")
            # print(family, " : ", lengthArray[0], "logs written. Now file write starting")
            with open(outPath + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeatures.pickle', 'wb') as handle:
                pickle.dump(finalConsideredFeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(outPath + family + "-" + str(len(consideredFeatsIndices)) + '-consideredFeaturesIndices.pickle', 'wb') as handle:
                pickle.dump(consideredFeatsIndices, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(family, " : ", lengthArray[0], "logs written. Files written to disk")
        else:
            continue

    # consideredFeatsIndices = consideredFeatsIndices
    # finalConsideredFeatures = features[consideredFeatsIndices]
    # print("Final Writing to file beginning: ", family, consideredFeatsIndices, len(consideredFeatsIndices))
    # with open(direc + "topNfeats/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeatures-laters.pickle', 'wb') as handle:
    #     pickle.dump(finalConsideredFeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(direc + "topNfeats/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeaturesIndices-laters.pickle', 'wb') as handle:
    #     pickle.dump(consideredFeatsIndices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0

def donesCreator(outPath):
    fam_cnt = {}
    for file in os.listdir(outPath):
        if os.path.isdir(outPath+file):
            continue
        fam = file.rsplit("-")[0]
        cnt = int(file.rsplit("-")[1])
        if fam not in fam_cnt:
            fam_cnt[fam] = cnt
        else:
            fam_cnt[fam] = max(fam_cnt[fam], cnt)
    print("Dones: ", fam_cnt)
    return fam_cnt

def main(N):
    direc = '../../data/featureAnalysis/frequentFeatOccurrence/'
    outPath = '../../data/featureAnalysis/topNfeats-R2/'
    if not os.path.isdir(outPath):
        os.mkdir(outPath)
    if not os.path.isdir(outPath+"Logs/"):
        os.mkdir(outPath+"Logs/")

    dones = donesCreator(outPath)
    familyList = list(pickle.load(open('../../data/trainFamily_MalwareList.pickle', 'rb')).keys())
    Arguments = []
    for family in familyList:
        try:
            if dones[family] >= N:
                continue
        except:
            pass

        try:
            Arguments.append([direc, family, N, dones[family], outPath])
        except:
            Arguments.append([direc, family, N, 0, outPath])

    print(Arguments[:5])
    p = mp.Pool(processes=len(familyList))
    corpus = p.map(mapFIFtoFeatures, Arguments)
    p.close()
    p.join()
    # for itm in Arguments:
    #     mapFIFtoFeatures(itm)
    print("************************************ ALHAMDULILLAH!!! *****************************************")


if __name__=="__main__":
    N = 20500
    main(N)