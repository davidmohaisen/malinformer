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
                                               consideredFeatsIndices, discovered, FeatFreqMatrix, start_time, uniqueFIFs, N, alreadyDones, doneCount):
    for uniqueFIF in uniqueFIFs:
        certainFIFindices = uniqueFIFs_sortedIndiceList[uniqueFIF]
        famFeatsAtCertainFIF = uniqueFIFs_sortedFeatsList[uniqueFIF]
        # certainFIFindices_dupl = certainFIFindices

        idxs = np.where(np.isin(certainFIFindices, alreadyDones))[0]
        certainFIFindices = np.delete(certainFIFindices, idxs)
        famFeatsAtCertainFIF = np.delete(famFeatsAtCertainFIF, idxs)

        deleteAtIndices = np.asarray([])
        for k1 in range(len(famFeatsAtCertainFIF)):
            # if certainFIFindices[k1] in deleteAtIndices:
            #     certainFIFindices = np.delete(certainFIFindices, k1)
            #     continue
            for k2 in range(len(famFeatsAtCertainFIF)):
                if k1 == k2:
                    continue
                # if famFeatsAtCertainFIF[k2] in famFeatsAtCertainFIF[k1]:
                if famFeatsAtCertainFIF[k1].startswith(famFeatsAtCertainFIF[k2]):
                    deleteAtIndices = np.append(deleteAtIndices, k1)
                    # certainFIFindices = np.delete(certainFIFindices, k1)
                    break

        if len(certainFIFindices) > len(deleteAtIndices):
            if float(uniqueFIF) <=float(2):
                print("family: ", family, "Should do the final saving", "F/IF: ", uniqueFIF, "length of consideredFeatsIndices", len(consideredFeatsIndices))
                idxs2 = np.where(np.isin(certainFIFindices, deleteAtIndices))[0]
                certainFIFindices = np.delete(certainFIFindices, idxs2)
                consideredFeatsIndices = np.append(consideredFeatsIndices,certainFIFindices)[:20000]
                if len(consideredFeatsIndices) < 20000:
                    continue
                # consideredFeatsIndices = np.array([itm for itm in list(certainFIFindices) if itm not in deleteAtIndices][:20000])
                # consideredFeatsIndices = certainFIFindices[:20000-doneCount]
                # for itm in deleteAtIndices:
                #     if itm in consideredFeatsIndices:
                #         consideredFeatsIndices = np.delete(consideredFeatsIndices, itm)
                # consideredFeatsIndices = np.array([itm for itm in list(certainFIFindices) if itm not in list(deleteAtIndices)][:N-doneCount])
                with open(family+"-run-withoutThreshold_rest.log", 'a') as toof:
                    toof.write(", ".join(["Finale ::::", family, "::: F/IF: ", str(uniqueFIF), "Number of top features selected: ", str(len(consideredFeatsIndices)), "Discovered Samples: ", str(np.sum(discovered)) , "Delete length: ", str(len(deleteAtIndices)), "Time taken: ", str(time.time() - start_time)])+"\n")
                return consideredFeatsIndices, discovered, 1

            # print("family: ", family, "::: F/IF: ", str(uniqueFIF), "entering discoverability.")
            consideredFeatsIndices, discovered = discoverability(certainFIFindices, FeatFreqMatrix, discovered,
                                                                 consideredFeatsIndices, deleteAtIndices)

            if np.sum(discovered) == 100:
                # print("Family: ", family, "100 samples covered. From ", len(certainFIFindices), " features (certainFIFindices), Length of consideredFeatsIndices: ", len(consideredFeatsIndices), "uniqueFIFs: ", uniqueFIFs[i])
                print(family, "::: F/IF: ", uniqueFIF, "Number of top features selected: ", len(consideredFeatsIndices), "Discovered Samples: ", np.sum(discovered) , "Delete length: ", len(deleteAtIndices), "Time taken: ", time.time() - start_time)
                with open(family+"-run-withoutThreshold_rest.log", 'a') as toof:
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
    [direc, family, N, doneCount] = args

    fIF_perSample = open(direc + family + "-Freq_InverseFreq.csv", "r").read().replace("\n", "").rsplit(",")
    fIF_perSample = [float(itm.rsplit(":")[-1]) for itm in fIF_perSample]
    fIF_perSample = np.array(fIF_perSample)

    FeatFreqMatrix = pickle.load(
        open(direc + family + "/" + family + "-" + family + "-reducedFeatsFrequency.pickle", "rb"))
    FeatFreqMatrix[FeatFreqMatrix > 1] = 1
    FeatFreqSum = FeatFreqMatrix.sum(axis=0)

    features = np.array(list(pickle.load(open(direc + family + "/" + family + "-" + family + "-reducedFeats.pickle", "rb"))))

    # remove the higher n-grams for every threshold

    uniqueFIFs = np.sort(np.unique(fIF_perSample))[::-1]
    # print(uniqueFIFs)
    print("*****************")
    print(len(uniqueFIFs), doneCount)
    print("-------------")

    alreadyDones = pickle.load(open(direc + "topNfeats/withoutThreshold/" + family + "-" + str(doneCount) + '-consideredFeaturesIndices-laters.pickle', 'rb'))
    consideredFeatsIndices = np.array(list(alreadyDones))
    discovered = np.zeros(100, dtype=int)

    uniqueFIFs_sortedIndiceList, uniqueFIFs_sortedFeatsList = indicesByFIF(uniqueFIFs, fIF_perSample, FeatFreqSum, features)
    start_time = time.time()

    lengthArray = list(range(doneCount, N, 500))
    while N > (len(consideredFeatsIndices)):
        if np.sum(discovered) == 100:
            discovered = np.zeros(100, dtype=int)
        consideredFeatsIndices, discovered, flag = consideredFeaturesExtractor_FromUniqueFIFs(uniqueFIFs_sortedIndiceList,
                                                                                        uniqueFIFs_sortedFeatsList,
                                                                                        family,
                                                                                        consideredFeatsIndices,
                                                                                        discovered,
                                                                                        FeatFreqMatrix, start_time, uniqueFIFs, N, alreadyDones, doneCount)

        # if flag == 1 and len(consideredFeatsIndices) >= 20000:
        if flag != 0:
            print("Finishing now for Family: ", family, "COMPLETING!!!!!")
            # consideredFeatsIndices = consideredFeatsIndices[:20000]
            # consideredFeatsIndices = np.append(consideredFeatsIndices, certainFIFindices)
            print(consideredFeatsIndices)
            print("length of features: ", len(features))
            finalConsideredFeatures = features[consideredFeatsIndices]
            with open(direc + "topNfeats/withoutThreshold_rest/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeatures-laters.pickle', 'wb') as handle:
                pickle.dump(finalConsideredFeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(direc + "topNfeats/withoutThreshold_rest/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeaturesIndices-laters.pickle', 'wb') as handle:
                pickle.dump(consideredFeatsIndices, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Family: ", family, "COMPLETED!!!!!")
            # if len(consideredFeatsIndices) < 20000:
            #     continue
            break
        if len(consideredFeatsIndices) >= lengthArray[0]:
            print("Number of features considered", len(consideredFeatsIndices), "Module Number: ", lengthArray[0], "Family: ", family)
            lengthArray.remove(lengthArray[0])
            # consideredFeatsIndices = consideredFeatsIndices[:len(consideredFeatsIndices)]
            finalConsideredFeatures = features[consideredFeatsIndices]
            with open(family + "-run-withoutThreshold_rest.log", 'a') as toof:
                toof.write("Finished f1or "+ str(len(consideredFeatsIndices))+"\n")
            with open(direc + "topNfeats/withoutThreshold_rest/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeatures-laters.pickle', 'wb') as handle:
                pickle.dump(finalConsideredFeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(direc + "topNfeats/withoutThreshold_rest/" + family + "-" + str(len(consideredFeatsIndices)) + '-consideredFeaturesIndices-laters.pickle', 'wb') as handle:
                pickle.dump(consideredFeatsIndices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # consideredFeatsIndices = consideredFeatsIndices
    # finalConsideredFeatures = features[consideredFeatsIndices]
    # print("Final Writing to file beginning: ", family, "Length of considered features: ", len(consideredFeatsIndices))
    # with open(direc + "topNfeats/withoutThreshold/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeatures-laters.pickle', 'wb') as handle:
    #     pickle.dump(finalConsideredFeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(direc + "topNfeats/withoutThreshold/" + family + "-" + str(len(consideredFeatsIndices)) + '-finalConsideredFeaturesIndices-laters.pickle', 'wb') as handle:
    #     pickle.dump(consideredFeatsIndices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0

def donesCreator():
    path = "../../data/featureAnalysis/topNfeats/withoutThreshold/"
    fam_countList = {}
    for file in os.listdir(path):
        fam = file.rsplit("-")[0]
        cnt = int(file.rsplit("-")[1])
        if fam not in fam_countList:
            fam_countList[fam] = []
            fam_countList[fam].append(cnt)
        else:
            fam_countList[fam].append(cnt)
    fam_doneCnt = {}
    for fam in fam_countList:
        fam_doneCnt[fam] = sorted(fam_countList[fam])[-1]

    del fam_countList
    return fam_doneCnt

def main(N):
    direc = "../../data/featureAnalysis/"
    
    fam_doneCnt = donesCreator()
    print(fam_doneCnt)
    # exit()

    # familyList = ['mirai', 'xorddos', 'local', 'tsunami', 'generica', "dofloo", "elknot", "gafgyt", "ganiw", "setag"]

    Arguments = []
    for family in fam_doneCnt:
        if fam_doneCnt[family] >=20000:
            continue
        Arguments.append([direc, family, N, fam_doneCnt[family]])

    p = mp.Pool(processes=len(Arguments))
    corpus = p.map(mapFIFtoFeatures, Arguments)
    p.close()
    p.join()
    print("************************************ ALHAMDULILLAH!!! *****************************************")


if __name__=="__main__":
    N = 20500
    main(N)