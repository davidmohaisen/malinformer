import os, pickle, json, time
import pandas as pd
import numpy as np
import  logging
import multiprocessing as mp

mpl = mp.log_to_stderr()
mpl.setLevel(logging.INFO)

def topFeatOccurrence(args):
    [direc, family, indices, ctr] = args
    print(family, len(indices))
    # for family in familyList:
    overallFeatsMatrix = np.array(list(pickle.load(open(direc + "overallNgramOccurrences/" + family + "-frequency0.pickle", "rb"))))
    # for itm in itm_FeatIndices:
    # indices = np.array(list(itmFeatIndices))
    overallFeatsMatrixMod = overallFeatsMatrix[:, indices]
    print("Starting for family: ", family, "shape:", overallFeatsMatrixMod.shape, "before: ", overallFeatsMatrix.shape)

    with open(direc + "discriminativeFeatsOccurrences/discoverability/" + family + "-" + str(ctr)+'-DiscoveryTopFeatsOccurrences.pickle', 'wb') as handle:
        pickle.dump(overallFeatsMatrixMod, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Completed for family for Family: ", family, "shape:", overallFeatsMatrixMod.shape)
    del indices
    del overallFeatsMatrixMod
    del overallFeatsMatrix

    return 0

def unifiedFeatCreator(args0):
    [direct, direc, family, famctr, fam_firstDiscoverCnt] = args0
    #
    famTopFeats = pickle.load(
        open(direct + family + "-" + str(famctr) + "-finalConsideredFeatures-laters.pickle", "rb"))[
                  :fam_firstDiscoverCnt]
    # print("Family: ", family, "Length of indices: ", len(famTopFeats))
    overallFeats = np.array(
        list(pickle.load(open(direc + "overallNgramOccurrences/" + family + "-features.pickle", "rb"))))
    indices = np.where(np.isin(overallFeats, famTopFeats))[0]
    print("Family: ", family, "Length of indices: ", len(famTopFeats), "after where condition", len(indices))
    print(indices)
    return indices

def main():
    direct = "../../data/featureAnalysis/topNfeats/withoutThreshold/"
    direc = "../../data/"
    fam_count = {}
    for file in os.listdir(direct):
        if os.path.isdir(direct+file):
            continue
        if "consideredFeaturesIndices" in file:
            continue
        else:
            fam = file.rsplit("-")[0]
            count = int(file.rsplit("-")[1])
            if fam not in fam_count:
                fam_count[fam] = []
                fam_count[fam].append(count)
                fam_count[fam] = sorted(fam_count[fam])
            else:
                fam_count[fam].append(count)
                fam_count[fam] = sorted(fam_count[fam])
            del fam
            del count

        #del fam
        #del count
        #del file

    itm_FeatIndices = {}
    s_t = time.time()

    fam_firstDiscover = {'mirai': [21, 42, 62, 82, 103, 124, 145, 166, 186, 206], 'xorddos': [11, 21,32,44,54,64,74,84,95,107], 'local': [7, 14, 22,30,38,46,53,60,67,74], 'tsunami': [9, 18,27,35,41,48,54,59,64,68], 'generica': [7,14,19,24,29,34,39,46,53,59], "dofloo": [16, 33, 50, 66, 82, 98, 113, 128, 143, 158], "elknot": [15,30,44,57,69,81,93,105,117,128], "gafgyt": [10, 19,28,36,44,52,60,68,76,84], "ganiw": [27,57,85,113,140,166,193,218,244,269], "setag": [17,33,49,67,84,103,122,140,158,176]}
    # ['mirai', 'xorddos', 'local', 'tsunami', 'generica', "dofloo", "elknot", "gafgyt", "ganiw", "setag"]

    # overallFeats = np.asarray([])
    # UnifArguments = []
    for ctr in list(range(0,10)):
        overallFeatIndices = set()
        for family in fam_firstDiscover:
            overallFeatIndices = overallFeatIndices.union(set(unifiedFeatCreator([direct, direc, family, fam_count[family][0], fam_firstDiscover[family][ctr]])))

        overallFeatIndices = np.array(list(overallFeatIndices))
        print("overallFeats", len(overallFeatIndices))
        # print(overallFeatIndices)


        Arguments = []
        for fam in fam_firstDiscover:
            Arguments.append([direc, fam, overallFeatIndices, ctr])
            # topFeatOccurrence(direc, itm_FeatIndices[itm], list(fam_count.keys()), itm)

        p = mp.Pool(processes=len(Arguments))
        corpus = p.map(topFeatOccurrence, Arguments)
        p.close()
        p.join()
    print("************************************ ALHAMDULILLAH!!! *****************************************")


if __name__=="__main__":
    main()
