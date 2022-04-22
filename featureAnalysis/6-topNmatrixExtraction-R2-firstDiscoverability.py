import os, pickle, json, time
import pandas as pd
import numpy as np
import  logging
import multiprocessing as mp

mpl = mp.log_to_stderr()
mpl.setLevel(logging.INFO)

def topFeatOccurrence(args):
    [direc, family, indices] = args
    print(family, len(indices))
    # for family in familyList:
    overallFeatsMatrix = np.array(list(pickle.load(open(direc + "overallNgramOccurrences/" + family + "-frequency0.pickle", "rb"))))
    # for itm in itm_FeatIndices:
    # indices = np.array(list(itmFeatIndices))
    overallFeatsMatrixMod = overallFeatsMatrix[:, indices]
    print("Starting for family: ", family, "shape:", overallFeatsMatrixMod.shape, "before: ", overallFeatsMatrix.shape)

    with open(direc + "discriminativeFeatsOccurrences/discoverability/" + family + "-" + '-firstDiscoveryTopFeatsOccurrences.pickle', 'wb') as handle:
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

    fam_firstDiscover = {'mirai': 21, 'xorddos': 11, 'local': 7, 'tsunami': 9, 'generica': 7, "dofloo": 16, "elknot": 15, "gafgyt": 10, "ganiw": 27, "setag": 17}
    # ['mirai', 'xorddos', 'local', 'tsunami', 'generica', "dofloo", "elknot", "gafgyt", "ganiw", "setag"]

    # overallFeats = np.asarray([])
    # UnifArguments = []
    overallFeatIndices = set()
    for family in fam_firstDiscover:
        overallFeatIndices = overallFeatIndices.union(set(unifiedFeatCreator([direct, direc, family, fam_count[family][0], fam_firstDiscover[family]])))
    #     UnifArguments.append([direct, direc, family, fam_count[family][0], fam_firstDiscover[family]])
    # # overallFeats = np.append(overallFeats,indices)
    # p0 = mp.Pool(processes=len(UnifArguments))
    # overallFeats = p0.map(unifiedFeatCreator, UnifArguments)
    # p0.close()
    # p0.join()


    overallFeatIndices = np.array(list(overallFeatIndices))
    print("overallFeats", len(overallFeatIndices))


    Arguments = []
    for fam in fam_firstDiscover:
        Arguments.append([direc, fam, overallFeatIndices])
        # topFeatOccurrence(direc, itm_FeatIndices[itm], list(fam_count.keys()), itm)

    p = mp.Pool(processes=len(Arguments))
    corpus = p.map(topFeatOccurrence, Arguments)
    p.close()
    p.join()
    print("************************************ ALHAMDULILLAH!!! *****************************************")


if __name__=="__main__":
    main()
