import os, pickle, json, time
import pandas as pd
import numpy as np
import  logging
import multiprocessing as mp

# mpl = mp.log_to_stderr()
# mpl.setLevel(logging.INFO)

def topFeatOccurrence(args):
    [direc, itmFeatIndices, family, itm] = args
    # for family in familyList:
    overallFeatsMatrix = np.array(list(pickle.load(open(direc + "overallNgramOccurrences/" + family + "-frequency0.pickle", "rb"))))
    # for itm in itm_FeatIndices:
    # indices = np.array(list(itmFeatIndices))
    overallFeatsMatrixMod = overallFeatsMatrix[:, itmFeatIndices]

    with open(direc + "discriminativeFeatsOccurrences/noThresholdWithDuplicatedIndices/" + family + "-" + str(itm) + '-topFeatsOccurrences.pickle', 'wb') as handle:
        pickle.dump(overallFeatsMatrixMod, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Completed for family for N = ", itm,"for Family: ", family, "shape:", overallFeatsMatrixMod.shape)
    del itmFeatIndices
    del overallFeatsMatrixMod
    del overallFeatsMatrix

    return 0

def main():
    direct = "../../data/featureAnalysis/topNfeats/withoutThreshold_rest/"
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


    s_t = time.time()

    maxCount = 0
    for fam in fam_count:
        if fam_count[fam][-1] > maxCount:
            maxCount = fam_count[fam][-1]
        continue
    familyList = list(fam_count.keys())
    print(familyList)
    print(maxCount)
    with open(direc + "discriminativeFeatsOccurrences/noThresholdWithDuplicatedIndices/" + 'familyList.pickle', 'wb') as handle:
        pickle.dump(familyList, handle, protocol=pickle.HIGHEST_PROTOCOL)
    lengthArray = list(range(500, maxCount + 400, 500))
    # lengthArray = lengthArray[::-1]
    for itm in lengthArray:
        itmFeatIndices = np.asarray([], dtype=int)
        for family in familyList:
            famTopFeats = pickle.load(open(direct+family+"-"+str(fam_count[family][-1])+"-finalConsideredFeatures-laters.pickle", "rb"))[:itm]
            print("Family: ", family, "Length of indices: ", len(famTopFeats), ". Time taken: ", time.time() - s_t)
            overallFeats = np.array(list(pickle.load(open(direc + "overallNgramOccurrences/" + family + "-features.pickle", "rb"))))
            indices = np.where(np.isin(overallFeats, famTopFeats))[0]
            print(family, ":", len(famTopFeats), ":", len(indices))
            # itmFeatIndices = itmFeatIndices.union(set(list(indices)))
            itmFeatIndices = np.append(itmFeatIndices, indices)
            del overallFeats
            del famTopFeats

        Arguments = []
        for family in familyList:
            Arguments.append([direc, itmFeatIndices, family, itm])
            # topFeatOccurrence(direc, itm_FeatIndices[itm], list(fam_count.keys()), itm)

        p = mp.Pool(processes=len(Arguments))
        corpus = p.map(topFeatOccurrence, Arguments)
        p.close()
        p.join()


    print("************************************ ALHAMDULILLAH!!! *****************************************")


if __name__=="__main__":
    main()
