import os, pickle, json, time
import pandas as pd
import numpy as np
import  logging
import multiprocessing as mp

mpl = mp.log_to_stderr()
mpl.setLevel(logging.INFO)

def topFeatOccurrence(args):
    [direc, itmFeatIndices, familyList, itm] = args
    for family in familyList:
        overallFeatsMatrix = np.array(list(pickle.load(open(direc + "overallNgramOccurrences/" + family + "-frequency0.pickle", "rb"))))
        # for itm in itm_FeatIndices:
        indices = np.array(list(itmFeatIndices))
        overallFeatsMatrixMod = overallFeatsMatrix[:, indices]

        with open(direc + "discriminativeFeatsOccurrences/threshold/" + family + "-" + str(itm) + '-topFeatsOccurrences.pickle', 'wb') as handle:
            pickle.dump(overallFeatsMatrixMod, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Completed for family for N = ", itm,"for Family: ", family, "shape:", overallFeatsMatrixMod.shape)
        del indices
        del overallFeatsMatrixMod
        del overallFeatsMatrix

        return 0

def main():
    direct = "../../data/featureAnalysis/topNfeats/threshold/"
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

    maxCountSet = set()
    for family in fam_count:
        if family == "generica":
            famTopFeats = pickle.load(open(direct.replace("threshold", "withoutThreshold")+family+"-500-finalConsideredFeatures-laters.pickle", "rb"))[:17]
            # withoutThreshold
        else:
            famTopFeats = pickle.load(open(direct+family+"-"+str(fam_count[family][-1])+"-finalConsideredFeatures-laters.pickle", "rb"))
        print("Family: ", family, "Length of indices: ", len(famTopFeats), ". Time taken: ", time.time() - s_t)
        overallFeats = np.array(list(pickle.load(open(direc + "overallNgramOccurrences/" + family + "-features.pickle", "rb"))))
        maxCountSet.add(fam_count[family][-1])
        # overallFeatsMatrix = np.array(list(pickle.load(open(direc + "overallNgramOccurrences/" + family + "-frequency0.pickle", "rb"))))

        # indices = np.where(np.isin(overallFeats, famTopFeats_module))[0]
        # indices = [np.where(overallFeats==ft)[0][0] for ft in famTopFeats]

        maxCount = max(list(maxCountSet))
        lengthArray = list(range(500, maxCount+400, 500))
        for itm in lengthArray:
            famTopFeats_module = famTopFeats[:itm]
            indices = np.where(np.isin(overallFeats, famTopFeats_module))[0]
            if itm not in itm_FeatIndices:
                itm_FeatIndices[itm] = set()
                itm_FeatIndices[itm] = set(list(indices))
            else:
                itm_FeatIndices[itm] = itm_FeatIndices[itm].union(set(list(indices)))
        del overallFeats

    Arguments = []
    for itm in itm_FeatIndices:
        Arguments.append([direc, itm_FeatIndices[itm], list(fam_count.keys()), itm])
        # topFeatOccurrence(direc, itm_FeatIndices[itm], list(fam_count.keys()), itm)

    p = mp.Pool(processes=len(Arguments))
    corpus = p.map(topFeatOccurrence, Arguments)
    p.close()
    p.join()
    print("************************************ ALHAMDULILLAH!!! *****************************************")


if __name__=="__main__":
    main()
