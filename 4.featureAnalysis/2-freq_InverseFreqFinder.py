import os, pickle
import numpy as np

inverseFreqAnalysisPath = '../../data/featureAnalysis/inverseFreqAnalysis/'

freqInvFreq = pickle.load(open(inverseFreqAnalysisPath+"familyFeat-Frequency_InverseFrequency.pickle", 'rb'))
fam_freq = freqInvFreq[0]
fam_invFreq = freqInvFreq[1]

overallFam_FIF = {}
for fam in fam_freq:
    fam_FIF = {}
    freq = fam_freq[fam]
    invFreq = fam_invFreq[fam]

    for feat in freq:
        f = freq[feat]
        if feat in invFreq:
            invf = invFreq[feat]
        else:
            invf = invFreq[feat] = 0.1
        fif = float(f/invf)
        fam_FIF[feat] = fif

    # overallFam_FIF[fam] = fam_FIF
    print("Completed for ", fam)

    with open(inverseFreqAnalysisPath+fam+"-Feat-FIF.pickle", 'wb') as handle:
        pickle.dump(fam_FIF, handle, protocol=pickle.HIGHEST_PROTOCOL)
    