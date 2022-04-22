import pickle, os
import numpy as np
from sklearn.decomposition import PCA


def testingDataCreator(direc):
    Data = []
    Family = []

    fam_files = {}

    for file in os.listdir(direc):
        if "-frequency" not in file:#file.endswith("-frequency0.pickle"):
            continue
        fam = file.rsplit("-")[0]
        if fam not in fam_files:
            fam_files[fam] = []
            fam_files[fam].append(file)
        else:
            fam_files[fam].append(file)

    for fam in fam_files:
        for file in fam_files[fam]:
            famFreqMatrix = pickle.load(open(direc+file, "rb"))

            fam = file.rsplit("-")[0]
            for i in range(famFreqMatrix.shape[0]):
                Family.append(fam)

            if len(Data) == 0:
                Data = famFreqMatrix
            else:
                Data = np.concatenate((Data, famFreqMatrix), axis=0)
            del famFreqMatrix

    print("length of Data ", len(Data))
    print("length of Data ", len(Family))
    return Data, Family


def trainingDataCreator(direc):
    Data = []
    Family = []

    for file in os.listdir(direc):
        if not file.endswith("-frequency0.pickle"):
            continue

        famFreqMatrix = pickle.load(open(direc+file, "rb"))
        fam = file.rsplit("-")[0]
        for i in range(famFreqMatrix.shape[0]):
            Family.append(fam)

        if len(Data) == 0:
            Data = famFreqMatrix
        else:
            Data = np.concatenate((Data, famFreqMatrix), axis=0)
        del famFreqMatrix

    print("length of Data ", len(Data))
    print("length of Data ", len(Family))
    return Data, Family


trainDirec = "../../data/overallNgramOccurrences/"
DataTrain, trainFamily = trainingDataCreator(trainDirec)

testDirec = "../data/dynamic/transFormedOverTrainingVocab/"
DataTest, testFamily = testingDataCreator(testDirec)

if not os.path.isdir("../data/pca/"):
    os.mkdir("../data/pca")

f0 = open("../data/pca/Train_Family.pickle","wb")
pickle.dump(trainFamily,f0)
f0.close()

f1 = open("../data/pca/Test_Family.pickle", "wb")
pickle.dump(testFamily, f1)
f1.close()

print("Starting PCA")

variances = [0.999, 0.99, 0.95, 0.90, 0.85]
transformed = []
for variance in variances:
    pca = PCA(n_components = variance)
    pca.fit(DataTrain)
    print("Fitting completed")


    transTrain = pca.transform(DataTrain)
    print("Transforming completed for", "Variance = ", variance)
    print(transTrain.shape)
    f2 = open("../data/pca/PCA2components-Train-family_variance-" + str(variance) + ".pickle", "wb")
    pickle.dump(transTrain, f2)
    f2.close()
    del transTrain


    transTest = pca.transform(DataTest)
    print("Transforming completed for", "Variance = ", variance)
    print(transTest.shape)
    f3 = open("../data/pca/PCA2components-Test-family_variance-"+str(variance)+".pickle","wb")
    pickle.dump(transTest,f3)
    f3.close()
    del transTest
