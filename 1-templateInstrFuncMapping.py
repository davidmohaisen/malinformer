import pickle, re, time, os
import multiprocessing as mp

# jne 0x8048f62                         - call 0x4084e0, jmp 0x407d5b

def registerReferenceUnifier(line,bits):
    line = str(line)
    lineUnified = ""
    registers = ["rax","rcx","rdx","rbx","rsi","rdi","rsp","rbp","r8d","r9d","r10d","r11d","r12d","r13d","r14d","r15d","r8w","r9w","r10w","r11w","r12w","r13w","r14w","r15w","r8b","r9b","r10b","r11b","r12b","r13b","r14b","r15b","r8","r9","r10","r11","r12","r13","r14","r15","eax","ecx","edx","ebx","esi","edi","esp","ebp","sil","dil","spl","bpl","ax","al","cx","cl","dx","dl","bx","bl","cs","ds","es","fs","gs","ss","ah","bh","ch","dh","ip","sp","bp","si","di","of","df","if","tf","sf","zf","af","pf","cf"]
    regMatch = re.split(r'\ |\,|\+|\-|\:|\%|\$|\[|\*|\\|\@|\#|\$|\^|\^|\]', line)
    for itm in regMatch:
        if itm == "":
            continue
        if itm in registers:
            itm = "reg"
        lineUnified = lineUnified + " "+ str(itm)
    return lineUnified

def pointerAndMemlocUnifier(line):
    line = line.replace(" + ", "+").replace(" - ", "-").replace(" : ", ":").replace(" , ", ",")

    ptrMatches = re.findall(r"\[[\w+-*()#@$%^&|'\"]{1,}\]", line)
    for offs in ptrMatches:
        line = line.replace(offs, "ptr")

    lineMatch2 = re.findall(r",0x[a-zA-Z0-9]{8,}", line)
    for offs in lineMatch2:
        line = line.replace(offs, offs[0] + "offset")

    lineMatch1 = re.findall(r"0x[a-zA-Z0-9]{6,7}", line)
    for offs in lineMatch1:
        line = line.replace(offs, "memLoc")

    lineMatch3 = re.findall(r"[+-][a-zA-Z0-9]{1,}\]", line)
    for offs in lineMatch3:
        line = line.replace(offs, offs[0] + "offset]")

    lineMatch4 = re.findall(r",[0-9]{1,}", line)
    for offs in lineMatch4:
        line = line.replace(offs, "offset")

    lineMatch5 = re.findall(r":0x[a-zA-Z0-9]{1,}", line)
    for offs in lineMatch5:
        line = line.replace(offs, offs[0] + ":offset")
    
    return line
# jne 0x8048f62                         - call 0x4084e0, jmp 0x407d5b

def logCleaner(args):
    [logPath, mal, func, bits, feat] = args
    print(type(feat))
    t0 = time.time()
    logFile = logPath+mal+"/"+func
    f_str = ""
    instrList, templateList = [], []
    # fp = "/".join(logFile.rsplit('/')[-2:])
    # print("Now runninf for ", os.system("ls -lrth "+logFile))

    for line in pickle.load(open(logFile, "rb")):
        line = str(line)
        preline = line
        line = pointerAndMemlocUnifier(line)
        line = registerReferenceUnifier(line,bits)

        feat = pointerAndMemlocUnifier(feat)
        feat = registerReferenceUnifier(feat, bits)

        # print(preline, " ::: ", line)

        if feat.replace(" ", "").startswith(line.replace(" ", "")) and line not in templateList: #all the instructions that have the first instruction in the top feats are considered as possible starting component of the feat.
            # instrList, templateList, preline
            templateList.append(line)
            instrList.append(preline)
            print("------------ FIrst Entry ------------")
            print("Feature: ", feat)
            print("Template: ", line)
            print("Instruction: ", preline)
            print("---------------------------------")
            continue

        if len(templateList) == 0:
            continue

        # templateList_tmp = templateList
        # instrList_tmp = instrList
        # removeCnt = 0
        for idx in templateList.copy():
            instrID = instrList[templateList.index(idx)]
            if line.replace(" ", "") not in feat.replace(" ", ""):
                if len(feat.replace(" ", "")) > len(idx.replace(" ", "")):
                    print("------------ Deleted Entry ------------")
                    print("Feature: ", feat)
                    print("Template: ", idx)
                    print("Instruction: ", instrID)
                    print("This one not Found: ", line)
                    print("---------------------------------")
                    templateList.remove(idx)
                    instrList.remove(instrID)
                    # removeCnt+=1
                    print("template List after Deletion : ", templateList)

                else:
                    featDict = {feat: {"template": idx, "instruction": instrID, "function": mal}}
                    print("**********************************")
                    print("Final ::: ", featDict)
                    print("**********************************")
                    with open("../data/templateInstrFunctionMapping.log", 'a') as foo:
                        foo.write(str(featDict)+"\n")
                    return featDict
            else:
                idx = idx+"/"+line
                instrID = instrID + "/" + preline
                print("------------ Iterative Entry ------------")
                print("Feature: ", feat)
                print("Template: ", idx)
                print("Instruction: ", instrID)
                print("template List: ", templateList)
                print("---------------------------------")
        # templateList, instrList = templateList_tmp, instrList_tmp
    return f_str



def main():
    famFeats = ["dofloo-20000-finalConsideredFeatures-laters.pickle", "elknot-20001-finalConsideredFeatures-laters.pickle",
    "gafgyt-20000-finalConsideredFeatures-laters.pickle", "ganiw-20001-finalConsideredFeatures-laters.pickle",
    "generica-20000-finalConsideredFeatures-laters.pickle", "local-20000-finalConsideredFeatures-laters.pickle",
    "mirai-20000-finalConsideredFeatures-laters.pickle", "setag-20000-finalConsideredFeatures-laters.pickle",
    "tsunami-20000-finalConsideredFeatures-laters.pickle", "xorddos-20000-finalConsideredFeatures-laters.pickle"]

    outPath = '../data/templateFunctionMap/'
    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    for folder in famFeats:
        family = folder.rsplit("-")[0]
        topFeats = list(pickle.load(open("/home/mabuhamad/ma/afsah/data/featureAnalysis/topNfeats/withoutThreshold_rest/"+folder, "rb"))[:10])
        # f_strPath = "../../data/f_str-Pickles/"
        logPath = "../data/malwareFunctionDissembly/"
        bits = ""
        Arguments = []
        Results = []
        functionsConsidered = set()
        for feat in topFeats:
            feat = str(feat)
            # print(type(feat), type(str(feat)))
            for mal in os.listdir(logPath):
                for func in os.listdir(logPath+mal):
                    if func not in functionsConsidered:
                        Arguments.append([logPath, mal, func, bits, feat])
                        functionsConsidered.add(func)
                        # logPath = logPath+mal
                        # x = logCleaner([logPath+mal, bits, feat])
                        # Results.append(x)

        p = mp.Pool(processes=60)
        Results = p.map(logCleaner, Arguments)
        p.close()
        p.join()
        # print(corpus)
        with open(outPath+family+"-topFeatMap-templateInstrFunc.pickle", 'wb') as handle:
            pickle.dump(Results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    main()
