import os, time, pickle, re, sys, logging
import multiprocessing as mp

def registerReferenceUnifier(line):
    line = str(line)
    lineUnified = ""
    registers = ["rax","rcx","rdx","rbx","rsi","rdi","rsp","rbp","r8d","r9d","r10d","r11d","r12d",
    "r13d","r14d","r15d","r8w","r9w","r10w","r11w","r12w","r13w","r14w","r15w","r8b","r9b","r10b",
    "r11b","r12b","r13b","r14b","r15b","r8","r9","r10","r11","r12","r13","r14","r15","eax","ecx",
    "edx","ebx","esi","edi","esp","ebp","sil","dil","spl","bpl","ax","al","cx","cl","dx","dl","bx",
    "bl","cs","ds","es","fs","gs","ss","ah","bh","ch","dh","ip","sp","bp","si","di","of","df","if",
    "tf","sf","zf","af","pf","cf"]
    regMatch = re.split(r'\ |\,|\+|\-|\:|\%|\$|\[|\*|\\|\@|\#|\$|\^|\^|\]', line)
    for itm in regMatch:
        itm = itm.strip()
        if itm == "":
            continue
        if itm in registers:
            itm = "reg"
        if lineUnified == "":
            lineUnified = itm.strip()
        else:
            lineUnified = lineUnified + " "+ itm.strip()
    return lineUnified.strip()

def pointerAndMemlocUnifier(line):
    line = line.replace(" + ", "+").replace(" - ", "-").replace(" : ", ":").replace(" , ", ",")

    ptrMatches = re.findall(r"\[[\w+\-*() #@$%^&|'\"]{1,}\]", line)
    for offs in ptrMatches:
        line = line.replace(offs, "ptr")
    
    lineMatch2 = re.findall(r"0x[a-zA-Z0-9]{9,}", line)
    for offs in lineMatch2:
        line = line.replace(offs, offs[0] + "offset")

    lineMatch2 = re.findall(r"[,: ]0x[a-zA-Z0-9]{1,}", line)
    for offs in lineMatch2:
        line = line.replace(offs, offs[0] + "offset")

    lineMatch3 = re.findall(r"[+-, ][a-zA-Z0-9]{1,}\]", line)
    for offs in lineMatch3:
        line = line.replace(offs, offs[0] + "offset]")
    
    return line

def logCleaner(argArr):
    t0 = time.time()
    [logFile, logPath, outPath] = argArr
    fp = logFile.rsplit("/")[-1]
    f = pickle.load(open(logFile, "rb"))
    f_str = ""
    for line in f.rsplit("/"):
        line = line.replace("\n", "").strip()
        line = pointerAndMemlocUnifier(line)
        line = registerReferenceUnifier(line)

        f_str = f_str + "/\n" + line.strip()

    with open(outPath + str(fp) + '.pickle', 'wb') as handle:
        pickle.dump(f_str, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(logPath+"f_strDonesCorrection.txt", "a") as fstr:
        fstr.write(fp+",")
    print("ngram Count for ", fp, "is", len(f_str.rsplit("/")), "Finished in ", time.time() - t0, " seconds.")
    return f_str



def main(fstrPickleDir, logPath, outPath):
    if not os.path.isdir(logPath):
        if not os.path.isdir("../../data/Logs/"):
            os.mkdir("../../data/Logs/")
        os.mkdir(logPath)
    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    allLogs = set(os.listdir(fstrPickleDir))
    print("________________________")

    startTime = time.time()

    try:
        dones = set(open(logPath+"f_strDonesCorrection.txt", 'r').read().rsplit(','))
    except:
        dones = set()

    allLogs = allLogs - dones
    print("Length of 1st Next Run Set", "||", len(allLogs))
    print("-----------------")


    print("Starting run for log ", len(dones), "---", len(allLogs))
    setCount = len(dones)
    # jobs = []
    Arguments = []
    for fp in allLogs:
        if fp in dones:
            continue
        Arguments.append([fstrPickleDir+fp, logPath, outPath])

    print("Length of Arguments: ", len(Arguments))  
    print(Arguments[:10])  

    p = mp.Pool(processes=60)
    corpus = p.map(logCleaner, Arguments)
    p.close()
    p.join()


if __name__=="__main__":
    fstrPickleDir = "../../data/f_str-Pickles_Pre/"

    outPath = "../../data/f_str-Pickles/"
    logPath = "../../data/Logs/f_str-Pickles/"
    # splitter = int(sys.argv[1])
    # main(splitter)
    main(fstrPickleDir, logPath, outPath)