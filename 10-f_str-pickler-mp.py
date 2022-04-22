import os, time, pickle, re, sys, logging
import multiprocessing as mp

mpl = mp.log_to_stderr()
mpl.setLevel(logging.INFO)


def registerReferenceUnifier(line,bits):
    opcode, operands = line.rsplit(" ")[0], " ".join(line.rsplit(" ")[1:])
    if bits == 32:
        registers = ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "eip", "esp", "cs", "ds", "es", "fs", "gs", "ss"]
        regMatch = re.split(r'\ |\,|\+|\-|\:|\%|\$|\[|\]', operands)
        for itm in regMatch:
            if itm in registers:
                operands = operands.replace(itm, "reg")
            # operands = operands.replace(","+itm, ",reg")

    else:
        registers = ['rax','rcx','rdx','rbx','rsi','rdi','rsp','rbp','r8d','r9d','r10d','r11d','r12d','r13d','r14d','r15d','r8w','r9w','r10w','r11w','r12w','r13w','r14w','r15w','r8b','r9b','r10b','r11b','r12b','r13b','r14b','r15b','r8','r9','r10','r11','r12','r13','r14','r15','eax','ecx','edx','ebx','esi','edi','esp','ebp','sil','dil','spl','bpl','si','di','sp','bp','ax','al','cx','cl','dx','dl','bx','bl']
        regMatch = re.split(r'\ |\,|\+|\-|\:|\%|\$|\[|\]', operands)
        for itm in regMatch:
            if itm in registers:
                operands = operands.replace(itm, "reg")

    line = opcode+" "+operands
    return line

def logCleaner(argArr):
    [logFile, bits, outPath] = argArr
    t0 = time.time()

    f_str = ""
    fp = logFile.rsplit('/')[-1]
    print("Now runninf for ", os.system("ls -lrth "+logFile))

    for line in open(logFile, "r"):
        if bits == 32:
            if not line.startswith("0xc"):
                continue
        else:
            if not line.startswith("0xf"):
                continue
        line = ":".join(line.replace("\n", "").rsplit(":")[1:])
        line = " ".join([elem.strip() for elem in line.rsplit(" ") if elem.strip()])
        if "# 0xfff" in line:
            line = line.replace("# ", "#")
            line = " ".join(line.rsplit(" ")[:-1])

        # print("aaya")
        ptrMatches = re.findall(r"\[[\w+-]{1,}\]", line)
        for offs in ptrMatches:
            line = line.replace(offs, "ptr")

        lineMatch = re.findall(r"\[0x[a-zA-Z0-9]{1,}\]", line)
        for offs in lineMatch:
            line = line.replace(offs, "[memLoc]")

        if bits == 32:
            lineMatch1 = re.findall(r"0x[a-zA-Z0-9]{8}", line)
        else:
            lineMatch1 = re.findall(r"0x[a-zA-Z0-9]{16}", line)
        for offs in lineMatch1:
            line = line.replace(offs, "memLoc")

        lineMatch2 = re.findall(r",0x[a-zA-Z0-9]{1,}", line)
        for offs in lineMatch2:
            line = line.replace(offs, offs[0] + "offset")

        lineMatch3 = re.findall(r"[\+-][a-zA-Z0-9]{1,}\]", line)
        for offs in lineMatch3:
            line = line.replace(offs, offs[0] + "offset]")

        lineMatch4 = re.findall(r",[0-9]{1,}", line)
        for offs in lineMatch4:
            line = line.replace(offs, offs[0] + "offset")

        lineMatch5 = re.findall(r":0x[a-zA-Z0-9]{1,}", line)
        for offs in lineMatch5:
            line = line.replace(offs, offs[0] + ":offset")

        # print("Before:", line)
        line = registerReferenceUnifier(line,bits)
        # print("After:", line)

        if f_str == "":
            f_str = line
        else:
            f_str = f_str + "/\n" + line

        if len(f_str.rsplit("/")) > 250000:
            with open(outPath + str(fp) + '.pickle', 'wb') as handle:
                pickle.dump(f_str, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open("./runLogs/f_strDones.txt", "a") as fstr:
                fstr.write(fp + ",")
            with open("./runLogs/LogsWithGt200KLines.txt", "a") as fstr:
                fstr.write(fp + ",")
            print("ngram Count for ", fp, "is", len(f_str.rsplit("/")), "Finished in ", time.time() - t0, " seconds.")
            return f_str

    with open(outPath + str(fp) + '.pickle', 'wb') as handle:
        pickle.dump(f_str, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("./runLogs/f_strDones.txt", "a") as fstr:
        fstr.write(fp+",")
    print("ngram Count for ", fp, "is", len(f_str.rsplit("/")), "Finished in ", time.time() - t0, " seconds.")
    return f_str



def main():
    parentPath = "../data/dynamic/"
    dir = parentPath+"qemuLog/i386/"
    dir2 = parentPath+"qemuLog/x86/"

    outPath = parentPath + 'f_str-Pickles/'
    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    allLogs = set(os.listdir(dir)).union(set(os.listdir(dir2)))
    print("________________________")

    startTime = time.time()

    try:
        if not os.path.isdir("./runLogs/"):
            os.mkdir("./runLogs/")
        dones = set(open("./runLogs/f_strDones.txt", 'r').read().rsplit(','))
    except:
        dones = set()

    allLogs = allLogs - dones
    print("Length of allLogs: ", len(allLogs))

    nextRunSet = list(allLogs - dones)
    print(len(allLogs))
    print("Length of 1st Next Run Set", "||", len(nextRunSet))
    print("-----------------")


    print("Starting run for log ", len(dones), "---", len(nextRunSet))
    setCount = len(dones)
    # jobs = []
    Arguments = []
    for fp in nextRunSet:
        if not fp.endswith(".log"):
            continue
        if fp in dones:
            nextRunSet.remove(fp)
            continue

        if os.path.isfile(dir+fp):
            fileLoc = dir+fp
            bits = 32
        else:
            fileLoc = dir2+fp
            bits = 64

        Arguments.append([fileLoc, bits, outPath])


    p = mp.Pool(processes=60)
    for arg in Arguments:
        p.apply_async(logCleaner, args=(arg,))

    p.close()
    p.join()


if __name__=="__main__":
    # splitter = int(sys.argv[1])
    # main(splitter)
    main()