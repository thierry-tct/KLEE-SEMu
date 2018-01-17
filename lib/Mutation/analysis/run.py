#! /bin/bash

#######################
# This script takes as input:
# - The Strong Mutation matrix (passing this enable selecting by matrix)
# - The Mart output directory (passing this enable semu selection)
# - The test list file
# - The test running script
# - The path to executable in project
# - topDir for output (required)
########################
## TODO: implement runZesti

import os, sys
import json, re
import shutil, glob
import argparse
import random
import pandas as pd
import struct

# Other files
import matrixHardness
import rankSemuMutants
import analyse
import ktest_tool

OutFolder = "OUTPUT"
KleeSemuBCSuff = ".MetaMu.bc"
ZestiBCSuff = ".Zesti.bc"
WRAPPER_TEMPLATE = None
SEMU_CONCOLIC_WRAPPER = "wrapper-call-semu-concolic.in"
ZESTI_CONCOLIC_WRAPPER = "wrapper-call-zesti-concolic.in"
MY_SCANF = None
mutantInfoFile = "mutantsInfos.json"

def error_exit(errstr):
    print "\nERROR: "+errstr+'\n'
    assert False
    exit(1)

def dumpJson (data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def getTestSamples(testListFile, samplePercent, matrix):
    assert samplePercent > 0 and samplePercent <= 100, "invalid sample percent"
    samples = {}
    unwrapped_testlist = []
    with open(testListFile) as f:
        for line in f:
            t = line.strip()
            if t != "":
                unwrapped_testlist.append(t)
    
    with open(matrix) as f:
        p = re.compile('\s')
        testlist = p.split(f.readline().strip())[1:]
    # make samples for sizes samplePercent, 2*samplePercent, 3*samplePercent,..., 100
    random.shuffle(testlist)
    for s in range(samplePercent, 101, samplePercent):
        samples[s] = testlist[:int(s * len(testlist) / 100.0)]
        assert len(samples[s]) > 0, "too few test to sample percentage"
    return samples, testlist, unwrapped_testlist
#~ def getTestSamples()

def processMatrix (matrix, testSample, outname, thisOutDir):
    outFilePath = os.path.join(thisOutDir, outname)
    matrixHardness.libMain(matrix, testSample, outFilePath)
#~ def processMatrix()

def runZestiOrSemuTC (unwrapped_testlist, alltests, exePath, runtestScript, kleeZestiSemuInBCLink, semuworkdir, mode="zesti+symbex"):
    test2outdirMap = {}

    # Prepare outdir and copy bc
    if os.path.isdir(semuworkdir):
        shutil.rmtree(semuworkdir)
    os.mkdir(semuworkdir)
    kleeZestiSemuInBC = os.path.basename(kleeZestiSemuInBCLink) 
    if mode == "zesti+symbex":
        print "# Extracting tests Infos with ZESTI\n"
        cmd = "llvm-gcc -c -std=c89 -emit-llvm "+MY_SCANF+" -o "+MY_SCANF+".bc"
        ec = os.system(cmd)
        if ec != 0:
            error_exit("Error: failed to compile my_scanf to llvm for Zesti. Returned "+str(ec)+".\n >> Command: "+cmd)
        ec = os.system("llvm-link "+kleeZestiSemuInBCLink+" "+MY_SCANF+".bc -o "+os.path.join(semuworkdir, kleeZestiSemuInBC))
        if ec != 0:
            error_exit("Error: failed to link myscanf and zesti bc. Returned "+str(ec))
        os.remove(MY_SCANF+".bc")
    else:
        print "# Running SEMU Concretely\n"
        shutil.copy2(kleeZestiSemuInBCLink, os.path.join(semuworkdir, kleeZestiSemuInBC))

    # Install wrapper
    wrapper_fields = {
                        "IN_TOOL_DIR": semuworkdir,
                        "IN_TOOL_NAME": kleeZestiSemuInBC[:-3], #remove ".bc"
                        "TOTAL_MAX_TIME_": "7200.0", #600
                        "SOLVER_MAX_TIME_": "240.0"    #60
                     }
    ## Backup existing exe
    exePathBak = exePath+'.bak'
    if os.path.isfile(exePath):
        shutil.copy2(exePath, exePathBak)

    ## copy wraper to exe
    assert WRAPPER_TEMPLATE is not None
    assert os.path.isfile(WRAPPER_TEMPLATE)
    ### Load wrapper contain to memory
    with open(WRAPPER_TEMPLATE) as f:
        wrapContent = f.read()
    ### Replace all templates
    for t in wrapper_fields:
        wrapContent = wrapContent.replace(t, wrapper_fields[t])
    ### Write it as exe
    with open(exePath, 'w') as f:
        f.write(wrapContent)
    if exePathBak:
        # copy stats (creation time, ...)
        shutil.copystat(exePathBak, exePath)

    # Run Semu through tests running
    testrunlog = "> /dev/null"
    nKleeOut = len(glob.glob(os.path.join(semuworkdir, "klee-out-*")))
    assert nKleeOut == 0, "Must be no klee out in the begining"
    for tc in unwrapped_testlist:
        # Run Semu with tests (wrapper is installed)
        print "# Running Tests", tc, "..."
        retCode = os.system(" ".join(["bash", runtestScript, tc, testrunlog]))
        nNew = len(glob.glob(os.path.join(semuworkdir, "klee-out-*")))
        if nNew == nKleeOut:
            error_exit ("Test execution failed for test case '"+tc+"', retCode was: "+str(retCode))
        assert nNew > nKleeOut, "Test was not run: "+tc
        for devtid, kleetid in enumerate(range(nKleeOut, nNew)):
            kleeoutdir = os.path.join(semuworkdir, 'klee-out-'+str(kleetid))
            wrapTestName = os.path.join(tc.replace('/', '_') + "-out", "Dev-out-"+str(devtid), "devtest.ktest")

            test2outdirMap[wrapTestName] = kleeoutdir
        # update
        nKleeOut = nNew
    for wtc in alltests:
        assert wtc in test2outdirMap, "test not in Test2SemuoutdirMap: \nMap: "+str(test2outdirMap)+"\nTest: "+wtc
    return test2outdirMap
#~ def runZestiOrSemuTC()

'''
    serialize the ktest data(ktest_tool.KTest type) into a ktest file
'''
def ktestToFile(ktestData, outfilename):
    with open(outfilename, "wb") as f:
        f.write(b'KTEST')
        f.write(struct.pack('>i', ktestData.version))
        
        f.write(struct.pack('>i', len(ktestData.args)))
        for i in range(len(ktestData.args)):
            f.write(struct.pack('>i', len(ktestData.args[i])))
            f.write(ktestData.args[i].encode(encoding="ascii"))
        
        if ktestData.version > 2:
            f.write(struct.pack('>i', ktestData.symArgvs))
            f.write(struct.pack('>i', ktestData.symArgvLen))
        
        f.write(struct.pack('>i', len(ktestData.objects)))
        for i in range(len(ktestData.objects)):
            f.write(struct.pack('>i', len(ktestData.objects[i][0]))) #name length
            f.write(ktestData.objects[i][0])
            f.write(struct.pack('>i', len(ktestData.objects[i][1]))) #name length
            f.write(ktestData.objects[i][1])
    #print "Done ktest!"       
#~ def ktestToFile()

'''
    return a list representing the ordered list of argument 
    where each argument is represented by a pair of argtype (argv or file or stdin and the corresponding size)
'''
def parseTextKtest(filename):
    datalist = []
    b = ktest_tool.KTest.fromfile(filename)
    # get the object one ate the time and obtain its stuffs
    seenFileStatsPos = set()
    stdin = None
    model_version_pos = -1
    for i,(name,data) in enumerate(b.objects):
        if i in seenFileStatsPos:
            continue
        if i == 0:
            if name != "model_version":
                error_exit("The first argument in the ktest must be 'model_version'")
            else:
                model_version_pos = i
        else:
            # File passed
            if name in b.args:  # filename not in args, just verify that size is 0 (data is empty)
                if len(data) > 0:
                    error_exit ("object name in args but not a file: "+name) # beware of argument with value 'argv'
                # seach for its stat
                found = False
                for si,(sname,sdata) in enumerate(b.objects):
                    if sname == name+"-stat":
                        datalist.append(('FILE', len(sdata))) #XXX
                        seenFileStatsPos.add(si)
                        found = True
                        break
                if not found:
                    error_exit("File is not having stat in ktest")
            elif name == "stdin-stat": #case of stdin
                stdin = ('STDIN', len(data)) #XXX 
            else: #ARGV
                datalist.append(('ARGV', len(data))) #XXX
    if stdin is not None:
        datalist.append(stdin)

    # put 'model_version' last
    assert model_version_pos >= 0, "'model_version' not found in ktest file: "+filename
    b.objects.append(b.objects[model_version_pos])
    del b.objects[model_version_pos]

    return b, datalist
#~ def parseTextKtest()

def bestFit(outMaxVals, outNonTaken, inVals):
    assert len(inVals) <= len(outMaxVals)
    enabledArgs = [True] * len(outMaxVals)
    for i in range(len(inVals)):
        outMaxVals[i] = max(outMaxVals[i], inVals[i])
    for i in range(len(inVals), len(outMaxVals)):
        outNonTaken[i] = True
        enabledArgs[i] = False
    return enabledArgs
#~ def bestFit()

def getSymArgsFromKtests (pathCondFilesList): #, program='Expr'):
    #if program == 'Expr':
    #    symArgs = "--sym-args 0 1 10 --sym-args 0 3 2 --sym-stdout"
    #else:
    #    symArgs = "--sym-stdin 2"
    #    symArgs += " --sym-args 0 1 10 --sym-args 0 2 2 --sym-files 1 8 --sym-stdout"

    # XXX implement this. For program with file as parameter, make sure that the filenames are renamed in the path conditions
    listTestArgs = []
    ktestContains = {"FILENAME":[], "KTEST-OBJ":[]}
    for pcfile in pathCondFilesList:
        ktestfile = os.path.splitext(pcfile)[0] + '.ktest'
        # XXX Zesti do not generate valid Ktest file when an argument is the empty string. Example tests 'basic_s18' of EXPR which is: expr "" "|" ""
        # The reson is that when writing ktest file, klee want the name to be non empty thus it fail (I think). 
        # Thus, we skip such tests here
        if os.system(" ".join(['ktest-tool ', ktestfile, "> /dev/null 2>&1"])) != 0:
            print "@WARNING: Skipping test because Zesti generated invalid KTEST file:", ktestfile
            continue

        # sed because Zesti give argv, argv_1... while sym args gives arg0, arg1,...
        ktestdat, testArgs = parseTextKtest(ktestfile)
        listTestArgs.append(testArgs)
        ktestContains["FILENAME"].append(ktestfile)
        ktestContains["KTEST-OBJ"].append(ktestdat)
    if len(listTestArgs) <= 0:
        print "Err: no ktest data, ktest PCs:", pathCondFilesList
        error_exit ("No ktest data could be extracted from ktests.")

    # Make a general form out of listTestArgs by inserting what is needed with size 0
    # Make use of the sym-args param that can unset a param (klee care about param order)
    # Split each test args according to the FILE type (STDIN is always last), as follow: ARGV ARGV FILE ARGV FILE ...
    # then use -sym-args to flexibly set the number of enables argvs. First process the case before the first FILE, then between 1st and 2nd
    commonArgs = []
    commonArgsNumPerTest = {t: [] for t in range(len(listTestArgs))}
    while (True):
        testsCurFilePos = [0 for i in range(len(listTestArgs))]
        testsNumArgvs = [0 for i in range(len(listTestArgs))]
        # Find the next non ARGV argument for all tests
        for t in range(len(testsNumArgvs)):
            nonargvfound = False
            for a in range(testsCurFilePos[t], len(listTestArgs[t])):
                if listTestArgs[t][a][0] != "ARGV":
                    testsNumArgvs[t] = a - testsCurFilePos[t]
                    nonargvfound = True
                    break
            if not nonargvfound:
                testsNumArgvs[t] = len(listTestArgs[t]) - testsCurFilePos[t]
        # Rank test by num of ARGV args at this point
        indexes = range(len(testsNumArgvs))
        indexes.sort(reverse=True, key=lambda x: testsNumArgvs[x])
        maxArgNum = testsNumArgvs[indexes[0]]
        maxlens = [0 for i in range(maxArgNum)]
        canDisable = [False for i in range(maxArgNum)]
        if maxArgNum > 0:
            enabledArgs = {t: None for t in range(len(testsNumArgvs))}
            for tid in indexes:
                if testsNumArgvs[tid] == maxArgNum:
                    for pos,aid in enumerate(range(testsCurFilePos[tid], testsCurFilePos[tid] + testsNumArgvs[tid])):
                        maxlens[pos] = max(maxlens[pos], listTestArgs[tid][aid][1])
                    enabledArgs[tid] = [True] * maxArgNum
                else:
                    # make the best fit on existing sizes
                    enabledArgs[tid] = bestFit(maxlens, canDisable, [listTestArgs[tid][aid][1] for aid in range(testsCurFilePos[tid], testsCurFilePos[tid] + testsNumArgvs[tid])]) 
            
            # File related argument not cared about
            catchupS = len(commonArgs) - len(commonArgsNumPerTest[0])
            if catchupS > 0:
                for t in commonArgsNumPerTest:
                    commonArgsNumPerTest[t] += [None] * catchupS

            for i in range(len(maxlens)):
                if canDisable[i]:
                    arg = " ".join(["-sym-args 0 1", str(maxlens[i])])
                else:
                    arg = " ".join(["-sym-arg", str(maxlens[i])])
                # if previous is "-sym-args 0 <max-num> <size>" and arg is also "-sym-args 0 1 <size>", with same <size>, just update the previous
                if len(commonArgs) > 0 and commonArgs[-1].startswith("-sym-args 0 ") and commonArgs[-1].endswith(" "+str(maxlens[i])):
                    tmpsplit = commonArgs[-1].split(' ')
                    assert len(tmpsplit) == 4
                    tmpsplit[2] = str(int(tmpsplit[2]) + 1)
                    commonArgs[-1] = " ".join(tmpsplit)
                    for t in commonArgsNumPerTest:
                        commonArgsNumPerTest[t][-1] += int(enabledArgs[t][i])
                else:
                    commonArgs.append(arg)
                    for t in commonArgsNumPerTest:
                        commonArgsNumPerTest[t].append(int(enabledArgs[t][i]))

            # Update
            for t in range(len(testsNumArgvs)):
                testsCurFilePos[t] += testsNumArgvs[t]

        # Process non ARGV argument stdin or file argument
        fileMaxSize = -1
        stdinMaxSize = -1
        for t in range(len(testsNumArgvs)):
            # if the last arg was ARGV do nothing
            if testsCurFilePos[t] >= len(listTestArgs[t]):
                continue
            # If next is FILE
            if listTestArgs[t][testsCurFilePos[t]][0] == "FILE":
                fileMaxSize = max(fileMaxSize, listTestArgs[t][testsCurFilePos[t]][0])
                testsCurFilePos[t] += 1
            # If nex is STDIN (last)
            elif listTestArgs[t][testsCurFilePos[t]][0] == "STDIN":
                stdinMaxSize = max(stdinMaxSize, listTestArgs[t][testsCurFilePos[t]][0])
                #testsCurFilePos[t] += 1  # XXX Not needed since stdin is the last arg
            else:
                error_exit("unexpected arg type here: Neither FILE nor STDIN (type is "+listTestArgs[t][testsCurFilePos[t]][0]+")")
        if fileMaxSize >= 0:
            commonArgs.append(" ".join(["-sym-files 1", str(fileMaxSize)]))
        else:
            if stdinMaxSize >= 0:
                commonArgs.append(" ".join(["-sym-stdin", str(stdinMaxSize)]))
            break

    commonArgs.append('--sym-stdout')

    # TODO: UPDATE KTEST CONTAINS WITH NEW ARGUMENT LIST AND INSERT THE "n_args" FOR '-sym-args'. ALSO PLACE 'model_version' AT THE END
    # For each Test, Change ktestContains args, go through common args and compute the different "n_args" using 'listTestArgs'  and stdin stdout default.
    # Then write out the new KTEST that will be used to generate tests for mutants.
    
    # XXX; The objects are ordered here in a way they the ARGV come firts, then we have files, and finally model_version
    # File related argument not cared about
    catchupS = len(commonArgs) - len(commonArgsNumPerTest[0])
    if catchupS > 0:
        for t in commonArgsNumPerTest:
            commonArgsNumPerTest[t] += [None] * catchupS
    for t in commonArgsNumPerTest:
        objpos = 0
        for apos in range(len(commonArgsNumPerTest[t])):
            if commonArgs[apos].startswith("-sym-args "):
                assert commonArgsNumPerTest[t][apos] is not None
                if commonArgsNumPerTest[t][apos] > 0 and not ktestContains["KTEST-OBJ"][t].objects[objpos][0].startswith("argv"):
                    print "\nCommonArgs:", commonArgs
                    print "CommonArgsNumPerTest:", commonArgsNumPerTest[t]
                    print "Args:", ktestContains["KTEST-OBJ"][t].args[1:]
                    print "Objects:", ktestContains["KTEST-OBJ"][t].objects
                    error_exit("must be argv, but found: "+ktestContains["KTEST-OBJ"][t].objects[objpos][0])  # the name must be argv...

                # Insert n_args
                ktestContains["KTEST-OBJ"][t].objects.insert(objpos, ("n_args", str(commonArgsNumPerTest[t][apos])))       
                objpos += 1 #pass just added 'n_args'

                if commonArgsNumPerTest[t][apos] > 0: #Else no object for this, no need to advance this (will be done bellow)
                    objpos += commonArgsNumPerTest[t][apos]
            elif commonArgsNumPerTest[t][apos] is not None:  # is an ARGV non file
                if not ktestContains["KTEST-OBJ"][t].objects[objpos][0].startswith("argv"):
                    print "\nCommonArgs:", commonArgs
                    print "CommonArgsNumPerTest:", commonArgsNumPerTest[t]
                    print "Args:", ktestContains["KTEST-OBJ"][t].args[1:]
                    print "Objects:", ktestContains["KTEST-OBJ"][t].objects
                    error_exit("must be argv, but found: "+ktestContains["KTEST-OBJ"][t].objects[objpos][0])  # the name must be argv...
                objpos += 1

    # Write the new ktest files
    for i in range(len(ktestContains["FILENAME"])):
        shutil.move(ktestContains["FILENAME"][i], ktestContains["FILENAME"][i]+".bak")
        ktestToFile(ktestContains["KTEST-OBJ"][i], ktestContains["FILENAME"][i])

    return commonArgs
#~ getSymArgsFromKtests()

# put information from concolic run for the passed test set into a temporary dir, then possibly
# Compute SEMU symbex and rank according to SEMU. outpout in outFilePath
def processSemu (semuworkdir, testSample, test2semudirMap,  outname, thisOutDir, metaMutantBC, candidateMutantsFile,  mode="zesti+symbex"):
    outFilePath = os.path.join(thisOutDir, outname)
    tmpdir = semuworkdir+".tmp"
    #assert not os.path.isdir(tmpdir), "For Semu temporary dir already exists: "+tmpdir
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)

    # aggregated for the sample tests (semuTC mode)
    if mode == "zesti+symbex":
        symbexPreconditions = []
        for tc in testSample:
            tcdir = test2semudirMap[tc]
            for pathcondfile in glob.glob(os.path.join(tcdir, "*.pc")):
                symbexPreconditions.append(pathcondfile)
                # In th parth condition file, replace argv with arg: XXX temporary, DBG
                os.system(" ".join(["sed -i'' 's/argv_/arg/g; s/argv/arg0/g'", pathcondfile])) #DBG
        # use the collected preconditions and run semy in symbolic mode
        kleeArgs = "-allow-external-sym-calls -libc=uclibc -posix-runtime -search=bfs -solver-backend=stp -max-time=30000 -max-memory=9000 --max-solver-time=300"
        kleeArgs += " -max-sym-array-size=4096 --max-instruction-time=10. -watchdog -use-cex-cache"
        kleeArgs += " --output-dir="+tmpdir
        semuArgs = " ".join(["-semu-precondition-file="+prec for prec in symbexPreconditions]+["-semu-mutant-max-fork=4"])
        if candidateMutantsFile is not None:
            semuArgs += " -semu-candidate-mutants-list-file " + candidateMutantsFile
        symArgs = getSymArgsFromKtests (symbexPreconditions) #, program="Expr") #TODO : change program here
        print "\nSYM ARGS:\n", symArgs,"\n\n"
        exit(1)
        sretcode = os.system(" ".join(["klee-semu", kleeArgs, semuArgs, metaMutantBC, " ".join(symArgs), "> /dev/null"]))
        if sretcode != 0 and sretcode != 256: # 256 for tieout
            error_exit("Error: klee-semu symbex failled with code "+str(sretcode))
    else:
        os.mkdir(tmpdir)
        mutDataframes = {}
        for tc in testSample:
            tcdir = test2semudirMap[tc]
            for mutFilePath in glob.glob(os.path.join(tcdir, "mutant-*.semu")):
                mutFile = os.path.basename(mutFilePath)
                tmpdf = pd.read_csv(mutFilePath)
                if mutFile not in mutDataframes:
                    mutDataframes[mutFile] = tmpdf
                else:
                    mutDataframes[mutFile] = pd.concat([mutDataframes[mutFile], tmpdf])
        for mutFile in mutDataframes:
            aggrmutfilepath = os.path.join(tmpdir, mutFile)
            mutDataframes[mutFile].to_csv(aggrmutfilepath, index=False)

    # extract for Semu accordinc to sample
    rankSemuMutants.libMain(tmpdir, outFilePath)

    shutil.rmtree(tmpdir)
#~ def processSemu()

def analysis_plot(thisOut, groundKilledMutants):
    analyse.libMain(thisOut, mutantListForRandom=groundKilledMutants)
#~ def analysis_plot()


def main():
    global WRAPPER_TEMPLATE 
    global MY_SCANF
    runMode = "zesti+symbex" #semuTC
    #runMode = "semuTC"
    if runMode == "zesti+symbex":
        WRAPPER_TEMPLATE = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), ZESTI_CONCOLIC_WRAPPER))
        MY_SCANF = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "FixScanfForShadow/my_scanf.c"))
    else:
        WRAPPER_TEMPLATE = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), SEMU_CONCOLIC_WRAPPER))
    parser = argparse.ArgumentParser()
    parser.add_argument("outTopDir", help="topDir for output (required)")
    parser.add_argument("--exepath", type=str, default=None, help="The path to executable in project")
    parser.add_argument("--runtest", type=str, default=None, help="The test running script")
    parser.add_argument("--testlist", type=str, default=None, help="The test list file")
    parser.add_argument("--martout", type=str, default=None, help="The Mart output directory (passing this enable semu selection)")
    parser.add_argument("--matrix", type=str, default=None, help="The Strong Mutation matrix (passing this enable selecting by matrix)")
    args = parser.parse_args()

    outDir = os.path.join(args.outTopDir, OutFolder)
    exePath = args.exepath
    runtestScript = args.runtest
    testList = args.testlist
    martOut = args.martout
    matrix = args.matrix

    # get abs path in case not
    outDir = os.path.abspath(outDir)
    exePath = os.path.abspath(exePath) if exePath is not None else None 
    runtestScript = os.path.abspath(runtestScript) if runtestScript is not None else None 
    testList = os.path.abspath(testList) if testList is not None else None 
    martOut = os.path.abspath(martOut) if martOut is not None else None 
    matrix = os.path.abspath(matrix) if matrix is not None else None

    # Create outdir if absent
    cacheDir = os.path.join(outDir, "caches")
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
        os.mkdir(cacheDir)

    # We need to set size fraction of test samples
    testSamplePercent = 10

    # Get all test samples before starting experiment
    print "# Getting Test Samples .."
    testSamples, alltests, unwrapped_testlist = getTestSamples(testList, testSamplePercent, matrix) 
    dumpJson([testSamples, alltests, unwrapped_testlist], os.path.join(cacheDir, "testsamples.json"))

    candidateMutantsFile = None
    groundKilledMutants = None
    if matrix is not None:
        groundKilledMutants = matrixHardness.getKillableMutants(matrix) 
        candidateMutantsFile = os.path.join(cacheDir, "candidateMutants.list")
        with open(candidateMutantsFile, "w") as f:
            for mid in groundKilledMutants:
                f.write(str(mid)+"\n")

    # get Semu for all tests
    if martOut is not None:
        print "# Running Semu..."
        zestiInBC = os.path.basename(exePath) + ZestiBCSuff
        zestiInBCLink = os.path.join(martOut, zestiInBC)
        kleeSemuInBC = os.path.basename(exePath) + KleeSemuBCSuff
        kleeSemuInBCLink = os.path.join(martOut, kleeSemuInBC)
        semuworkdir = os.path.join(outDir, "SemuWorkDir")
        inBCFilePath = zestiInBCLink if runMode == "zesti+symbex" else kleeSemuInBCLink
        test2semudirMap = runZestiOrSemuTC (unwrapped_testlist, alltests, exePath, runtestScript, inBCFilePath, semuworkdir, mode=runMode) #mode can also be "semuTC"
        dumpJson(test2semudirMap, os.path.join(cacheDir, "test2semudirMap.json"))

    # processfor each test Sample
    for ts_size in testSamples:
        print "# Procesing for test size", ts_size, "..."

        # Make temporary outdir for test sample size
        thisOut = os.path.join(outDir, "out_testsize_"+str(ts_size))

        if martOut is not None or matrix is not None:
            if os.path.isdir(thisOut):
                shutil.rmtree(thisOut)
            os.mkdir(thisOut)

        # process for matrix
        if matrix is not None:
            processMatrix (matrix, alltests, 'groundtruth', thisOut) 
            processMatrix (matrix, testSamples[ts_size], 'classic', thisOut) 

        # process for SEMU
        if martOut is not None:
            processSemu (semuworkdir, testSamples[ts_size], test2semudirMap, 'semu', thisOut, kleeSemuInBCLink, candidateMutantsFile, mode=runMode) 

        # Make final Analysis and plot
        if martOut is not None and matrix is not None:
            analysis_plot(thisOut, groundKilledMutants) 

#~ def main()

if __name__ == "__main__":
    main()
