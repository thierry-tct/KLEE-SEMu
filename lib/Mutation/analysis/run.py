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

# Other files
import matrixHardness
import rankSemuMutants
import analyse

OutFolder = "OUTPUT"
KleeSemuBCSuff = ".MetaMu.bc"
ZestiBCSuff = ".Zesti.bc"
WRAPPER_TEMPLATE = None
SEMU_CONCOLIC_WRAPPER = "wrapper-call-semu-concolic.in"
ZESTI_CONCOLIC_WRAPPER = "wrapper-call-zesti-concolic.in"
MY_SCANF = None

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
        cmd = "llvm-gcc -c -std=c89 -emit-llvm "+MY_SCANF+" -o "+MY_SCANF+".bc"
        ec = os.system(cmd)
        if ec != 0:
            error_exit("Error: failed to compile my_scanf to llvm for Zesti. Returned "+str(ec)+".\n >> Command: "+cmd)
        ec = os.system("llvm-link "+kleeZestiSemuInBCLink+" "+MY_SCANF+".bc -o "+os.path.join(semuworkdir, kleeZestiSemuInBC))
        if ec != 0:
            error_exit("Error: failed to link myscanf and zesti bc. Returned "+str(ec))
        os.remove(MY_SCANF+".bc")
    else:
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

def getSymArgsFromKtests (pathCondFilesList, program='Expr'):
    if program == 'Expr':
        symArgs = "--sym-args 0 1 10 --sym-args 0 3 2 --sym-stdout"
    else:
        symArgs = "--sym-stdin 2"
        symArgs += " --sym-args 0 1 10 --sym-args 0 2 2 --sym-files 1 8 --sym-stdout"

    # TODO implement this. For program with file as parameter, make sure that the filenames are renamed in the path conditions
    if False:
        for pcfile in pathCondFilesList:
            ktestfile = os.path.splitext(pcfile)[0] + '.ktest'
            ktesttoolfile = ktestfile + '.ktesttool'
            if os.system(" ".join(['ktest-tool --write-ints', ktestfile, ">", ktesttoolfile])) != 0:
                error_exit ("Failed to create ktesttool file for ktest file '"+ktestfile+"'")
            with open(ktesttoolfile) as fp:
                # TODO Extrac klee test information
                pass
            
            os.remove(ktesttoolfile)
        
        symArgs += '--sym-stdout'
    return symArgs
#~ getSymArgsFromKtests()

# put information from concolic run for the passed test set into a temporary dir, then possibly
# Compute SEMU symbex and rank according to SEMU. outpout in outFilePath
def processSemu (semuworkdir, testSample, test2semudirMap,  outname, thisOutDir, metaMutantBC, mode="zesti+symbex"):
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
        # use the collected preconditions and run semy in symbolic mode
        kleeArgs = "-allow-external-sym-calls -libc=uclibc -posix-runtime -search=bfs -solver-backend=stp -max-time=30000 -max-memory=9000 --max-solver-time=300"
        kleeArgs += " -max-sym-array-size=4096 --max-instruction-time=10. -watchdog -use-cex-cache"
        kleeArgs += " --output-dir="+tmpdir
        semuArgs = " ".join(["-semu-precondition-file="+prec for prec in symbexPreconditions]+["-semu-mutant-max-fork=4"])
        symArgs = getSymArgsFromKtests (symbexPreconditions, program="Expr") #TODO : change program here
        sretcode = os.system(" ".join(["klee-semu", kleeArgs, semuArgs, metaMutantBC, symArgs, "> /dev/null"]))
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

def analysis_plot(thisOut):
    analyse.libMain(thisOut)
#~ def analysis_plot()


def main():
    global WRAPPER_TEMPLATE 
    global MY_SCANF
    #runMode = "zesti+symbex" #semuTC
    runMode = "semuTC"
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
            processSemu (semuworkdir, testSamples[ts_size], test2semudirMap, 'semu', thisOut, kleeSemuInBCLink, mode=runMode) 

        # Make final Analysis and plot
        if martOut is not None and matrix is not None:
            analysis_plot(thisOut) 

#~ def main()

if __name__ == "__main__":
    main()
