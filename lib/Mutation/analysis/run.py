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
## TODO: implement runSemu

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
WRAPPER_TEMPLATE = None

def error_exit(errstr):
    print "\nERROR: "+errstr
    assert False
    exit(1)

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

def runSemu (unwrapped_testlist, alltests, exePath, runtestScript, kleeSemuInBCLink, semuworkdir):
    test2outdirMap = {}

    # Prepare outdir and copy bc
    if os.path.isdir(semuworkdir):
        shutil.rmtree(semuworkdir)
    os.mkdir(semuworkdir)
    kleeSemuInBC = os.path.basename(kleeSemuInBCLink) 
    shutil.copy2(kleeSemuInBCLink, os.path.join(semuworkdir, kleeSemuInBC))

    # Install wrapper
    wrapper_fields = {
                        "IN_TOOL_DIR": semuworkdir,
                        "IN_TOOL_NAME": kleeSemuInBC[:-3], #remove ".bc"
                        "TOTAL_MAX_TIME_": "600",
                        "SOLVER_MAX_TIME_": "60"
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
    for tc in unwrapped_testlist:
        # Run Semu with tests (wrapper is installed)
        print "# Running Tests", tc, "..."
        retCode = os.system(" ".join(["bash", runtestScript, tc, testrunlog]))
        if retCode not in [0,1]:
            error_exit ("Test execution failed for test case '"+tc+"', retCode was: "+str(retCode))
        nNew = len(glob.glob(os.path.join(semuworkdir, "klee-out-*")))
        assert nNew > nKleeOut, "Test was not run: "+tc
        for kleetid, devtid in enumerate(range(nKleeOut, nNew)):
            kleeoutdir = os.path.join(semuworkdir, 'klee-out-'+str(kleetid))
            wrapTestName = os.path.join(tc.replace('/', '_'), "Dev-out-"+str(devtid), "devtest.ktest")

            test2outdirMap[wrapTestName] = kleeoutdir
    for wtc in alltests:
        assert wtc in test2outdirMap, "test not int Test2SemuoutdirMap: "+wtc
    return test2outdirMap
#~ def runSemu()

def processSemu (semuworkdir, testSample, test2semudirMap,  outname, thisOut):
    outFilePath = os.path.join(thisOutDir, outname)
    tmpdir = semuworkdir+".tmp"
    assert not os.path.isdir(tmpdir), "For Semu temporary dir already exists: "+tmpdir

    # aggregated for the sample tests
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
        pd.to_csv(aggrmutfilepath, index=False)

    # extract for Semu accordinc to sample
    rankSemuMutants.libMain(tmpdir, outFilePath)

    shutil.rmtree(tmpdir)
#~ def processSemu()

def analysis_plot(thisOut):
    analyse.libMain(thisOut)
#~ def analysis_plot()


def main():
    global WRAPPER_TEMPLATE 
    WRAPPER_TEMPLATE = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "wrapper-call-semu-concolic.in"))
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
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    # We need to set size fraction of test samples
    testSamplePercent = 20

    # Get all test samples before starting experiment
    print "# Getting Test Samples .."
    testSamples, alltests, unwrapped_testlist = getTestSamples(testList, testSamplePercent, matrix) 

    # get Semu for all tests
    if martOut is not None:
        print "# Running Semu..."
        kleeSemuInBC = os.path.basename(exePath) + KleeSemuBCSuff
        kleeSemuInBCLink = os.path.join(martOut, kleeSemuInBC)
        semuworkdir = os.path.join(outDir, "SemuWorkDir")
        test2semudirMap = runSemu (unwrapped_testlist, alltests, exePath, runtestScript, kleeSemuInBCLink, semuworkdir)

    # processfor each test Sample
    for ts_size in testSamples:
        print "# Procesing for test size", ts_size, "..."

        # Make temporary outdir for test sample size
        thisOut = os.path.join(outDir, "out_testsize_"+str(ts_size))

        # process for matrix
        if matrix is not None:
            processMatrix (matrix, alltests, thisOut, 'groundtruth') 
            processMatrix (matrix, testSamples[ts_size], thisOut, 'classic') 

        # process for SEMU
        if martOut is not None:
            processSemu (semuworkdir, testSamples[ts_size], test2semudirMap, 'semu', thisOut) 

        # Make final Analysis and plot
        if martOut is not None and matrix is not None:
            analysis_plot(thisOut) 

#~ def main()

if __name__ == "__main__":
    main()
