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
## TODO: (1) implement libMain for all three libs, (2) implement runSemu

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

def runSemu (unwrapped_testlist, alltests, exePath, runtestScript, kleeSemuInBC, kleeSemuInBCPath, semuworkdir):
    # TODO: 1) Install wrapper, 2) run Semu with all tests 3)
    test2outdirMap = {}

    for tc in alltests:
        #TODO
        test2outdirMap[tc] = kleeoutdir
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

    # We need to set size fraction of test samples
    testSamplePercent = 20

    # Get all test samples before starting experiment
    testSamples, alltests, unwrapped_testlist = getTestSamples(testList, testSamplePercent, matrix) 

    # get Semu for all tests
    if martOut is not None:
        kleeSemuInBC = os.path.basename(exePath) + KleeSemuBCSuff
        kleeSemuInBCPath = os.path.join(martOut, kleeSemuInBC)
        semuworkdir = os.path.join(outDir, "SemuWorkDir")
        test2semudirMap = runSemu (unwrapped_testlist, alltests, exePath, runtestScript, kleeSemuInBC, kleeSemuInBCPath, semuworkdir)

    # processfor each test Sample
    for ts_size in testSamples:
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
