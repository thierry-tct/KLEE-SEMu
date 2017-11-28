#! /usr/bin/python

# This file takes a mutant execution matrix as input, together with a file containing a subset of tests to consider
# Then rank the mutants according to harndness to kill w.r.t the test subset. relative equivalent are automatically deduced.
# This is similar to the file 'rankSemuMutants.py' for the output format.

import os, sys
import argparse
import re, json

def error_exit(errstr):
    print "  #CollectandAnalyse-Error:", errstr
    exit(1)
#~ error_exit()

PASS = ["0"]
FAIL = ["1"]
SM_index_string = "ktestSM"

def loadMatrix(matrixfile, testsfile, noKlee=False):
    dataAll = {}

    p = re.compile('\s')

    testname2testid = {}
    sortedTestNameList = []

    # read mutation matrix
    with open(matrixfile, 'r') as f:
        # read header
        tclist = p.split(f.readline().strip())
        assert tclist[0] == SM_index_string, "invalid SMdatfile: " + \
            matrixfile

        testid = 0 
        for tc in tclist[1:]:
            testname2testid[tc] = testid
            sortedTestNameList.append(tc)
            testid += 1
        dataAll[tclist[0]] = [testname2testid[testname]
                              for testname in tclist[1:]]  # header (test case list)

        # read data
        for line in f:
            # strip to remove leading and trailing space and avoid empty word
            a = p.split(line.strip())

            # strip pregram name, only keep mutant ID
            mID = int(os.path.basename(a[0]))

            assert (mID not in dataAll), "Mutant " + \
                a[0] + " appear twice in the dataAll"
            dataAll[mID] = a[1:]

    if noKlee:
        delpos = []
        for pos, tc in enumerate(dataAll[SM_index_string]):
            if os.path.basename(os.path.dirname(tc)).startswith("klee-out-"):
                delpos.append(pos)
        for pos in sorted(delpos, reverse=True):
            for tcmut in dataAll:
                del(dataAll[tcmut][pos])

    if testsfile is not None:
        selectedT = set()
        with open(testsfile) as ftp:
            for tcstr in ftp:
                tc = tcstr.strip()
                if tc:
                    selectedT.add(tc)
        assert len(selectedT) > 0, "empty tests subset file given"
        delpos = []
        for pos, tc in enumerate(dataAll[SM_index_string]):
            if sortedTestNameList[tc] not in selectedT:
                delpos.append(pos)
        for pos in sorted(delpos, reverse=True):
            for tcmut in dataAll:
                del(dataAll[tcmut][pos])
        assert selectedT == set([sortedTestNameList[tcid] for tcid in dataAll[SM_index_string]]), "tests mismatch... "+str((selectedT))+" <> "+str((dataAll[SM_index_string]))
    # print " ".join([matrixfile, "Loaded"])
    return dataAll #, sortedTestNameList
#~ def loadMatrix()

'''
    get the list of test that kill mutants
'''
def TestsKilling(mutant, dataAll):
    tests = set()

    for pos, pf in enumerate(dataAll[mutant]):
        if pf not in PASS:
            tests.add(dataAll[SM_index_string][pos])

    return list(tests)
#~ def TestsKilling()

def computeHardness(matrixdata):
    outData = {'Relative-Equivalent': [], 'Hardness': {}}
    nTests = len(matrixdata[SM_index_string])
    for mutID in set(matrixdata) - {SM_index_string}:
        #compute hardness as proportion of tests killing the mutant
        nKill = len(TestsKilling(mutID, matrixdata))

        if nKill == 0:
            outData['Relative-Equivalent'].append(mutID)
        else:
            m_hardness = 1 - float(nKill) / nTests
            
            # add mutant to output
            if m_hardness not in outData['Hardness']:
                outData['Hardness'][m_hardness] = list()
            outData['Hardness'][m_hardness].append(mutID)
    return outData
#~ def computeHardness()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mutantMatrix", help="input file of mutant execution as matrix (tests as column, mutants as rorw)")
    parser.add_argument("--testset", type=str, default=None, help="optional input file of subset of tests to consider")
    parser.add_argument("-o", "--outfile", type=str, default=None, help="JSON output file for mutant classement")
    args = parser.parse_args()

    matrixFile = args.mutantMatrix
    testSubSetFile = args.testset
    outFile = args.outfile

    assert (outFile is not None), "Must specify output file: option -o or --outfile"

    print "# Starting", matrixFile, "..."

    matrixKA = loadMatrix (matrixFile, testSubSetFile)

    outDataObj = computeHardness(matrixKA)

    with open(outFile+'.json', "w") as fp:
        json.dump(outDataObj, fp)

    print "# Done"
#~ def main()

if __name__ == "__main__":
    main()
