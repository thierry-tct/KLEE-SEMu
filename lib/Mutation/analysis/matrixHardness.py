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

def loadJson (filename):
    with open(filename) as f:
        return json.load(f)

def dumpJson (data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

PASS = ["0"]
FAIL = ["1"]
SM_index_string = "ktestSM"
WM_index_string = "ktestWM"
COVM_index_string = "ktestMCOV"

def loadMatrix(matrixfile, selectedT, X_index_string=SM_index_string, noKlee=False):
    dataAll = {}

    p = re.compile('\s')

    testname2testid = {}
    sortedTestNameList = []

    # read mutation matrix
    with open(matrixfile, 'r') as f:
        # read header
        tclist = p.split(f.readline().strip())
        assert tclist[0] == X_index_string, "invalid Matrix datfile ("+X_index_string+"): " + \
            matrixfile

        testid = 0 
        for tc in tclist[1:]:
            testname2testid[tc] = testid
            sortedTestNameList.append(tc)
            testid += 1
        dataAll[tclist[0]] = [testname2testid[testname]
                              for testname in tclist[1:]]  # header (test case list)

        nTotTests = len(dataAll[tclist[0]])

        # read data
        for line in f:
            # strip to remove leading and trailing space and avoid empty word
            a = p.split(line.strip())

            # strip pregram name, only keep mutant ID
            mID = int(os.path.basename(a[0]))

            assert (mID not in dataAll), "Mutant " + \
                a[0] + " appear twice in the dataAll"
            dataAll[mID] = a[1:]
            assert len(dataAll[mID]) == nTotTests, "uneven number od columns for mutant: "+str(mID)

    if noKlee:
        delpos = []
        for pos, tc in enumerate(dataAll[X_index_string]):
            if os.path.basename(os.path.dirname(tc)).startswith("klee-out-"):
                delpos.append(pos)
        for pos in sorted(delpos, reverse=True):
            for tcmut in dataAll:
                del(dataAll[tcmut][pos])

    if selectedT is not None:
        assert len(selectedT) > 0, "empty tests subset file given"
        delpos = []
        for pos, tc in enumerate(dataAll[X_index_string]):
            if sortedTestNameList[tc] not in selectedT:
                delpos.append(pos)
        for pos in sorted(delpos, reverse=True):
            for tcmut in dataAll:
                del(dataAll[tcmut][pos])
        assert selectedT == set([sortedTestNameList[tcid] for tcid in dataAll[X_index_string]]), "tests mismatch... "+str((selectedT))+" <> "+str((set([sortedTestNameList[tcid] for tcid in dataAll[X_index_string]])))
    
    # put back the test names ()useful for matching between SM and mutant Coverage
    for i in range(len(dataAll[X_index_string])):
        dataAll[X_index_string][i] = sortedTestNameList[dataAll[X_index_string][i]]

    # print " ".join([matrixfile, "Loaded"])
    return dataAll #, sortedTestNameList
#~ def loadMatrix()

def getFaultyTests(ktestPassFailFile, testset=None):
    passing = []
    failing = []
    with open(ktestPassFailFile) as f:
        for line in f:
            tc, pf = line.strip().split()
            if pf in PASS:
                passing.append(tc)
            else:
                failing.append(tc) 
    if testset is not None:
        passing = list(set(passing) & testset)
        failing = list(set(failing) & testset)
    return failing
#~ def getFaultyTests()

'''
    get the list of test that kill mutants
'''
def TestsKilling(mutant, dataAll, X_index_string=SM_index_string):
    tests = set()

    for pos, pf in enumerate(dataAll[mutant]):
        if pf not in PASS:
            tests.add(dataAll[X_index_string][pos])

    return list(tests)
#~ def TestsKilling()

def getKillableMutants(matrixFile, testset=None, mutantset=None, testkillinglist=None, minimal_testkillinglist=None):
    M = loadMatrix(matrixFile, None, SM_index_string)
    basetests = testset if testset is not None else set(M[SM_index_string])
    killablesMuts = []
    testskiling = set()
    test2killedmuts = {}
    cand_muts = set(M) - {SM_index_string}
    if mutantset is not None:
        cand_muts &= set(mutantset)
    for mid in cand_muts:
        tmp_kill_tests = basetests & set(TestsKilling(mid, M))
        if len(tmp_kill_tests) > 0:
            killablesMuts.append(mid)
            testskiling |= tmp_kill_tests
            for tc in tmp_kill_tests:
                if tc not in test2killedmuts:
                    test2killedmuts[tc] = set()
                test2killedmuts[tc].add(mid)

    if testkillinglist is not None:
        assert type(testkillinglist) == list
        assert len(testkillinglist) == 0
        testkillinglist.extend(list(testskiling))
    
    if minimal_testkillinglist is not None:
        minimal_testkilling = set()
        # use greedy algorithm
        all_muts = set()
        for tc in test2killedmuts:
            all_muts |= test2killedmuts[tc]
        mut_killed = set()
        visited = set()
        while mut_killed != all_muts:
            cand = None
            for tc in test2killedmuts.keys():
                if tc in minimal_testkilling:
                    continue
                if cand is None:
                    cand = tc
                else:
                    if len(mut_killed | test2killedmuts[tc]) > len(mut_killed | test2killedmuts[cand]):
                        cand = tc
            assert cand is not None, "Bug"
            minimal_testkilling.add(cand)
            mut_killed |= test2killedmuts[cand]
        minimal_testkillinglist.extend(list(minimal_testkilling))

    return killablesMuts
#~ def getKillableMutants()

def getUnKillableMutants(matrixFile, testset=None):
    M = loadMatrix(matrixFile, None, SM_index_string)
    basetests = testset if testset is not None else set(M[SM_index_string])
    unkillablesMuts = []
    for mid in  set(M) - {SM_index_string}:
        if len(basetests & set(TestsKilling(mid, M))) == 0:
            unkillablesMuts.append(mid)
    return unkillablesMuts
#~ def getUnKillableMutants()

def getListCoveredMutants(matrixFile, testset=None):
    M = loadMatrix(matrixFile, None, COVM_index_string)
    basetests = testset if testset is not None else set(M[SM_index_string])
    coveredMuts = []
    for mid in  set(M) - {COVM_index_string}:
        if len(basetests & set(TestsKilling(mid, M, COVM_index_string))) > 0:
            coveredMuts.append(mid)
    return coveredMuts
#~ def getListCoveredMutants()

'''
    Return a map of mutant and covering tests for mutants having more than thresh tests covering
    Do not return the mutants with number of covering tests cases less than threshold
'''
def getCoveredMutants(covMatFile, mutantInfo, fdupes_dup, testTresh_str='1'):
    M = loadMatrix(covMatFile, None, COVM_index_string)
    nTests = len(M[COVM_index_string])
    if testTresh_str[-1] == '%':
        percent = float(testTresh_str[:-1])
        assert (percent >= 0 and percent <= 100), "Invalid test coverage threshold percentage: "+str(percent)
        testTresh = int(percent * nTests / 100)
    else:
        testTresh = int(testTresh_str)
    covMuts = {}
    mutsincovmatrix = set(M) - {COVM_index_string}
    for mid in mutsincovmatrix:
        tccov = TestsKilling(mid, M, COVM_index_string)
        if len(tccov) >= testTresh:
            covMuts[mid] = tccov
    if testTresh == 0:  # even non covered mutants are considered
        mInfObj = loadJson(mutantInfo)
        if fdupes_dup is not None:
            fdupesObj = loadJson(fdupes_dup)
            # remove fdupes dups from mut info
            for remain_m in fdupesObj:
                for todel_m in fdupesObj[remain_m]:
                    del mInfObj[todel_m]
        for mid in set([int(m) for m in mInfObj.keys()]) - mutsincovmatrix:
            covMuts[mid] = []
    return covMuts
#~ def getCoveredMutants()

def computeHardness(matrixdata, mutantCovDict):
    outData = {'Relative-Equivalent': [], 'Hardness': {}}
    all_tests = set(matrixdata[SM_index_string])
    nTests = len(all_tests)
    
    # If mutantCovDict is None, consider hardness regardless of coverage: assume every test covers
    if mutantCovDict is None:
        candMutants = set(matrixdata) - {SM_index_string} 
        mutantCovDict = {mid: matrixdata[SM_index_string] for mid in candMutants}

    for mutID in mutantCovDict:
        #assert nTCCov > 0, "No test covering candidate mutant (BUG). Mutant ID: "+str(mitID)
        # compute hardness as proportion of tests killing the mutant among tests covering
        nTCCov = len(set(mutantCovDict[mutID]) & all_tests)
        nKill = len(TestsKilling(mutID, matrixdata))

        assert nTCCov <= nTests, "BUG: ncovtest > nTests: nTests="+str(nTests)+", nCovTests="+str(nTCCov)
        assert nKill <= nTests, "BUG: nkilltest > nTests: nTests="+str(nTests)+", nKillTests="+str(nKill)
        if nKill == 0 or nTCCov == 0: # The secon condition mus never be seen
            outData['Relative-Equivalent'].append(mutID)
        else:
            m_hardness = 1 - float(nKill) / nTCCov
            
            # add mutant to output
            if m_hardness not in outData['Hardness']:
                outData['Hardness'][m_hardness] = list()
            outData['Hardness'][m_hardness].append(mutID)
    return outData
#~ def computeHardness()

def libMain(mutantMatrixFile, testset, mutantCovDict, outfile):
    matrixKA = loadMatrix (mutantMatrixFile, set(testset))

    outDataObj = computeHardness(matrixKA, mutantCovDict)

    with open(outfile+'.json', "w") as fp:
        json.dump(outDataObj, fp)

    print "# Done"
#~ libMain()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mutantMatrix", help="input file of mutant execution as matrix (tests as column, mutants as rorw)")
    parser.add_argument("--testset", type=str, default=None, help="optional input file of subset of tests to consider")
    parser.add_argument("--coverage", type=str, default=None, help="optional mutant coverage matrix")
    parser.add_argument("-o", "--outfile", type=str, default=None, help="JSON output file for mutant classement")
    args = parser.parse_args()

    matrixFile = args.mutantMatrix
    coverage = args.coverage
    testSubSetFile = args.testset
    outFile = args.outfile

    assert (outFile is not None), "Must specify output file: option -o or --outfile"

    from_coverge = None
    if coverage is not None:
        assert False,"For coverage in command line to be implemented (choosing cov thresh)"

    print "# Starting", matrixFile, "..."

    selectedT = set()
    with open(testSubSetFile) as ftp:
        for tcstr in ftp:
            tc = tcstr.strip()
            if tc:
                selectedT.add(tc)

    libMain(matrixFile, selectedT, from_coverge, outFile)
#~ def main()

if __name__ == "__main__":
    main()
