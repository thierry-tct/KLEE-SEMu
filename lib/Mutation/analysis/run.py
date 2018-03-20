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

import os, sys, stat
import json, re
import shutil, glob
import argparse
import random
import pandas as pd
import struct
import itertools

from multiprocessing.pool import ThreadPool

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

KLEE_TESTGEN_SCRIPT_TESTS = "MFI_KLEE_TOPDIR_TEST_TEMPLATE.sh"

FilterHardToKill = "FilterHardToKill"
GenTestsToKill = "GenTestsToKill"

def error_exit(errstr):
    print "\nERROR: "+errstr+'\n'
    assert False
    exit(1)

def loadJson (filename):
    with open(filename) as f:
        return json.load(f)

def dumpJson (data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def getTestSamples(testListFile, samplePercent, matrix, discards={}, hasKleeTests=True):
    assert samplePercent >= 0 and samplePercent <= 100, "invalid sample percent"
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
    
    if hasKleeTests and KLEE_TESTGEN_SCRIPT_TESTS in unwrapped_testlist:
        unwrapped_testlist.remove(KLEE_TESTGEN_SCRIPT_TESTS)

    kleetestlist = []
    devtestlist = []
    for t in testlist:
        if t in discards:
            continue
        if t.startswith(KLEE_TESTGEN_SCRIPT_TESTS+"-out/klee-out-"):
            kleetestlist.append(t)
        else:
            devtestlist.append(t)

    # make samples for sizes samplePercent, 2*samplePercent, 3*samplePercent,..., 100
    if samplePercent > 0:
        random.shuffle(testlist)
        for s in [samplePercent]: #range(samplePercent, 101, samplePercent):
            samples[s] = testlist[:int(s * len(testlist) / 100.0)]
            assert len(samples[s]) > 0, "too few test to sample percentage"
    return samples, {'GENTESTS': kleetestlist, 'DEVTESTS':devtestlist}, unwrapped_testlist
#~ def getTestSamples()

def processMatrix (matrix, testSample, outname, candMutants_covTests, thisOutDir):
    outFilePath = os.path.join(thisOutDir, outname)
    matrixHardness.libMain(matrix, testSample, candMutants_covTests, outFilePath)
#~ def processMatrix()

def runZestiOrSemuTC (unwrapped_testlist, devtests, exePath, runtestScript, kleeZestiSemuInBCLink, outpdir, zestiexedir): #, mode="zesti+symbex"):
    test2outdirMap = {}

    # copy bc
    kleeZestiSemuInBC = os.path.basename(kleeZestiSemuInBCLink) 
    #if mode == "zesti+symbex":
    print "# Extracting tests Infos with ZESTI\n"
    cmd = "llvm-gcc -c -std=c89 -emit-llvm "+MY_SCANF+" -o "+MY_SCANF+".bc"
    ec = os.system(cmd)
    if ec != 0:
        error_exit("Error: failed to compile my_scanf to llvm for Zesti. Returned "+str(ec)+".\n >> Command: "+cmd)
    ec = os.system("llvm-link "+kleeZestiSemuInBCLink+" "+MY_SCANF+".bc -o "+os.path.join(outpdir, kleeZestiSemuInBC))
    if ec != 0:
        error_exit("Error: failed to link myscanf and zesti bc. Returned "+str(ec))
    os.remove(MY_SCANF+".bc")

    # Verify that Zesti is accessible
    zextmp = "klee"
    if zestiexedir is None:
       zextmp = os.path.join(zestiexedir, zextmp)
    if os.system(zextmp+" --help | grep zest > /dev/null") != 0:
        error_exit ("The available klee do not have Zesti: "+zextmp)
    #else:
    #    print "# Running SEMU Concretely\n"
    #    shutil.copy2(kleeZestiSemuInBCLink, os.path.join(outpdir, kleeZestiSemuInBC))

    # Install wrapper
    wrapper_fields = {
                        "ZESTI_KLEE_EXECUTABLE_DIRNAME/": zestiexedir+'/' if zestiexedir is not None else "",
                        "IN_TOOL_DIR": outpdir,
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
    assert os.path.isfile(exePath), "wrapper wasn't written as exe"

    if os.path.isfile(exePathBak):
        # copy stats (creation time, ...)
        shutil.copystat(exePathBak, exePath)
    else:
        # just make the wrapper executable
        st = os.stat(exePath)
        os.chmod(exePath, st.st_mode | stat.S_IEXEC)

    # Run Semu through tests running
    testrunlog = " > /dev/null" #+" 2>&1"
    nKleeOut = len(glob.glob(os.path.join(outpdir, "klee-out-*")))
    assert nKleeOut == 0, "Must be no klee out in the begining"
    for tc in unwrapped_testlist:
        # Run Semu with tests (wrapper is installed)
        print "# Running Tests", tc, "..."
        zestCmd = " ".join(["bash", runtestScript, tc, testrunlog])
        retCode = os.system(zestCmd)
        nNew = len(glob.glob(os.path.join(outpdir, "klee-out-*")))
        if nNew == nKleeOut:
            print ">> command: "+ zestCmd
            error_exit ("Test execution failed for test case '"+tc+"', retCode was: "+str(retCode))
        assert nNew > nKleeOut, "Test was not run: "+tc
        for devtid, kleetid in enumerate(range(nKleeOut, nNew)):
            kleeoutdir = os.path.join(outpdir, 'klee-out-'+str(kleetid))
            wrapTestName = os.path.join(tc.replace('/', '_') + "-out", "Dev-out-"+str(devtid), "devtest.ktest")

            test2outdirMap[wrapTestName] = kleeoutdir
        # update
        nKleeOut = nNew
    for wtc in devtests:
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
            f.write(struct.pack('>i', len(ktestData.objects[i][1]))) #data length
            f.write(ktestData.objects[i][1])
    #print "Done ktest!"       
#~ def ktestToFile()

'''
    return a list representing the ordered list of argument 
    where each argument is represented by a pair of argtype (argv or file or stdin and the corresponding size)
    IN KLEE, sym-files is taken into account only once (the last one)
'''
def parseZestiKtest(filename):

    datalist = []
    b = ktest_tool.KTest.fromfile(filename)
    # get the object one at the time and obtain its stuffs
    # the objects are as following: [model_version, <the argvs> <file contain, file stat>]
    # Note that files appear in argv. here we do not have n_args because concrete (from Zesti)

    # XXX Watch this: TMP: make sure all files are at the end
    firstFile_ind = -1
    postFileArgv_ind = []
    assert b.objects[0][0] == 'model_version'
    for pos, (name, data) in enumerate(b.objects[1:]): #skip model_version
        if name == 'argv':
            if firstFile_ind >= 0:
                postFileArgv_ind.append(pos+1)
        else:
            firstFile_ind = pos+1 if firstFile_ind < 0 else firstFile_ind
    if len(postFileArgv_ind) > 0:
        tmp_postFdat = []
        for ind_ in sorted(postFileArgv_ind, reverse=True):
            tmp_postFdat.append(b.objects[ind_])
            del b.objects[ind_]
        b.objects[firstFile_ind: firstFile_ind] = tmp_postFdat[::-1]
    #~

    # ZESTI (shadow) has issues handling forward slash(/) as argument. I thinks that it is a file while maybe not
    # XXX Fix that here. Since KLEE do not support directory it should be fine
    if '/' in b.args[1:]:
        for ind, (name,data) in enumerate(b.objects):
            if name == '/':
                if data == '\0'*4096:
                    assert b.objects[ind+1][0] == '/-stat', "Stat not following file"
                    del b.objects[ind:ind+2]
                else:
                    error_exit ("ERROR-BUG? data for forward slash not with data '\0'*4096 (zesti but workaround)")

    seenFileStatsPos = set()
    stdin = None
    model_version_pos = -1
    fileargsposinObj_remove = []
    filesNstatsIndex = []
    maxFileSize = -1
    for ind,(name,data) in enumerate(b.objects):
        if ind in seenFileStatsPos:
            continue
        if ind == 0:
            if name != "model_version":
                error_exit("The first argument in the ktest must be 'model_version'")
            else:
                model_version_pos = ind
        else:
            # File passed , In this case, there is: (1) an ARGV obj with data the filename, (2) an Obj with name the filename and data the file data, (3) an Obj for file stat (<filename>-stat) 
            indexes_ia = [i for i,x in enumerate(b.args[1:]) if x == name]
            if len(indexes_ia) > 0: # filename in args, the corresponding position in datalist is indexes_ia
                # in case the same name appears many times in args, let the user manually verify
                if len(indexes_ia) != 1:
                    print "\n>> CONFLICT: the file object at position ",ind,"with name",name,"in ktest",filename,"appears several times in args. please choose its positions (",indexes_ia,"):"
                    raw = raw_input()
                    indinargs = [int(v) for v in raw.split()]
                    assert len(set(indinargs) - set(indexes_ia)) == 0, "input wrond indexes. do not consider program name"
                else:
                    indinargs = indexes_ia

                for iv in indinargs:
                    datalist[iv] = ('ARGV', 1)  #XXX len is 1 because this will be the file name which is 1 char string in klee: 'A', 'B', ... ('A' + k | 0 <= k <= 255-'A')

                fileargsposinObj_remove.append(indinargs)  # ARGVs come before files in objects

                # seach for its stat, it should be the next object
                found = False
                if b.objects[ind + 1][0] == name+"-stat":
                    seenFileStatsPos.add(ind + 1)
                    found = True
                if not found:
                    error_exit("File is not having stat in ktest")

                filesNstatsIndex += [ind, ind+1]
                maxFileSize = max (maxFileSize, len(data))
            #elif name == "stdin-stat": #case of stdin
            #    stdin = ('STDIN', len(data)) #XXX 
            else: #ARGV
                assert name == "argv", "not in args and not argv: "+filename
                datalist.append(('ARGV', len(data))) #XXX

    if len(filesNstatsIndex) > 0:
        assert filesNstatsIndex == range(filesNstatsIndex[0], filesNstatsIndex[-1]+1), "File objects are not continuous: (File "+filename+"): "+str(filesNstatsIndex)+str(range(filesNstatsIndex[0], filesNstatsIndex[-1]+1))

    if model_version_pos == 0:
        #for ii in range(len(fileargsposinObj_remove)):
        #    for iii in range(len(fileargsposinObj_remove[ii])):
        #        fileargsposinObj_remove[ii][iii] += 1
        # Do bothing for fileargsposinObj_remove because already indexed to not account for model version XXX
        if len(fileargsposinObj_remove) > 0:
            assert max(fileargsposinObj_remove[-1]) < filesNstatsIndex[0], "arguments do not all come before files in object"
        filesNstatsIndex = [(v - 1) for v in filesNstatsIndex] #-1 for model_versio which will be move to end later

        # stdin and after files obj
        if stdin is not None:
            afterLastFilenstatObj = max(filesNstatsIndex) + 1 if len(filesNstatsIndex) > 0 else (len(b.objects) - 2 -1) # -2 because of stdin and stdin-stat, -1 for mode_version
            datalist.append(stdin)
        else:
            afterLastFilenstatObj = max(filesNstatsIndex) + 1 if len(filesNstatsIndex) > 0 else (len(b.objects) - 1) # -1 for model_version
            # shadow-zesti ay have problem with stdin, use our hack on wrappe to capture that
            stdin_file = os.path.join(os.path.dirname(filename), "stdin-ktest-data")
            assert os.path.isfile(stdin_file), "The stdin exported in wrapper is missing of test: "+filename
            with open(stdin_file) as f:
                sidat = f.read()
                if len(sidat) > 0:
                    symin_obj = ('stdin', sidat) #'\0'*1024)
                    syminstat_obj = ('stdin-stat', '\0'*144)
                    b.objects.append(symin_obj)
                    b.objects.append(syminstat_obj)
                    stdin = ('STDIN', len(sidat)) #XXX 
                    datalist.append(stdin)
    else:
        assert False, "model version need to be either 1st"  # For last, take care of putting stdin before it  or last object initially"

    # put 'model_version' last
    assert model_version_pos >= 0, "'model_version' not found in ktest file: "+filename
    b.objects.append(b.objects[model_version_pos])
    del b.objects[model_version_pos]
    return b, datalist, filesNstatsIndex, maxFileSize, fileargsposinObj_remove, afterLastFilenstatObj
#~ def parseZestiKtest()

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

class FileShortNames:
    def __init__(self):
        #schar = [chr(i) for i in range(ord('A'),ord('Z')+1)+range(ord('a'),ord('z')+1)]
        #ShortNames = {'names':[z[0]+z[1] for z in itertools.product(schar,['']+schar)], 'pos':0}
        self.ShortNames = [chr(i) for i in range(ord('A'), 256)]  # From KLEE: 'void klee_init_fds()' in runtime/POSIX/fd_init.c
        self.pos = 0
    def reinitialize_count (self):
        self.pos = 0
    def getShortname(self): 
        self.pos += 1
        if self.pos >= len(self.ShortNames):
            error_exit("too many file arguments, exeeded shortname list")
        return self.ShortNames[self.pos-1]
#~ class FileShortNames:

def getSymArgsFromZestiKtests (ktestFilesList, testNamesList):
    assert len(ktestFilesList) == len(testNamesList), "Error: size mismatch btw ktest and names: "+str(len(ktestFilesList))+" VS "+str(len(testNamesList))
    # XXX implement this. For program with file as parameter, make sure that the filenames are renamed in the path conditions(TODO double check)
    listTestArgs = []
    ktestContains = {"CORRESP_TESTNAME":[], "KTEST-OBJ":[]}
    maxFileSize = -1
    filenstatsInObj = []
    fileArgInd = []
    afterFileNStat = []
    for ipos, ktestfile in enumerate(ktestFilesList):
        # XXX Zesti do not generate valid Ktest file when an argument is the empty string. Example tests 'basic_s18' of EXPR which is: expr "" "|" ""
        # The reson is that when writing ktest file, klee want the name to be non empty thus it fail (I think). 
        # Thus, we skip such tests here TODO: remove thes from all tests so to have fair comparison with semu
        if os.system(" ".join(['ktest-tool ', ktestfile, "> /dev/null 2>&1"])) != 0:
            print "@WARNING: Skipping test because Zesti generated invalid KTEST file:", ktestfile
            continue

        # sed because Zesti give argv, argv_1... while sym args gives arg0, arg1,...
        ktestdat, testArgs, fileNstatInd, maxFsize, fileargind, afterFnS = parseZestiKtest(ktestfile)
        listTestArgs.append(testArgs)
        ktestContains["CORRESP_TESTNAME"].append(testNamesList[ipos])
        ktestContains["KTEST-OBJ"].append(ktestdat)
        filenstatsInObj.append(fileNstatInd)
        maxFileSize = max(maxFileSize, maxFsize)
        fileArgInd.append(fileargind)
        afterFileNStat.append(afterFnS)

    if len(listTestArgs) <= 0:
        print "Err: no ktest data, ktest PCs:", ktestFilesList
        error_exit ("No ktest data could be extracted from ktests.")

    # update file data in objects (shortname and size)
    nmax_files = max([len(fpv) for fpv in filenstatsInObj]) / 2 # divide by 2 beacause has stats
    if nmax_files > 0:
        shortFnames = FileShortNames().ShortNames[:nmax_files]
        for ktpos in range(len(ktestContains["CORRESP_TESTNAME"])):
            ktdat = ktestContains["KTEST-OBJ"][ktpos]

            # update file argument
            for ind_fai, fainds in enumerate(fileArgInd[ktpos]):
                for fai in fainds:
                    ktdat.objects[fai] = (ktdat.objects[fai][0], shortFnames[ind_fai])

            # first add file object of additional files
            addedobj = []
            for iadd in range(nmax_files - len(filenstatsInObj[ktpos])/2): # divide by two because also has stat
                symf_obj = ('', '\0'*maxFileSize)
                symfstat_obj = ('-stat', '\0'*144)
                addedobj.append(symf_obj)
                addedobj.append(symfstat_obj)
            insat = afterFileNStat[ktpos] #filenstatsInObj[ktpos][-1] + 1 if len(filenstatsInObj[ktpos]) > 0 else len(ktdat.objects)  # if no existing file, just append
            ktdat.objects[insat:insat] = addedobj
            filenstatsInObj[ktpos] += range(insat, insat+len(addedobj))

            # Now update the filenames and data
            for ni, fi_ in enumerate(range(0, len(filenstatsInObj[ktpos]), 2)):
                find_ = filenstatsInObj[ktpos][fi_]
                fsind_ = filenstatsInObj[ktpos][fi_ + 1]
                assert ktdat.objects[find_][0] + '-stat' == ktdat.objects[fsind_][0]
                ktdat.objects[find_] = (shortFnames[ni]+"-data", ktdat.objects[find_][1] + '\0'*(maxFileSize - len(ktdat.objects[find_][1]))) #file
                ktdat.objects[fsind_] = (shortFnames[ni]+"-data" + '-stat', ktdat.objects[fsind_][1]) #file

    # Make a general form out of listTestArgs by inserting what is needed with size 0
    # Make use of the sym-args param that can unset a param (klee care about param order)
    # Split each test args according to the FILE type (STDIN is always last), as follow: ARGV ARGV FILE ARGV FILE ...
    # then use -sym-args to flexibly set the number of enables argvs. First process the case before the first FILE, then between 1st and 2nd
    commonArgs = []
    commonArgsNumPerTest = {t: [] for t in range(len(listTestArgs))}
    testsCurFilePos = [0 for i in range(len(listTestArgs))]
    testsNumArgvs = [0 for i in range(len(listTestArgs))]
    symFileNameSize_ordered = []
    while (True):
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
#        fileMaxSize = -1
        fileMaxSize =  maxFileSize
        stdinMaxSize = -1
        for t in range(len(testsNumArgvs)):
            # if the last arg was ARGV do nothing
            if testsCurFilePos[t] >= len(listTestArgs[t]):
                continue
            # If next is FILE
#            if listTestArgs[t][testsCurFilePos[t]][0] == "FILE":
#                fileMaxSize = max(fileMaxSize, listTestArgs[t][testsCurFilePos[t]][1])
#                testsCurFilePos[t] += 1
            # If next is STDIN (last)
#            elif listTestArgs[t][testsCurFilePos[t]][0] == "STDIN":
            if listTestArgs[t][testsCurFilePos[t]][0] == "STDIN":
                stdinMaxSize = max(stdinMaxSize, listTestArgs[t][testsCurFilePos[t]][1])
                #testsCurFilePos[t] += 1  # XXX Not needed since stdin is the last arg
            else:
#                error_exit("unexpected arg type here: Neither FILE nor STDIN (type is "+listTestArgs[t][testsCurFilePos[t]][0]+")")
                error_exit("unexpected arg type here: Not STDIN (type is "+listTestArgs[t][testsCurFilePos[t]][0]+")")

        if fileMaxSize >= 0:
            commonArgs.append(" ".join(["-sym-files", str(nmax_files), str(fileMaxSize)]))
            symFileNameSize_ordered.append(fileMaxSize)
#        else:
        if stdinMaxSize >= 0:
            commonArgs.append(" ".join(["-sym-stdin", str(stdinMaxSize)]))
            # Update object's stdin size. add if not present
            
            for i in range(len(ktestContains["CORRESP_TESTNAME"])):
                siindex = len(ktestContains["KTEST-OBJ"][i].objects) - 1
                while siindex>=0 and ktestContains["KTEST-OBJ"][i].objects[siindex][0] != "stdin":
                    siindex -= 1
                if siindex >= 0:
                    assert ktestContains["KTEST-OBJ"][i].objects[siindex+1][0] == "stdin-stat", "stdin must be followed by its stats"
                    pre_si_dat = ktestContains["KTEST-OBJ"][i].objects[siindex][1]
                    ktestContains["KTEST-OBJ"][i].objects[siindex] = ('stdin', pre_si_dat + "\0"*(stdinMaxSize - len(pre_si_dat)))
                else:
                    symin_obj = ('stdin', '\0'*stdinMaxSize)
                    syminstat_obj = ('stdin-stat', '\0'*144)
                    ktestContains["KTEST-OBJ"][i].objects.insert(-1, symin_obj)
                    ktestContains["KTEST-OBJ"][i].objects.insert(-1, syminstat_obj)
            
        break

    # Sym stdout, is this really needed
    commonArgs.append('--sym-stdout')
    # add sym-out to ktets just before the last (model_version)
    for i in range(len(ktestContains["CORRESP_TESTNAME"])):
        symout_obj = ('stdout', '\0'*1024)
        symoutstat_obj = ('stdout-stat', '\0'*144)
        ktestContains["KTEST-OBJ"][i].objects.insert(-1, symout_obj)
        ktestContains["KTEST-OBJ"][i].objects.insert(-1, symoutstat_obj)


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

                # Pad the argument data with '\0' until args maxlen
                maxlen = int(commonArgs[apos].strip().split(' ')[-1])
                for sharedarg_i in range(commonArgsNumPerTest[t][apos]):
                    curlen = len(ktestContains["KTEST-OBJ"][t].objects[objpos + sharedarg_i][1])
                    curval = ktestContains["KTEST-OBJ"][t].objects[objpos + sharedarg_i]
                    ktestContains["KTEST-OBJ"][t].objects[objpos + sharedarg_i] = (curval[0], curval[1]+'\0'*(maxlen-curlen+1)) #+1 Because last zero added after sym len

                # Insert n_args
                ktestContains["KTEST-OBJ"][t].objects.insert(objpos, ("n_args", struct.pack('<i', commonArgsNumPerTest[t][apos])))       
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

                # Pad the argument data with '\0' until args maxlen
                maxlen = int(commonArgs[apos].strip().split(' ')[-1])
                curlen = len(ktestContains["KTEST-OBJ"][t].objects[objpos][1])
                curval = ktestContains["KTEST-OBJ"][t].objects[objpos]
                ktestContains["KTEST-OBJ"][t].objects[objpos] = (curval[0], curval[1]+'\0'*(maxlen-curlen+1)) #+1 Because last zero added after sym len
                
                objpos += 1
            else:  # File or stdin, stdout
                pass #TODO handle the case of files (NB: check above how the files are recpgnized from zesti tests (size may not be 0)

    # Change the args list in each ktest object with the common symb args 
    for ktdat in ktestContains["KTEST-OBJ"]:
        ktdat.args = ktdat.args[:1]
        for s in commonArgs:
            ktdat.args += s.strip().split(' ') 

    return commonArgs, ktestContains
#~ getSymArgsFromZestiKtests()

'''
    the list of klee tests (ktest names) are il ktestsList. 
    Each represent the location of the ktest w.r.t. teststopdir
'''
def loadAndGetSymArgsFromKleeKTests(ktestsList, teststopdir):
    # load the ktests
    commonArgs = None
    tmpsplitcommon = None
    stdin_fixed_common = None
    ktestContains = {"CORRESP_TESTNAME":[], "KTEST-OBJ":[]}
    for kt in ktestsList:
        ktestfile = os.path.join(teststopdir, kt)
        assert os.path.isfile(ktestfile), "Ktest file for test is missing :" + ktestfile

        b = ktest_tool.KTest.fromfile(ktestfile)
        if commonArgs is None:
            commonArgs = [] 
            # make chunk (sym-args together with its params)
            anArg = None
            tmpsplitcommon = b.args[1:]
            for c in tmpsplitcommon:
                if c.startswith('-sym') or c.startswith('--sym'):
                    if anArg is not None:
                        commonArgs.append(anArg)
                    anArg = c
                else:
                    anArg += " " + c
            if anArg is not None:
                commonArgs.append(anArg)
    
            # XXX fix problem with klee regarding stdin
            a_has_stdin = False
            for sa in commonArgs:
                if '-sym-stdin ' in sa:
                    a_has_stdin = True
                    break
            if not a_has_stdin:
                o_stdin_len = -1
                for o in b.objects:
                    if o[0] == 'stdin':
                        o_stdin_len = len(o[1])
                        break
                if o_stdin_len >= 0:
                    si_pos = len(commonArgs)-1 if '-sym-stdout' in commonArgs[-1] else len(commonArgs)
                    commonArgs.insert(si_pos, "-sym-stdin "+str(o_stdin_len))
                    stdin_fixed_common = []
                    for s_a in commonArgs:
                        stdin_fixed_common += s_a.split()
            # make sure that model_version is the last object
            assert b.objects[-1][0] == "model_version", "The last object is not 'model_version' in klee's ktest"
        else:
            assert tmpsplitcommon == b.args[1:], "Sym Args are not common among KLEE's ktests"
            # handle case when problem with klee and stdin
            if stdin_fixed_common is not None:
                b.args[1:] = list(stdin_fixed_common)

        ktestContains["KTEST-OBJ"].append(b)
        ktestContains["CORRESP_TESTNAME"].append(kt)

    return commonArgs, ktestContains
#~ def loadAndGetSymArgsFromKleeKTests()

'''
    Use old and new from argv_old_new to update the argv in ktestContain
    Sequence in ktestobject: 1) argv(arg<i>), 2) [<ShortName>-data, <ShortName>-data-stat], 3) [stdin, stdin-stat], 4) [stdout, stdout-stat], 5) model_version
'''
def updateObjects(argvinfo, ktestContains):
    def isCmdArg(name):
        if name == 'n_args' or name.startswith('argv') or name.startswith('arg'):
            return True
        return False
    #~ def isCmdArg()

    # list_new_sym_args = [[<nmin>,<nmax>,<size>],...]
    def old2new_cmdargs(objSegment, list_new_sym_args):
        res = []
        nargs = len(objSegment) - 1 # First obj is n_args object
        nums = [x[0] for x in list_new_sym_args]
        n_elem = sum(nums)
        assert n_elem <= nargs , "min sum do not match to nargs. n_args="+str(nargs)+", min sum="+str(n_elem)
        for i in range(len(nums))[::-1]:
            rem = nargs - n_elem
            if rem <= 0:
                break
            inc = min(rem, list_new_sym_args[i][1] - nums[i])
            n_elem += inc
            nums[i] += inc
        assert n_elem == nargs, "n_elem must be equal to nargs here. Got: "+str(n_elem)+" VS "+str(nargs)

        # put elements in res according to nums
        ao_ind = 1
        for i,n in enumerate(nums):
            res.append(("n_args", struct.pack('<i', n)))
            for j in range(ao_ind, ao_ind+n):
                res.append((objSegment[j][0], objSegment[j][1] + '\0'*(list_new_sym_args[i][2] - len(objSegment[j][1]) + 1)))
            ao_ind += n
        assert ao_ind == len(objSegment)
        return res
    #~ def old2new_cmdargs()

    assert len(argvinfo['old']) == len(argvinfo['new'])

    argvinfo_new_extracted = []
    for vl in argvinfo['new']:
        argvinfo_new_extracted.append([])
        for v in vl:
            kw, nmin, nmax, size = v.split()        #assuming all args in new are sym-args (no sym-arg)
            argvinfo_new_extracted[-1].append((int(nmin), int(nmax), int(size)))

    for okt in ktestContains['KTEST-OBJ']:
        kt_obj = okt.objects
        # First process sym-files, sym-stdin and sym-stdout
        pointer = -1 #model_version
        # sym-stdout
        if argvinfo['sym-std']['out-present']:
            if not argvinfo['sym-std']['out-present-pre']:
                kt_obj.insert(pointer, ('stdout', '\0'*1024))
                kt_obj.insert(pointer, ('stdout-stat', '\0'*144))
            pointer -= 2
        
        # sym-stdin
        if argvinfo['sym-std']['in-present']:
            if not argvinfo['sym-std']['in-present-pre']:
                kt_obj.insert(pointer, ('stdin', '\0'*argvinfo['sym-std']['in-size']))
                kt_obj.insert(pointer, ('stdin-stat', '\0'*144))
                pointer -= 2
            elif argvinfo['sym-std']['in-size-pre'] != argvinfo['sym-std']['in-size']:
                stdin_stat_pos = pointer - 1 # first 2 because model_version is at position -1 and secaon because of stdout-stat
                stdin_pos = stdin_stat_pos - 1
                assert kt_obj[stdin_pos][0] == 'stdin', "Expected stdin as object just before stdout and model_version"
                kt_obj[stdin_pos] = (kt_obj[stdin_pos][0], kt_obj[stdin_pos][1] + '\0'*(argvinfo['sym-std']['in-size'] - argvinfo['sym-std']['in-size-pre']))
            pointer -= 2
       
        # sym-files
        if argvinfo['sym-files']['nFiles'] > 0:
            if argvinfo['sym-files']['nFiles-pre'] > 0 and argvinfo['sym-files']['size-pre'] != argvinfo['sym-files']['size']:
                for f_p in range(pointer - 2*argvinfo['sym-files']['nFiles-pre'], pointer, 2):
                    assert not isCmdArg(kt_obj[f_p][0]), "Expected sym file object, but found cmd arg: Pos="+str(f_p)+", object="+str(kt_obj[f_p])
                    kt_obj[f_p] = (kt_obj[f_p][0], kt_obj[f_p][1] + '\0'*(argvinfo['sym-files']['size'] - argvinfo['sym-files']['size-pre']))

            if argvinfo['sym-files']['nFiles-pre'] != argvinfo['sym-files']['nFiles']:
                # add empty file at position
                snames = FileShortNames().ShortNames[argvinfo['sym-files']['nFiles-pre']:argvinfo['sym-files']['nFiles']]
                for sn in snames:
                    kt_obj.insert(pointer, (sn+'-data', '\0'*argvinfo['sym-files']['size']))
                    kt_obj.insert(pointer, (sn+'-data'+'-stat', '\0'*144))
            pointer -= 2*argvinfo['sym-files']['nFiles']
        
        #sym cmdargv(arg)
        obj_ind = 0
        for arg_ind in range(len(argvinfo['old'])):
            if argvinfo['old'][arg_ind] is None: #added ones, all are -sym-args, just add n_args=0
                for sa in argvinfo['new'][arg_ind]:
                    kt_obj.insert(obj_ind, ("n_args", struct.pack('<i', 0)))
                    obj_ind += 1
            else:
                assert isCmdArg(kt_obj[obj_ind][0])
                if '-sym-arg ' in argvinfo['old'][arg_ind]:
                    assert len(argvinfo['new'][arg_ind]) == 1, "must be on sym-args here"
                    kt_obj.insert(obj_ind, ("n_args", struct.pack('<i', 1)))
                    obj_ind += 2 #Go after n_args and argv(arg)
                else: #sym-args
                    assert kt_obj[obj_ind][0] == 'n_args', "must ne n_args here"
                    nargs = struct.unpack('<i', kt_obj[obj_ind][1])[0]
                    if argvinfo['old'][arg_ind] != argvinfo['new'][arg_ind]:
                        # put the args enabled as match to the right most in new
                        tmppos_last = obj_ind + nargs 
                        replacement = old2new_cmdargs(kt_obj[obj_ind:tmppos_last+1], argvinfo_new_extracted[arg_ind])
                        kt_obj[obj_ind:tmppos_last+1] = replacement
                        obj_ind += len(replacement)
                    else:
                        obj_ind += nargs + 1 #+1 for n_args obj

        assert len(kt_obj) + pointer == obj_ind, "Some argv objects were not considered? ("+str(len(kt_obj))+", "+str(pointer)+", "+str(obj_ind)+")"
#~ def updateObjects()

'''
'''
def mergeZestiAndKleeKTests (outDir, ktestContains_zest, commonArgs_zest, ktestContains_klee, commonArgs_klee):
    commonArgs = []
    name2ktestMap = {}
    ktestContains = {"CORRESP_TESTNAME":[], "KTEST-OBJ":[]}

    if ktestContains_zest is None:
        assert commonArgs_zest is None
        assert ktestContains_klee is not None and commonArgs_klee is not None
        ktestContains = ktestContains_klee
        commonArgs = commonArgs_klee
    elif ktestContains_klee is None:
        assert commonArgs_klee is None
        assert ktestContains_zest is not None and commonArgs_zest is not None
        ktestContains = ktestContains_zest
        commonArgs = commonArgs_zest
    else:
        def getSymArgvParams(symargstr):
            outdict = {}
            tz = symargstr.split()
            if len(tz) == 2: #sym-arg
                outdict['min'] = outdict['max'] = 1
                outdict['size'] = int(tz[-1])
            else: #sym-args
                outdict['min'] = int(tz[-3])
                outdict['max'] = int(tz[-2])
                outdict['size'] = int(tz[-1])
            return outdict


        assert commonArgs_zest is not None and commonArgs_klee is not None
        # Common Args must have either sym-arg or sym-args or sym-files or sym-stdin or sym-stdout
        for common in [commonArgs_zest, commonArgs_klee]:
            for a in common:
                if "-sym-arg " in a or "-sym-args " in a or "-sym-files " in a or "-sym-stdin " in a or "-sym-stdout" in a:
                    continue
                error_exit ("Unsupported symbolic argument: "+a+", in "+str(common))

        # create merged commonArgs and a map between both zest and klee commonargs to new commonargs
        argv_zest = {'old':[x for x in commonArgs_zest if "-sym-arg" in x], 'new':[]}
        argv_zest['new'] = [[] for x in argv_zest['old']]
        argv_klee = {'old':[x for x in commonArgs_klee if "-sym-arg" in x], 'new':[]}
        argv_klee['new'] = [[] for x in argv_klee['old']]
        z_ind = 0
        k_ind = 0
        z_cur_inf = None
        k_cur_inf = None
        while True:
            if z_ind < len(argv_zest['old']) and k_ind < len(argv_klee['old']):
                if z_cur_inf is None:
                    z_cur_inf = getSymArgvParams (argv_zest['old'][z_ind])
                if k_cur_inf is None:
                    k_cur_inf = getSymArgvParams (argv_klee['old'][k_ind])
                m_min = min(z_cur_inf['min'], k_cur_inf['min'])
                m_max = min(z_cur_inf['max'], k_cur_inf['max'])
                M_size = max(z_cur_inf['size'], k_cur_inf['size'])
                newarg = " ".join(["-sym-args", str(m_min), str(m_max), str(M_size)])

                # add the new args to each as new, add also to commonArgs
                commonArgs.append(newarg)
                argv_zest['new'][z_ind].append(newarg)
                argv_klee['new'][k_ind].append(newarg)

                # Update the cur_infs and index
                if z_cur_inf['max'] == m_max:
                    z_cur_inf = None
                    z_ind += 1
                else:
                    z_cur_inf['min'] = max(0, z_cur_inf['min'] - m_max)
                    z_cur_inf['max'] = z_cur_inf['max'] - m_max
                if k_cur_inf['max'] == m_max:
                    k_cur_inf = None
                    k_ind += 1
                else:
                    k_cur_inf['min'] = max(0, k_cur_inf['min'] - m_max)
                    k_cur_inf['max'] = k_cur_inf['max'] - m_max

            else:
                # handle inequalities
                if z_ind < len(argv_zest['old']):
                    argv_klee['old'].append(None)
                    argv_klee['new'].append([])
                    if z_cur_inf is not None:
                        newarg = " ".join(["-sym-args", str(z_cur_inf['min']), str(z_cur_inf['max']), str(z_cur_inf['size'])])
                        commonArgs.append(newarg)
                        argv_zest['new'][z_ind].append(newarg)
                        argv_klee['new'][k_ind].append(newarg)
                        z_ind += 1
                    ineqs = [a.replace('sym-arg ', 'sym-args 0 1 ') for a in argv_zest['old'][z_ind:]] #replace sym-arg with sym-args 0 1 because not present in other
                    commonArgs += ineqs
                    argv_zest['new'][z_ind:] = [ineqs]
                    argv_klee['new'][k_ind] += ineqs
                    break
                if k_ind < len(argv_klee['old']):
                    argv_zest['old'].append(None)
                    argv_zest['new'].append([])
                    if k_cur_inf is not None:
                        newarg = " ".join(["-sym-args", str(k_cur_inf['min']), str(k_cur_inf['max']), str(k_cur_inf['size'])])
                        commonArgs.append(newarg)
                        argv_klee['new'][k_ind].append(newarg)
                        argv_zest['new'][z_ind].append(newarg)
                        k_ind += 1
                    ineqs = [a.replace('sym-arg ', 'sym-args 0 1 ') for a in argv_klee['old'][k_ind:]] #replace sym-arg with sym-args 0 1 because not present in other
                    commonArgs += ineqs
                    argv_klee['new'][k_ind:] = [ineqs]
                    argv_zest['new'][z_ind] += ineqs
                    break
                break

        # get sym-stdin, sym-stdout and symfiles
        argv_zest["sym-files"] = {'nFiles-pre':0, 'nFiles':0, 'size-pre':0, 'size':0}
        argv_klee["sym-files"] = {'nFiles-pre':0, 'nFiles':0, 'size-pre':0, 'size':0}
        argv_zest["sym-std"] = {'in-present-pre': False, 'in-size-pre':0, 'out-present-pre':False, 'in-present': False, 'in-size':0, 'out-present':False}
        argv_klee["sym-std"] = {'in-present-pre': False, 'in-size-pre':0, 'out-present-pre':False, 'in-present': False, 'in-size':0, 'out-present':False}
        for common, s_argv in [(commonArgs_zest, argv_zest), (commonArgs_klee, argv_klee)]:
            # klee considers the last sym-files and stdin so we just check from begining to end
            for a in common:
                if '-sym-files ' in a:
                    unused, n_f, f_s = a.split()
                    s_argv["sym-files"]["nFiles-pre"] = int(n_f)
                    s_argv["sym-files"]["size-pre"] = int(f_s)
                elif '-sym-stdin ' in a:
                    s_argv['sym-std']['in-present-pre'] = True
                    s_argv['sym-std']['in-size-pre'] = int(a.split()[-1])
                elif '-sym-stdout' in a:
                    s_argv['sym-std']['out-present-pre'] = True
        argv_zest['sym-files']['nFiles'] = argv_klee['sym-files']['nFiles'] = max(argv_zest['sym-files']['nFiles-pre'], argv_klee['sym-files']['nFiles-pre'])
        argv_zest['sym-files']['size'] = argv_klee['sym-files']['size'] = max(argv_zest['sym-files']['size-pre'], argv_klee['sym-files']['size-pre'])
        argv_zest['sym-std']['in-size'] = argv_klee['sym-std']['in-size'] = max(argv_zest['sym-std']['in-size-pre'], argv_klee['sym-std']['in-size-pre'])
        argv_zest['sym-std']['in-present'] = argv_klee['sym-std']['in-present'] = (argv_zest['sym-std']['in-present-pre'] or argv_klee['sym-std']['in-present-pre'])
        argv_zest['sym-std']['out-present'] = argv_klee['sym-std']['out-present'] = (argv_zest['sym-std']['out-present-pre'] or argv_klee['sym-std']['out-present-pre'])

        if argv_zest['sym-files']['nFiles'] > 0:
            commonArgs.append(" ".join(['-sym-files', str(argv_zest['sym-files']['nFiles']), str(argv_zest['sym-files']['size'])]))
        if argv_zest['sym-std']['in-present']:
            commonArgs.append(" ".join(['-sym-stdin', str(argv_zest['sym-std']['in-size'])]))
        if argv_zest['sym-std']['out-present']:
            commonArgs.append("-sym-stdout")

        updateObjects (argv_zest, ktestContains_zest)
        updateObjects (argv_klee, ktestContains_klee)

        # Merge the two objects
        ktestContains["CORRESP_TESTNAME"] = list(ktestContains_zest["CORRESP_TESTNAME"] + ktestContains_klee["CORRESP_TESTNAME"])
        ktestContains["KTEST-OBJ"] = list(ktestContains_zest["KTEST-OBJ"] + ktestContains_klee["KTEST-OBJ"])

        # Change the args list in each ktest object with the common symb args 
        for ktdat in ktestContains["KTEST-OBJ"]:
            ktdat.args = ktdat.args[:1]
            for s in commonArgs:
                ktdat.args += s.strip().split(' ') 

    # Write the new ktest files
    for i in range(len(ktestContains["CORRESP_TESTNAME"])):
        outFname = os.path.join(outDir, "test"+str(i)+".ktest")
        ktestToFile(ktestContains["KTEST-OBJ"][i], outFname)
        name2ktestMap[ktestContains["CORRESP_TESTNAME"][i]] = outFname

    return commonArgs, name2ktestMap
#~ def mergeZestiAndKleeKTests ()

# put information from concolic run for the passed test set into a temporary dir, then possibly
# Compute SEMU symbex and rank according to SEMU. outpout in outFilePath
# semuworkdir contains all the seeds and we sample some for execution
def executeSemu (semuworkdir, semuOutDirs, testSample, test2semudirMap, metaMutantBC, candidateMutantsFiles, symArgs, semuexedir, tuning, mergeThreadsDir=None, exemode=FilterHardToKill): #="zesti+symbex"):
    # Prepare the seeds to use
    threadsOutTop = os.path.dirname(semuOutDirs[0])
    if os.path.isdir(threadsOutTop):
        shutil.rmtree(threadsOutTop)
    os.mkdir(threadsOutTop)
    semuSeedsDir = threadsOutTop+".seeds.tmp"
    if os.path.isdir(semuSeedsDir):
        shutil.rmtree(semuSeedsDir)
    os.mkdir(semuSeedsDir)
    for tc in testSample:
        shutil.copy2(test2semudirMap[tc], semuSeedsDir)

    assert len(candidateMutantsFiles) == len(semuOutDirs), "Missmatch between number of candidate mutant files and number of outputs folders: "+str(len(candidateMutantsFiles))+" VS "+str(len(semuOutDirs))
    nThreads = len(candidateMutantsFiles)

    # Clean possible existing outdir
    for semuOutDir in semuOutDirs:
        if os.path.isdir(semuOutDir):
            shutil.rmtree(semuOutDir)

    # aggregated for the sample tests (semuTC mode)
    #if mode == "zesti+symbex":
    #symbexPreconditions = []
    #for tc in testSample:
    #    tcdir = test2semudirMap[tc]
    #    for pathcondfile in glob.glob(os.path.join(tcdir, "*.pc")):
    #        symbexPreconditions.append(pathcondfile)
            # In the path condition file, replace argv with arg: XXX temporary, DBG
    #        os.system(" ".join(["sed -i'' 's/argv_/arg/g; s/argv/arg0/g'", pathcondfile])) #DBG
    # use the collected preconditions and run semy in symbolic mode

    filter_mutestgen = "" if exemode == FilterHardToKill else " -semu-tests-gen-per-mutant=5" #5 test per mutant

    runSemuCmds = []
    for thread_id in range(nThreads):
        candidateMutantsFile = candidateMutantsFiles[thread_id]
        semuOutDir = semuOutDirs[thread_id]
        logFile = semuOutDir+".log"

        kleeArgs = "-allow-external-sym-calls -libc=uclibc -posix-runtime -search=bfs -solver-backend=stp"
        kleeArgs += ' ' + " ".join([par+'='+str(tuning['KLEE'][par]) for par in tuning['KLEE']])  #-max-time=50000 -max-memory=9000 --max-solver-time=300
        kleeArgs += " -max-sym-array-size=4096 --max-instruction-time=10. -use-cex-cache " # -watchdog"
        kleeArgs += " --output-dir="+semuOutDir
        semukleearg = "-seed-out-dir="+semuSeedsDir
        semukleearg += " -only-replay-seeds" #make sure that the states not of seed are removed
        semuArgs = " ".join([par+'='+str(tuning['SEMU'][par]) for par in tuning['SEMU']])  #" ".join(["-semu-precondition-length=3", "-semu-mutant-max-fork=2"])
        #semuArgs += " " + " ".join(["-semu-precondition-file="+prec for prec in symbexPreconditions])
        semuArgs += filter_mutestgen
        if candidateMutantsFile is not None:
            semuArgs += " -semu-candidate-mutants-list-file " + candidateMutantsFile
        
        semuExe = "klee-semu" if semuexedir is None else os.path.join(semuexedir, "klee-semu")
        runSemuCmd = " ".join([semuExe, kleeArgs, semukleearg, semuArgs, metaMutantBC, " ".join(symArgs), "> /dev/null"]) #,"2>&1"])
        #sretcode = os.system(runSemuCmd)
        runSemuCmd += " 2>"+logFile
        runSemuCmds.append(runSemuCmd)

    print "## Executing SEMU with", nThreads, "parallel threads. Execution log in <semu_out/Thread-<i>.log>"
    threadpool = ThreadPool(nThreads)
    sretcodes = threadpool.map(os.system, runSemuCmds)

    for thread_id, sretcode in enumerate(sretcodes):
        if sretcode != 0 :#and sretcode != 256: # 256 for timeout
            print "-- Returned Code:", sretcode, ", for thread", thread_id,". Command: ", runSemuCmds[thread_id]
            error_exit("Error: klee-semu symbex failled with code "+str(sretcode))
    if mergeThreadsDir is not None:
        if os.path.isdir(mergeThreadsDir):
            shutil.rmtree(mergeThreadsDir)
        os.mkdir(mergeThreadsDir)
        for thread_id in range(nThreads):
            for mutoutfp in glob.glob(os.path.join(semuOutDirs[thread_id],"mutant-*.semu")):
                merg_mutoutfp = os.path.join(mergeThreadsDir, os.path.basename(mutoutfp))
                assert not os.path.isfile(merg_mutoutfp), "Same mutant was treated in different threads (BUG). Mutant id file is: "+os.path.basename(merg_mutoutfp)
                # copy into merge, adding thread id to state (which is an address) to avoid considering different states with same address du to difference in execution threads, as one
                with open(merg_mutoutfp, "w") as f_out:
                    with open(mutoutfp, "r") as f_in:
                        for line in f_in:
                            f_out.write(line.replace(',0x', ','+str(thread_id)+'_0x'))
    #print sretcode, "@@@@ -- ", runSemuCmd  #DBG
    #exit(0)  #DBG

    #else:
    #    error_exit("This mode is not needed. TODO remove it")
    #    os.mkdir(semuOutDir)
    #    mutDataframes = {}
    #    for tc in testSample:
    #        tcdir = test2semudirMap[tc]
    #        for mutFilePath in glob.glob(os.path.join(tcdir, "mutant-*.semu")):
    #            mutFile = os.path.basename(mutFilePath)
    #            tmpdf = pd.read_csv(mutFilePath)
    #            if mutFile not in mutDataframes:
    #                mutDataframes[mutFile] = tmpdf
    #            else:
    #                mutDataframes[mutFile] = pd.concat([mutDataframes[mutFile], tmpdf])
    #    for mutFile in mutDataframes:
    #        aggrmutfilepath = os.path.join(semuOutDir, mutFile)
    #        mutDataframes[mutFile].to_csv(aggrmutfilepath, index=False)

    shutil.rmtree(semuSeedsDir)
#~ def executeSemu()

# Maxtime is for experiment, allow to only consider the data wrtten within the maxtime execution of semu
def processSemu (semuExecutionOutDir, outname, thisOutDir, maxtime=float('inf')):
    outFilePath = os.path.join(thisOutDir, outname)
    # extract for Semu accordincgto sample
    rankSemuMutants.libMain(semuExecutionOutDir, outFilePath, maxtime=maxtime)
#~ def processSemu()

def analysis_plot(thisOut, groundConsideredMutant_covtests):
    analyse.libMain(thisOut, mutantListForRandom=groundConsideredMutant_covtests)
#~ def analysis_plot()

'''
    For mutant killing test generation Mode
    take the ktests in the folders of semuoutputs, the put then together removing duplicates
    The result is put in newly created dir mfi_ktests_dir. The .ktestlist files of each mutant are updated
'''
def fdupeGeneratedTest (mfi_ktests_dir, semuoutputs):
    if os.path.isdir(mfi_ktests_dir):
        shutil.rmtree(mfi_ktests_dir)
    os.mkdir(mfi_ktests_dir)
    ktests = {}
    for fold in semuoutputs:
        kt_fold = glob.glob(fold+"/*.ktest")
        mut_fold = glob.glob(fold+"/mutant-*.ktestlist")
        for ktp in kt_fold:
            assert ktp not in ktests
            ktests[ktp] = []
        for minf in mut_fold:
            df = pd.read_csv(minf)
            for index, row in df.iterrows():
                 et = row["ellapsedTime(s)"]
                 mid = row["MutantID"]
                 ktp = os.path.join(fold, row["ktest"])
                 assert ktp in ktests, "test not in ktests: "+str(ktests)+", test: "+ktp
                 ktests[ktp].append((mid, et))
                  
    # Use fdupes across the dirs in semuoutputs to remove duplicates and update test infos
    fdupesout = mfi_ktests_dir+".tmp"
    fdupcmd = " ".join(["fdupes -1"]+semuoutputs+[">",fdupesout])
    if os.system(fdupcmd) != 0:
        error_exit ("fdupes failed. cmd: "+fdupcmd)
    assert os.path.isfile(fdupesout), "Fdupes failed to produce output"
    with open(fdupesout) as fp:
        for line in fp:
            la = line.strip().split()
            #assert la[0] not in dupmap, "fdupe line: "+la[0]+", is not in dupmap: "+str(dupmap)
            remain = la[0]
            dups = la[1:]
            assert remain in ktests
            for dpkt in dups:
                ktests[remain] += ktests[dpkt]
                del ktests[dpkt]
    os.remove(fdupesout)
    for ktp in ktests:
        ktests[ktp].sort(key=lambda x:x[0]) # sort according to mutant ids
    
    # Copy non duplicates into mfi_ktests_dir
    finalObj = {}
    etimeObj = {'ellapsedTime(s)':[], 'ktest':[]}
    testid = 1
    for ktp in ktests:
        newtname = "test"+str(testid)+".ktest"
        testid += 1
        shutil.copy2(ktp, os.path.join(mfi_ktests_dir, newtname))
        finalObj[newtname] = ktests[ktp]
        etimeObj['ktest'].append(newtname)
        etimeObj['ellapsedTime(s)'].append(min([float(v[1]) for v in ktests[ktp]]))

    etdf = pd.DataFrame(etimeObj)
    etdf.to_csv(os.path.join(mfi_ktests_dir, "tests_by_ellapsedtime.csv"), index=False)
    dumpJson(finalObj, os.path.join(mfi_ktests_dir, "mutant_ktests_mapping.json"))
#~ def fdupeGeneratedTest ()

def stripRootTest2Dir (rootdir, test2dir):
    res = {}
    for tc in test2dir:
        res[tc] = os.path.relpath(test2dir[tc], rootdir)
    return res
#~ def stripRootTest2Dir ()

def prependRootTest2Dir (rootdir, test2dir):
    res = {}
    for tc in test2dir:
        res[tc] = os.path.join(rootdir, test2dir[tc])
    return res
#~ def prependRootTest2Dir ()

'''
    Use this function to create the candidateFunctions.json
    run these in python from the project rootdir (containing cmd):
    >> import sys
    >> sys.path.append(<full path to semu analyse dir>)
    >> import run
    >> indir = "inputs/"
    >> run.mutantsOfFunctions(indir+"/candidateFunctions.json", indir+"/mutantsdata/mutantsInfos.json", create=True)
'''
def mutantsOfFunctions (candidateFunctionsJson, mutinfo, create=False):
    assert os.path.isfile(mutinfo), "mutant info file do not exist: "+mutinfo
    # load mutants info and get the list of mutants per function
    mInf = loadJson(mutinfo)
    mutbyfunc = {}
    for mid in mInf:
        if mInf[mid]["FuncName"] not in mutbyfunc:
            mutbyfunc[mInf[mid]['FuncName']] = []
        mutbyfunc[mInf[mid]['FuncName']].append(int(mid))

    if create:
        # create candidate function list
        cfo = {
                'comments': "Use the function mutantsOfFunctions() form run.py to create this. Note: Empty candidateFunctions mean candidateFunctions = allFunctions",
                'candidateFunctions': [],
                'allFunctions': list(sorted(mutbyfunc.keys()))
                }
        dumpJson(cfo, candidateFunctionsJson)
        return cfo
    else:
        if candidateFunctionsJson is not None:
            assert os.path.isfile(candidateFunctionsJson), "candidate function Json not found: "+candidateFunctionsJson
            cfo = loadJson(candidateFunctionsJson)
            assert  len(cfo['allFunctions']) > 0, "No function found"
            if len(cfo['candidateFunctions']) == 0:
                cfo['candidateFunctions'] += cfo['allFunctions']
            for func in set(cfo['allFunctions']) - set(cfo['candidateFunctions']):
                del mutbyfunc[func]
        return mutbyfunc
# def mutantsOfFunctions()

'''
    Read path files and extract the path lengths
'''
def getPathLengthsMinMaxOfKLeeTests(kleeteststopdir, errmsg):
    assert kleeteststopdir is not None and os.path.isdir(kleeteststopdir), "Klee ktest dir must be specified and existing here: "+kleeteststopdir+". \n"+errmsg

    kleetestsdir = os.path.join(kleeteststopdir, KLEE_TESTGEN_SCRIPT_TESTS+"-out/klee-out-0")
    assert os.path.isdir(kleetestsdir), "kleetestsdir not existing: "+kleetestsdir

    pathfiles = glob.glob(os.path.join(kleetestsdir, "*.path"))
    pathfiles = [v for v in pathfiles if not v.endswith(".sym.path")] #Only consider path files, not .sym.path files
    assert len(pathfiles) > 0, "No path files in klee out folder: "+kleetestsdir+". \n"+errmsg
    plen = []
    for p in pathfiles:
        num_lines = sum(1 for line in open(p))
        plen.append(num_lines)
    minlen = min(plen)
    maxlen = max(plen)
    assert maxlen > 0, "maxlen not greater than 0"
    return minlen, maxlen
#~ def getPathLengthsMinMaxOfKLeeTests()

def assignSemuJobs(mutantsbyfuncs, nMaxBlocks):
    nBlocks = min(nMaxBlocks, len(mutantsbyfuncs))
    mutantsBlocks = [[] for i in range(nBlocks)]
    funcsort = sorted(mutantsbyfuncs.keys(), reverse=True, key=lambda x:len(mutantsbyfuncs[x]))
    ind_min = 0
    i = 0
    while i < len(funcsort):
        mutantsBlocks[ind_min] += mutantsbyfuncs[funcsort[i]]
        i+= 1
        for ind in range(nBlocks):
            if len(mutantsBlocks[ind]) < len(mutantsBlocks[ind_min]):
                ind_min = ind
    return mutantsBlocks  
#~ def assignSemuJobs():

def main():
    global WRAPPER_TEMPLATE 
    global MY_SCANF
    #runMode = "zesti+symbex" #semuTC
    #runMode = "semuTC"
    #if runMode == "zesti+symbex":
    WRAPPER_TEMPLATE = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), ZESTI_CONCOLIC_WRAPPER))
    MY_SCANF = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "FixScanfForShadow/my_scanf.c"))
    #else:
    #    WRAPPER_TEMPLATE = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), SEMU_CONCOLIC_WRAPPER))

    ZESTI_DEV_TASK = 'ZESTI_DEV_TASK'
    TEST_GEN_TASK = 'TEST_GEN_TASK'
    SEMU_EXECUTION = 'SEMU_EXECUTION'
    COMPUTE_TASK = "COMPUTE_TASK"
    ANALYSE_TASK = "ANALYSE_TASK"
    tasksList = [ZESTI_DEV_TASK, TEST_GEN_TASK, SEMU_EXECUTION, COMPUTE_TASK, ANALYSE_TASK]

    parser = argparse.ArgumentParser()
    parser.add_argument("outTopDir", help="topDir for output (required)")
    parser.add_argument("--executionMode", type=str, default=FilterHardToKill, choices=[FilterHardToKill, GenTestsToKill], help="The execution mode for this script (Find hard mutants of generate test to kill mutants.)")
    parser.add_argument("--exepath", type=str, default=None, help="The path to executable in project")
    parser.add_argument("--runtest", type=str, default=None, help="The test running script")
    parser.add_argument("--testlist", type=str, default=None, help="The test list file")
    parser.add_argument("--martout", type=str, default=None, help="The Mart output directory (passing this enable semu selection)")
    parser.add_argument("--matrix", type=str, default=None, help="The Strong Mutation matrix (passing this enable selecting by matrix)")
    parser.add_argument("--coverage", type=str, default=None, help="The mutant Coverage matrix")
    parser.add_argument("--candidateFunctionsJson", type=str, default=None, help="List of Functions to consider (for scalability). Json File,  Empty list of considered means consider all functions")
    parser.add_argument("--zesti_exe_dir", type=str, default=None, help="The Optional directory containing the zesti executable (named klee). if not specified, the default klee must be zesti")
    parser.add_argument("--semu_exe_dir", type=str, default=None, help="The Optional directory containing the SEMu executable (named klee-semu). if not specified, must be available on the PATH")
    parser.add_argument("--klee_tests_topdir", type=str, default=None, help="The Optional directory containing the extra tests separately generated by KLEE")
    parser.add_argument("--covTestThresh", type=str, default='10%', help="Minimum number(percentage) of tests covering a mutant for it to be selected for analysis")
    parser.add_argument("--skip_completed", action='append', default=[], choices=tasksList, help="Specify the tasks that have already been executed")
    parser.add_argument("--testSampleMode", type=str, default="DEV", choices=["DEV", "KLEE", "NUM"], help="choose how to sample subset for evaluation. DEV means use Developer test, NUM, mean a percentage of all tests")
    parser.add_argument("--testSamplePercent", type=float, default=10, help="Specify the percentage of test suite to use for analysis") #, (require setting testSampleMode to NUM)")
    parser.add_argument("--semutimeout", type=int, default=86400, help="Specify the timeout for semu execution")
    parser.add_argument("--semumaxmemory", type=int, default=9000, help="Specify the max memory for semu execution")
    parser.add_argument("--semupreconditionlength", type=str, default='2', help="Specify precondition length semu execution")
    parser.add_argument("--semumutantmaxfork", type=str, default='2', help="Specify hard checkpoint for mutants (or post condition checkpoint) as PC length, in semu execution")
    parser.add_argument("--semuloopbreaktimeout", type=float, default=120.0, help="Specify the timeout delay for ech mutant execution on a test case (estimation), to avoid inifite loop")
    parser.add_argument("--nummaxparallel", type=int, default=1, help="Specify the number of parallel executions (the mutants will be shared accross at most this number of treads for SEMU)")
    args = parser.parse_args()

    outDir = os.path.join(args.outTopDir, OutFolder)
    exePath = args.exepath
    runtestScript = args.runtest
    testList = args.testlist
    martOut = args.martout
    matrix = args.matrix
    coverage = args.coverage
    candidateFunctionsJson = args.candidateFunctionsJson
    klee_tests_topdir = args.klee_tests_topdir
    zesti_exe_dir = args.zesti_exe_dir
    semu_exe_dir = args.semu_exe_dir

    # get abs path in case not
    outDir = os.path.abspath(outDir)
    executionMode = args.executionMode
    exePath = os.path.abspath(exePath) if exePath is not None else None 
    runtestScript = os.path.abspath(runtestScript) if runtestScript is not None else None 
    testList = os.path.abspath(testList) if testList is not None else None 
    martOut = os.path.abspath(martOut) if martOut is not None else None 
    matrix = os.path.abspath(matrix) if matrix is not None else None
    coverage = os.path.abspath(coverage) if coverage is not None else None
    candidateFunctionsJson = os.path.abspath(candidateFunctionsJson) if candidateFunctionsJson is not None else None
    klee_tests_topdir = os.path.abspath(klee_tests_topdir) if klee_tests_topdir is not None else None
    zesti_exe_dir = os.path.abspath(zesti_exe_dir) if zesti_exe_dir is not None else None
    semu_exe_dir = os.path.abspath(semu_exe_dir) if semu_exe_dir is not None else None

    covTestThresh = args.covTestThresh
    testSampleMode = args.testSampleMode
    if testSampleMode in ["KLEE", "NUM"]:
        assert klee_tests_topdir is not None, "klee_tests_topdir not give with KLEE or NUM test Smaple Mode"
    #assert testSampleMode == "DEV", "XXX: Other test sampling modes are not yet supported (KLEE and NUM require fixing symbargs, unless will implement Shadow base test gen)"

    #assert len(set(args.skip_completed) - set(tasksList)) == 0, "specified wrong tasks: "+str(set(args.skip_completed) - set(tasksList))
    toExecute = [v for v in tasksList if v not in args.skip_completed]
    assert len(toExecute) > 0, "Error: Specified skipping all tasks"

    # Check continuity of tasks
    pos = tasksList.index(toExecute[0])
    for v in toExecute[1:]:
        pos += 1
        assert tasksList[pos] == v, "Error: Specified skipping only middle task: "+str(args.skip_completed)

    # We need to set size fraction of test samples
    testSamplePercent = args.testSamplePercent
    assert testSamplePercent > 0 and testSamplePercent <= 100, "Error: Invalid testSamplePercent"

    # Parameter tuning for Semu execution (timeout, to precondition depth)
    ## get precondition and mutantmaxfork from klee tests if specified as percentage
    if args.semupreconditionlength[-1] == '%' or args.semumutantmaxfork[-1] == '%':
        minpath_len, max_pathlen = getPathLengthsMinMaxOfKLeeTests(klee_tests_topdir, "Expecting path file for longest ktest path extraction in klee-test-dir")
    
    if args.semupreconditionlength[-1] == '%':
        args_semupreconditionlength = int(float(args.semupreconditionlength[:-1]) * max_pathlen / 100.0)
    else:
        args_semupreconditionlength = int(args.semupreconditionlength)

    if args.semumutantmaxfork[-1] == '%':
        args_semumutantmaxfork = int(float(args.semumutantmaxfork[:-1]) * max_pathlen / 100.0)
    else:
        args_semumutantmaxfork= int(args.semumutantmaxfork)
    args_semumutantmaxfork = max(1, args_semumutantmaxfork)
    print "#>> SEMU Symbex - Precondition Param:", args_semupreconditionlength, ", Checkpoint Param:", args_semumutantmaxfork
        
    semuTuning = {
                    'KLEE':{'-max-time':args.semutimeout, '-max-memory':args.semumaxmemory, '--max-solver-time':300}, 
                    'SEMU':{"-semu-precondition-length":args_semupreconditionlength, "-semu-mutant-max-fork":args_semumutantmaxfork, "-semu-loop-break-delay":args.semuloopbreaktimeout }
                 }

    # Create outdir if absent
    cacheDir = os.path.join(outDir, "caches")
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
        os.mkdir(cacheDir)


    # get ktest using Zesti  --  Semu for all tests
    if martOut is not None:
        #if runMode == 'zesti+symbex':
        print "# Running Zesti to extract Dev tests"
        #else:
        #    print "# Running Semu..."
        zestiInBC = os.path.basename(exePath) + ZestiBCSuff
        zestiInBCLink = os.path.join(martOut, zestiInBC)
        kleeSemuInBC = os.path.basename(exePath) + KleeSemuBCSuff
        kleeSemuInBCLink = os.path.join(martOut, kleeSemuInBC)
        zestioutdir = os.path.join(cacheDir, "ZestiOutDir")
        inBCFilePath = zestiInBCLink #if runMode == "zesti+symbex" else kleeSemuInBCLink
        test2zestidirMapFile = os.path.join(cacheDir, "test2zestidirMap.json")
        if ZESTI_DEV_TASK in toExecute:
            # Prepare outdir and copy bc
            if os.path.isdir(zestioutdir):
                shutil.rmtree(zestioutdir)
            os.mkdir(zestioutdir)

            unused, alltestsObj, unwrapped_testlist = getTestSamples(testList, 0, matrix)   # 0 to not sample
            test2zestidirMap = runZestiOrSemuTC (unwrapped_testlist, alltestsObj['DEVTESTS'], exePath, runtestScript, inBCFilePath, zestioutdir, zesti_exe_dir) #, mode=runMode) #mode can also be "semuTC"
            dumpJson(stripRootTest2Dir(outDir, test2zestidirMap), test2zestidirMapFile)
        else:
            print "## Loading zesti test mapping from Cache"
            assert os.path.isdir(zestioutdir), "Error: zestioutdir absent when ZESTI_DEV mode skipped"
            test2zestidirMap = loadJson(test2zestidirMapFile)
            test2zestidirMap = prependRootTest2Dir(outDir, test2zestidirMap)

    # TODO: TEST GEN part here. if klee_tests_topdir is not None, means use the tests from klee to increase baseline and dev test to evaluate aproaches
    # prepare seeds and extract sym-args. Then store it in the cache
    semuworkdir = os.path.join(cacheDir, "SemuWorkDir")
    test2semudirMapFile = os.path.join(cacheDir, "test2semudirMap.json")
    if TEST_GEN_TASK in toExecute:
        print "# Doing TEST_GEN_TASK ..."
        assert os.path.isdir(zestioutdir), "Error: "+zestioutdir+" not existing. Please make sure to collect Dev tests with Zesti"
        
        if os.path.isdir(semuworkdir):
            shutil.rmtree(semuworkdir)
        os.mkdir(semuworkdir)

        zest_sym_args_param = None
        zestKTContains = None
        klee_sym_args_param = None
        kleeKTContains = None

        if testSampleMode in ['DEV', 'NUM']:
            zestKtests = []
            for tc in test2zestidirMap.keys():
                tcdir = test2zestidirMap[tc]
                listKtestFiles = glob.glob(os.path.join(tcdir, "*.ktest"))
                assert len(listKtestFiles) == 1, "Error: more than 1 or no ktest from Zesti for tests: "+tc+", zestiout: "+tcdir
                for ktestfile in listKtestFiles:
                    zestKtests.append(ktestfile)
            # refactor the ktest fom zesti and put in semu workdir, together with the sym
            zest_sym_args_param, zestKTContains = getSymArgsFromZestiKtests (zestKtests, test2zestidirMap.keys())

        if testSampleMode in ['KLEE', 'NUM']:
            unused, alltestsObj, unwrapped_testlist = getTestSamples(testList, 0, matrix)   # 0 to not sample
            klee_sym_args_param, kleeKTContains = loadAndGetSymArgsFromKleeKTests (alltestsObj['GENTESTS'], klee_tests_topdir)
            
        sym_args_param, test2semudirMap = mergeZestiAndKleeKTests (semuworkdir, zestKTContains, zest_sym_args_param, kleeKTContains, klee_sym_args_param)
        dumpJson([testSampleMode, sym_args_param, stripRootTest2Dir(outDir, test2semudirMap)], test2semudirMapFile)
    else:
        print "## Loading parametrized tests mapping from cache"
        assert os.path.isdir(semuworkdir), "Error: semuworkdir absent when TEST-GEN mode skipped"
        tmpSamplMode, sym_args_param, test2semudirMap = loadJson(test2semudirMapFile)
        assert tmpSamplMode == testSampleMode, "Given test Sample Mode ("+testSampleMode+") is different from caches ("+tmpSamplMode+"); should not skip TEST_GET_TASK!"
        test2semudirMap = prependRootTest2Dir(outDir, test2semudirMap)

    # Get all test samples before starting experiment
    ## TODO TODO: Fix this when supporting other testSampleModes
    print "# Getting Test Samples .."
    if SEMU_EXECUTION in toExecute: 
        invalid_ktests = set(test2zestidirMap) - set (test2semudirMap)
        testSamples, alltestsObj, unwrapped_testlist = getTestSamples(testList, testSamplePercent, matrix, discards=invalid_ktests) 

        assert testSamplePercent > 0, "testSamplePercent must be greater than 0"
        if testSampleMode == 'DEV':
            sampl_size = int(max(1, testSamplePercent * len(alltestsObj['DEVTESTS']) / 100))
            testSamples = {'DEV_'+str(testSamplePercent): random.sample(alltestsObj['DEVTESTS'], sampl_size)}
        elif testSampleMode == 'KLEE':
            sampl_size = int(max(1, testSamplePercent * len(alltestsObj['GENTESTS']) / 100))
            testSamples = {'KLEE_'+str(testSamplePercent): random.sample(alltestsObj['GENTESTS'], sampl_size)}
        elif testSampleMode == 'NUM':
            #already sampled above
            assert len(testSampleMode) > 0
        else:
            error_exit ("inavlid test sampling mode: "+testSampleMode)

        dumpJson([testSamples, alltestsObj, unwrapped_testlist], os.path.join(cacheDir, "testsamples.json"))
    else:
        print "## Loading test samples from cache"
        testSamples, alltestsObj, unwrapped_testlist = loadJson(os.path.join(cacheDir, "testsamples.json"))
    alltests = alltestsObj["DEVTESTS"] + alltestsObj["GENTESTS"]

    assert len(testSamples.keys()) == 1, "TestSamples must have only one key (correspond to one experiment - one set of mutants and tests)"

    # Get candidate mutants -----------------------------------------
    ## In filtering Mode, get killable mutant with threshold test coverage
    ## In test generation for killing mutants Mode, get test equivalent mutants, w.r.t the specified test sample percent and testSampleMode
    candidateMutantsFile = None
    groundConsideredMutant_covtests = None
    list_groundConsideredMutant_covtests = []
    if matrix is not None:
        if executionMode == FilterHardToKill:
            ground_UNK_K_illedMutants = set(matrixHardness.getKillableMutants(matrix)) 
        else:
            assert len(testSamples.keys()) == 1, "TestSamples must have only one key (correspond to one experiment - one set of mutants and tests)"
            ground_UNK_K_illedMutants = set(matrixHardness.getUnKillableMutants(matrix, testset=set(testSamples[testSamples.keys()[0]]))) 

        groundConsideredMutant_covtests = matrixHardness.getCoveredMutants(coverage, testTresh_str = covTestThresh)
        # keep only covered by treshold at least, and killed
        for mid in set(groundConsideredMutant_covtests) - ground_UNK_K_illedMutants:
            del groundConsideredMutant_covtests[mid]
        print "# Number of Mutants after coverage filtering:", len(groundConsideredMutant_covtests)
        
        # consider the specified functions
        afterFuncFilter_byfunc = mutantsOfFunctions (candidateFunctionsJson, os.path.join(martOut, mutantInfoFile), create=False)
        # make considered mutants and afterFuncFilter_byfunc be in sync
        gCM_c_set = set(groundConsideredMutant_covtests)
        intersect_ga = set()
        for func in afterFuncFilter_byfunc.keys():
            tmpa = set(afterFuncFilter_byfunc[func]) & gCM_c_set
            if len(tmpa) > 0:
                afterFuncFilter_byfunc[func] = list(tmpa)
                intersect_ga |= tmpa
            else:
                del afterFuncFilter_byfunc[func]

        for mid in gCM_c_set - intersect_ga:
            del groundConsideredMutant_covtests[mid]

        paraAssign = assignSemuJobs(afterFuncFilter_byfunc, args.nummaxparallel)
        for pj in paraAssign:
            pjs = set(pj)
            list_groundConsideredMutant_covtests.append({mid: groundConsideredMutant_covtests[mid] for mid in groundConsideredMutant_covtests if mid in pjs})
        
        print "# Number of Mutants Considered:", sum([len(x) for x in list_groundConsideredMutant_covtests]), ". With", len(paraAssign), "Semu executions in parallel"

        minMutNum = 10 if executionMode == FilterHardToKill else 1

        assert sum([len(x) for x in list_groundConsideredMutant_covtests]) > minMutNum, " ".join(["We have only", str(sum([len(x) for x in list_groundConsideredMutant_covtests])), "mutants fullfiling testcover treshhold",str(covTestThresh),"(Expected >= "+str(minMutNum)+")"])
        list_candidateMutantsFiles = []
        for th_id, mcd in enumerate(list_groundConsideredMutant_covtests):
            candidateMutantsFile = os.path.join(cacheDir, "candidateMutants_"+str(th_id)+".list")
            list_candidateMutantsFiles.append(candidateMutantsFile)
            with open(candidateMutantsFile, "w") as f:
                for mid in mcd.keys():
                    f.write(str(mid)+"\n")
    #---------------------------------------------------

    semuOutputsTop = os.path.join(cacheDir, "semu_outputs")
    if not os.path.isdir(semuOutputsTop):
        os.mkdir(semuOutputsTop)

    # process and analyse for each test Sample with each approach
    for ts_size in testSamples:
        # Make temporary outdir for test sample size
        outFolder = "out_testsize_"+str(ts_size)
        this_Out = os.path.join(outDir, outFolder)
        thisOut_list = [this_Out+"-"+str(i) for i in range(len(list_candidateMutantsFiles))]
        mergeSemuThisOut = this_Out+".semumerged"

        se_output = os.path.join(semuOutputsTop, outFolder, "Thread")
        semuoutputs = [se_output+"-"+str(i) for i in range(len(list_candidateMutantsFiles))]
        # No merge on Test Gen mode
        mergeSemuThreadsDir = se_output+".semumerged" if executionMode == FilterHardToKill else None

        # Execute SEMU
        if SEMU_EXECUTION in toExecute: 
            if martOut is not None:
                executeSemu (semuworkdir, semuoutputs, testSamples[ts_size], test2semudirMap, kleeSemuInBCLink, list_candidateMutantsFiles, sym_args_param, semu_exe_dir, semuTuning, mergeThreadsDir=mergeSemuThreadsDir, exemode=executionMode) 

        if executionMode == FilterHardToKill:
            if len(thisOut_list) == 1 and os.path.isdir(mergeSemuThreadsDir): #only have one thread, only process that
                zips = [(groundConsideredMutant_covtests, mergeSemuThisOut, mergeSemuThreadsDir)]
            else:
                zips = zip(list_groundConsideredMutant_covtests+[groundConsideredMutant_covtests], thisOut_list+[mergeSemuThisOut], semuoutputs+[mergeSemuThreadsDir])
            for groundConsMut_cov, thisOut, semuoutput in zips:
                # process with each approach
                if COMPUTE_TASK in toExecute: 
                    print "# Procesing for test size", ts_size, "..."

                    if martOut is not None or matrix is not None:
                        if os.path.isdir(thisOut):
                            shutil.rmtree(thisOut)
                        os.mkdir(thisOut)

                    # process for matrix
                    if matrix is not None:
                        processMatrix (matrix, alltests, 'groundtruth', groundConsMut_cov, thisOut) 
                        processMatrix (matrix, testSamples[ts_size], 'classic', groundConsMut_cov, thisOut) 

                    # process for SEMU
                    if martOut is not None:
                        processSemu (semuoutput, "semu", thisOut)

                # Analysing for each test Sample 
                if ANALYSE_TASK in toExecute:
                    print "# Analysing for test size", ts_size, "..."

                    # Make final Analysis and plot
                    if martOut is not None and matrix is not None:
                        analysis_plot(thisOut, None) #groundConsideredMutant_covtests.keys()) # None to not plot random
        else:
            mfi_mutants_list = os.path.join(this_Out, "mfirun_mutants_list.txt")
            mfi_ktests_dir = os.path.join(this_Out, "mfirun_ktests_dir")
            mfi_execution_output = os.path.join(this_Out, "mfirun_output")
            if COMPUTE_TASK in toExecute: 
                print "# Compute task Procesing for test size", ts_size, "..."
                if not os.path.isdir(this_Out):
                    os.mkdir(this_Out)
                if os.path.isdir(mfi_execution_output):
                    print ""
                    choice = raw_input("Are you sure you want to clean existing mfi_execution_output? [y/N] ")
                    if choice.lower() in ['y', 'yes']:
                        shutil.rmtree(mfi_execution_output)

                if not os.path.isdir(mfi_execution_output):
                    with open(mfi_mutants_list, "w") as fp:
                        for mid in groundConsideredMutant_covtests:
                            fp.write(str(mid)+'\n')
                    fdupeGeneratedTest (mfi_ktests_dir, semuoutputs)

            if ANALYSE_TASK in toExecute:
                nGenTests_ = len(glob.glob(mfi_ktests_dir+"/*.ktest"))
                if nGenTests_ > 0 and not os.path.isdir(mfi_execution_output):
                    print "\n--------------------------------------------------"
                    print ">> There are a total of", nGenTests_, "tests to run"
                    print ">> You now need to execute the generated tests using MFI (externale mode)"
                    print ">> For MFI, use the following:"
                    print ">>   mfi_mutants_list:", mfi_mutants_list
                    print ">>   mfi_ktests_dir:", mfi_ktests_dir
                    print ">>   mfi_execution_output:", mfi_execution_output
                    print "--------------------------------------------------"
                    print "@ Rexecute this when done."
                else:
                    nMutants = len(groundConsideredMutant_covtests)
                    outjsonfile = os.path.join(this_Out, "MS-increase")
                    if nGenTests_ > 0:
                        sm_file = os.path.join(mfi_execution_output, "data", "matrices")
                        nnewKilled = len(matrixHardness.getKillableMutants(sm_file))
                    else:
                        nnewKilled = 0
                    outobj_ = {"#Mutants": nMutants, "#Killed": nnewKilled, "#GenTests":nGenTests_, "MS-INC":(nnewKilled * 100.0 / nMutants)}
                    dumpJson(outobj_, outjsonfile)
                    print "Kill Mutant TestGen Analyse Result:", outobj_
        print "@ DONE!"

#~ def main()

if __name__ == "__main__":
    main()

'''
FLOW:
-----

1) Get DEV Tests KTESTS with ZESTI 
2-a) Transform Zesti ktests(seeds) for symbex, as well as the Sym args for running Semu
2-b) Use those to generate more tests using SHADOW-SEMU  (for now we only use KLEE)
3) Split the tests in training and evaluation (whith shadow random sample all. With KLEE, use Dev for training and KLEE for eval and vice vesa)
4) Evaluate the technique using Tranining and evaluation tests
'''
