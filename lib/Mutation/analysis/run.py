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
        for s in range(samplePercent, 101, samplePercent):
            samples[s] = testlist[:int(s * len(testlist) / 100.0)]
            assert len(samples[s]) > 0, "too few test to sample percentage"
    return samples, {'GENTESTS': kleetestlist, 'DEVTESTS':devtestlist}, unwrapped_testlist
#~ def getTestSamples()

def processMatrix (matrix, testSample, outname, candMutants_covTests, thisOutDir):
    outFilePath = os.path.join(thisOutDir, outname)
    matrixHardness.libMain(matrix, testSample, candMutants_covTests, outFilePath)
#~ def processMatrix()

def runZestiOrSemuTC (unwrapped_testlist, devtests, exePath, runtestScript, kleeZestiSemuInBCLink, outpdir, zestiexedir, mode="zesti+symbex"):
    test2outdirMap = {}

    # copy bc
    kleeZestiSemuInBC = os.path.basename(kleeZestiSemuInBCLink) 
    if mode == "zesti+symbex":
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
    else:
        print "# Running SEMU Concretely\n"
        shutil.copy2(kleeZestiSemuInBCLink, os.path.join(outpdir, kleeZestiSemuInBC))

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
'''
def parseTextKtest(filename):
    schar = [chr(i) for i in range(ord('A'),ord('Z')+1)+range(ord('a'),ord('z')+1)]
    ShortNames = {'names':[z[0]+z[1] for z in itertools.product(schar,['']+schar)], 'pos':0}
    def getShortname(dat=ShortNames):
        dat['pos'] += 1
        return dat['names'][dat['pos']-1]

    datalist = []
    b = ktest_tool.KTest.fromfile(filename)
    # get the object one at the time and obtain its stuffs
    # the objects are as following: [model_version, <the argvs> <file contain, file stat>]
    # Note that files appear in argv. here we do not have n_args because concrete (from Zesti)
    seenFileStatsPos = set()
    stdin = None
    model_version_pos = -1
    fileargsposinObj_remove = []
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
                    datalist[iv] = ('FILE', len(data)) #XXX

                fileargsposinObj_remove += indinargs  # ARGVs come before files in objects

                shortfname = getShortname(ShortNames)

                # seach for its stat
                found = False
                for si,(sname,sdata) in enumerate(b.objects):
                    if sname == name+"-stat":
                        seenFileStatsPos.add(si)
                        b.objects[si] = (shortfname+'-stat', sdata)
                        b.objects[ind] = (shortfname, data)
                        found = True
                        break
                if not found:
                    error_exit("File is not having stat in ktest")
            #elif name == "stdin-stat": #case of stdin
            #    stdin = ('STDIN', len(data)) #XXX 
            else: #ARGV
                datalist.append(('ARGV', len(data))) #XXX

    # remove the objects in fileargsposinObj_remove, considering model_version_pos
    fileargsposinObj_remove = list(set(fileargsposinObj_remove))
    if model_version_pos == 0:
        fileargsposinObj_remove = [v+1 for v in fileargsposinObj_remove]
    elif model_version_pos == b.objects:
        model_version_pos -= len(fileargsposinObj_remove)
    else:
        assert False, "model version need to be either 1st or last object initially"

    for pos in sorted(fileargsposinObj_remove, reverse=True):
        del b.objects[pos]

    if stdin is not None:
        datalist.append(stdin)
    else:
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

def getSymArgsFromKtests (ktestFilesList, testNamesList, outDir):
    assert len(ktestFilesList) == len(testNamesList), "Error: size mismatch btw ktest and names: "+str(len(ktestFilesList))+" VS "+str(len(testNamesList))
    name2ktestMap = {}
    # XXX implement this. For program with file as parameter, make sure that the filenames are renamed in the path conditions(TODO double check)
    listTestArgs = []
    ktestContains = {"CORRESP_TESTNAME":[], "KTEST-OBJ":[]}
    for ipos, ktestfile in enumerate(ktestFilesList):
        # XXX Zesti do not generate valid Ktest file when an argument is the empty string. Example tests 'basic_s18' of EXPR which is: expr "" "|" ""
        # The reson is that when writing ktest file, klee want the name to be non empty thus it fail (I think). 
        # Thus, we skip such tests here TODO: remove thes from all tests so to have fair comparison with semu
        if os.system(" ".join(['ktest-tool ', ktestfile, "> /dev/null 2>&1"])) != 0:
            print "@WARNING: Skipping test because Zesti generated invalid KTEST file:", ktestfile
            continue

        # sed because Zesti give argv, argv_1... while sym args gives arg0, arg1,...
        ktestdat, testArgs = parseTextKtest(ktestfile)
        listTestArgs.append(testArgs)
        ktestContains["CORRESP_TESTNAME"].append(testNamesList[ipos])
        ktestContains["KTEST-OBJ"].append(ktestdat)
    if len(listTestArgs) <= 0:
        print "Err: no ktest data, ktest PCs:", ktestFilesList
        error_exit ("No ktest data could be extracted from ktests.")

    # Make a general form out of listTestArgs by inserting what is needed with size 0
    # Make use of the sym-args param that can unset a param (klee care about param order)
    # Split each test args according to the FILE type (STDIN is always last), as follow: ARGV ARGV FILE ARGV FILE ...
    # then use -sym-args to flexibly set the number of enables argvs. First process the case before the first FILE, then between 1st and 2nd
    commonArgs = []
    commonArgsNumPerTest = {t: [] for t in range(len(listTestArgs))}
    testsCurFilePos = [0 for i in range(len(listTestArgs))]
    testsNumArgvs = [0 for i in range(len(listTestArgs))]
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
        fileMaxSize = -1
        stdinMaxSize = -1
        for t in range(len(testsNumArgvs)):
            # if the last arg was ARGV do nothing
            if testsCurFilePos[t] >= len(listTestArgs[t]):
                continue
            # If next is FILE
            if listTestArgs[t][testsCurFilePos[t]][0] == "FILE":
                fileMaxSize = max(fileMaxSize, listTestArgs[t][testsCurFilePos[t]][1])
                testsCurFilePos[t] += 1
            # If next is STDIN (last)
            elif listTestArgs[t][testsCurFilePos[t]][0] == "STDIN":
                stdinMaxSize = max(stdinMaxSize, listTestArgs[t][testsCurFilePos[t]][1])
                #testsCurFilePos[t] += 1  # XXX Not needed since stdin is the last arg
            else:
                error_exit("unexpected arg type here: Neither FILE nor STDIN (type is "+listTestArgs[t][testsCurFilePos[t]][0]+")")

        if fileMaxSize >= 0:
            commonArgs.append(" ".join(["-sym-files 1", str(fileMaxSize)]))
        else:
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

    # Write the new ktest files
    for i in range(len(ktestContains["CORRESP_TESTNAME"])):
        outFname = os.path.join(outDir, "test"+str(i)+".ktest")
        ktestToFile(ktestContains["KTEST-OBJ"][i], outFname)
        name2ktestMap[ktestContains["CORRESP_TESTNAME"][i]] = outFname

    return commonArgs, name2ktestMap
#~ getSymArgsFromKtests()

# put information from concolic run for the passed test set into a temporary dir, then possibly
# Compute SEMU symbex and rank according to SEMU. outpout in outFilePath
# semuworkdir contains all the seeds and we sample some for execution
def executeSemu (semuworkdir, semuOutDir, testSample, test2semudirMap, metaMutantBC, candidateMutantsFile, symArgs, semuexedir, tuning, mode="zesti+symbex"):
    # Prepare the seeds to use
    semuSeedsDir = semuOutDir+".seeds.tmp"
    if os.path.isdir(semuSeedsDir):
        shutil.rmtree(semuSeedsDir)
    os.mkdir(semuSeedsDir)
    for tc in testSample:
        shutil.copy2(test2semudirMap[tc], semuSeedsDir)

    # Clean possible existing outdir
    if os.path.isdir(semuOutDir):
        shutil.rmtree(semuOutDir)

    # aggregated for the sample tests (semuTC mode)
    if mode == "zesti+symbex":
        #symbexPreconditions = []
        #for tc in testSample:
        #    tcdir = test2semudirMap[tc]
        #    for pathcondfile in glob.glob(os.path.join(tcdir, "*.pc")):
        #        symbexPreconditions.append(pathcondfile)
                # In th parth condition file, replace argv with arg: XXX temporary, DBG
        #        os.system(" ".join(["sed -i'' 's/argv_/arg/g; s/argv/arg0/g'", pathcondfile])) #DBG
        # use the collected preconditions and run semy in symbolic mode
        kleeArgs = "-allow-external-sym-calls -libc=uclibc -posix-runtime -search=bfs -solver-backend=stp"
        kleeArgs += ' ' + " ".join([par+'='+str(tuning['KLEE'][par]) for par in tuning['KLEE']])  #-max-time=50000 -max-memory=9000 --max-solver-time=300
        kleeArgs += " -max-sym-array-size=4096 --max-instruction-time=10. -use-cex-cache " # -watchdog"
        kleeArgs += " --output-dir="+semuOutDir
        semukleearg = "-seed-out-dir="+semuSeedsDir
        semukleearg += " -only-replay-seeds" #make sure that the states not of seed are removed
        semuArgs = " ".join([par+'='+str(tuning['SEMU'][par]) for par in tuning['SEMU']])  #" ".join(["-semu-precondition-length=3", "-semu-mutant-max-fork=2"])
        #semuArgs += " " + " ".join(["-semu-precondition-file="+prec for prec in symbexPreconditions])
        if candidateMutantsFile is not None:
            semuArgs += " -semu-candidate-mutants-list-file " + candidateMutantsFile
        
        semuExe = "klee-semu" if semuexedir is None else os.path.join(semuexedir, "klee-semu")
        runSemuCmd = " ".join([semuExe, kleeArgs, semukleearg, semuArgs, metaMutantBC, " ".join(symArgs), "> /dev/null"])
        sretcode = os.system(runSemuCmd)
        if sretcode != 0 and sretcode != 256: # 256 for timeout
            print "-- Returned Code:", sretcode, ". Command: ", runSemuCmd 
            error_exit("Error: klee-semu symbex failled with code "+str(sretcode))
        #print sretcode, "@@@@ -- ", runSemuCmd  #DBG
        #exit(0)  #DBG
    else:
        os.mkdir(semuOutDir)
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
            aggrmutfilepath = os.path.join(semuOutDir, mutFile)
            mutDataframes[mutFile].to_csv(aggrmutfilepath, index=False)

    shutil.rmtree(semuSeedsDir)
#~ def executeSemu()

def processSemu (semuExecutionOutDir, outname, thisOutDir):
    outFilePath = os.path.join(thisOutDir, outname)
    # extract for Semu accordincgto sample
    rankSemuMutants.libMain(semuExecutionOutDir, outFilePath)
#~ def processSemu()

def analysis_plot(thisOut, groundConsideredMutant_covtests):
    analyse.libMain(thisOut, mutantListForRandom=groundConsideredMutant_covtests)
#~ def analysis_plot()

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

    ZESTI_DEV_TASK = 'ZESTI_DEV_TASK'
    TEST_GEN_TASK = 'TEST_GEN_TASK'
    SEMU_EXECUTION = 'SEMU_EXECUTION'
    COMPUTE_TASK = "COMPUTE_TASK"
    ANALYSE_TASK = "ANALYSE_TASK"
    tasksList = [ZESTI_DEV_TASK, TEST_GEN_TASK, SEMU_EXECUTION, COMPUTE_TASK, ANALYSE_TASK]

    parser = argparse.ArgumentParser()
    parser.add_argument("outTopDir", help="topDir for output (required)")
    parser.add_argument("--exepath", type=str, default=None, help="The path to executable in project")
    parser.add_argument("--runtest", type=str, default=None, help="The test running script")
    parser.add_argument("--testlist", type=str, default=None, help="The test list file")
    parser.add_argument("--martout", type=str, default=None, help="The Mart output directory (passing this enable semu selection)")
    parser.add_argument("--matrix", type=str, default=None, help="The Strong Mutation matrix (passing this enable selecting by matrix)")
    parser.add_argument("--coverage", type=str, default=None, help="The mutant Coverage matrix")
    parser.add_argument("--zesti_exe_dir", type=str, default=None, help="The Optional directory containing the zesti executable (named klee). if not specified, the default klee must be zesti")
    parser.add_argument("--semu_exe_dir", type=str, default=None, help="The Optional directory containing the SEMu executable (named klee-semu). if not specified, must be available on the PATH")
    parser.add_argument("--klee_tests_dir", type=str, default=None, help="The Optional directory containing the extra tests separately generated by KLEE")
    parser.add_argument("--covTestThresh", type=int, default=10, help="Minimum number of tests covering a mutant for it to be selected for analysis")
    parser.add_argument("--skip_completed", action='append', default=[], choices=tasksList, help="Specify the tasks that have already been executed")
    parser.add_argument("--testSampleMode", type=str, default="DEV", choices=["DEV", "KLEE", "NUM"], help="choose how to sample subset for evaluation. DEV means use Developer test, NUM, mean a percentage of all tests")
    parser.add_argument("--testSamplePercent", type=int, default=10, help="Specify the percentage of test suite to use for analysis, (require setting testSampleMode to NUM)")
    parser.add_argument("--semutimeout", type=int, default=86400, help="Specify the timeout for semu execution")
    parser.add_argument("--semumaxmemory", type=int, default=9000, help="Specify the max memory for semu execution")
    parser.add_argument("--semupreconditionlength", type=int, default=2, help="Specify precondition length semu execution")
    parser.add_argument("--semumutantmaxfork", type=int, default=2, help="Specify hard checkpoint for mutants (or post condition checkpoint) as PC length, in semu execution")
    parser.add_argument("--semuloopbreaktimeout", type=float, default=120.0, help="Specify the timeout delay for ech mutant execution on a test case (estimation), to avoid inifite loop")
    args = parser.parse_args()

    outDir = os.path.join(args.outTopDir, OutFolder)
    exePath = args.exepath
    runtestScript = args.runtest
    testList = args.testlist
    martOut = args.martout
    matrix = args.matrix
    coverage = args.coverage
    klee_tests_dir = args.klee_tests_dir
    zesti_exe_dir = args.zesti_exe_dir
    semu_exe_dir = args.semu_exe_dir

    # get abs path in case not
    outDir = os.path.abspath(outDir)
    exePath = os.path.abspath(exePath) if exePath is not None else None 
    runtestScript = os.path.abspath(runtestScript) if runtestScript is not None else None 
    testList = os.path.abspath(testList) if testList is not None else None 
    martOut = os.path.abspath(martOut) if martOut is not None else None 
    matrix = os.path.abspath(matrix) if matrix is not None else None
    coverage = os.path.abspath(coverage) if coverage is not None else None
    klee_tests_dir = os.path.abspath(klee_tests_dir) if klee_tests_dir is not None else None
    zesti_exe_dir = os.path.abspath(zesti_exe_dir) if zesti_exe_dir is not None else None
    semu_exe_dir = os.path.abspath(semu_exe_dir) if semu_exe_dir is not None else None

    covTestThresh = args.covTestThresh
    testSampleMode = args.testSampleMode
    assert testSampleMode == "DEV", "XXX: Other test sampling modes are not yet supported (KLEE and NUM require fixing symbargs, unless will implement Shadow base test gen)"

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
    semuTuning = {
                    'KLEE':{'-max-time':args.semutimeout, '-max-memory':args.semumaxmemory, '--max-solver-time':300}, 
                    'SEMU':{"-semu-precondition-length":args.semupreconditionlength, "-semu-mutant-max-fork":args.semumutantmaxfork, "-semu-loop-break-delay":args.semuloopbreaktimeout }
                 }

    # Create outdir if absent
    cacheDir = os.path.join(outDir, "caches")
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
        os.mkdir(cacheDir)

    # Get candidat mutants
    candidateMutantsFile = None
    groundConsideredMutant_covtests = None
    if matrix is not None:
        groundKilledMutants = set(matrixHardness.getKillableMutants(matrix)) 
        groundConsideredMutant_covtests = matrixHardness.getCoveredMutants(coverage, testTresh = covTestThresh)
        # keep only covered by treshold at least, and killed
        for mid in set(groundConsideredMutant_covtests) - groundKilledMutants:
            del groundConsideredMutant_covtests[mid]
        print "# Number of Mutants Considered:", len(groundConsideredMutant_covtests)
        assert len(groundConsideredMutant_covtests) > 10, " ".join(["We have only", str(len(groundConsideredMutant_covtests)), "mutant fullfiling testcover treshhold",str(covTestThresh),"(Expected >= 10)"])
        candidateMutantsFile = os.path.join(cacheDir, "candidateMutants.list")
        with open(candidateMutantsFile, "w") as f:
            for mid in groundConsideredMutant_covtests.keys():
                f.write(str(mid)+"\n")

    # get ktest using Zesti  --  Semu for all tests
    if martOut is not None:
        if runMode == 'zesti+symbex':
            print "# Running Zesti to extract Dev tests"
        else:
            print "# Running Semu..."
        zestiInBC = os.path.basename(exePath) + ZestiBCSuff
        zestiInBCLink = os.path.join(martOut, zestiInBC)
        kleeSemuInBC = os.path.basename(exePath) + KleeSemuBCSuff
        kleeSemuInBCLink = os.path.join(martOut, kleeSemuInBC)
        zestioutdir = os.path.join(cacheDir, "ZestiOutDir")
        inBCFilePath = zestiInBCLink if runMode == "zesti+symbex" else kleeSemuInBCLink
        test2zestidirMapFile = os.path.join(cacheDir, "test2zestidirMap.json")
        if ZESTI_DEV_TASK in toExecute:
            # Prepare outdir and copy bc
            if os.path.isdir(zestioutdir):
                shutil.rmtree(zestioutdir)
            os.mkdir(zestioutdir)

            unused, alltestsObj, unwrapped_testlist = getTestSamples(testList, 0, matrix)   # 0 to not sample
            test2zestidirMap = runZestiOrSemuTC (unwrapped_testlist, alltestsObj['DEVTESTS'], exePath, runtestScript, inBCFilePath, zestioutdir, zesti_exe_dir, mode=runMode) #mode can also be "semuTC"
            dumpJson(stripRootTest2Dir(outDir, test2zestidirMap), test2zestidirMapFile)
        else:
            print "## Loading zesti test mapping from Cache"
            assert os.path.isdir(zestioutdir), "Error: zestioutdir absent when ZESTI_DEV mode skipped"
            test2zestidirMap = loadJson(test2zestidirMapFile)
            test2zestidirMap = prependRootTest2Dir(outDir, test2zestidirMap)

    # TODO: TEST GEN part here. if klee_tests_dir is not None, means use the tests from klee to increase baseline and dev test to evaluate aproaches
    # prepare seeds and extract sym-args. Then store it in the cache
    semuworkdir = os.path.join(cacheDir, "SemuWorkDir")
    test2semudirMapFile = os.path.join(cacheDir, "test2semudirMap.json")
    if TEST_GEN_TASK in toExecute:
        print "# Doing TEST_GEN_TASK ..."
        assert os.path.isdir(zestioutdir), "Error: "+zestioutdir+" not existing. Please make sure to collect Dev tests with Zesti"
        
        if os.path.isdir(semuworkdir):
            shutil.rmtree(semuworkdir)
        os.mkdir(semuworkdir)

        # refactor the ktest fom zesti and put in semu workdir, together with the sym
        zestKtests = []
        for tc in test2zestidirMap.keys():
            tcdir = test2zestidirMap[tc]
            listKtestFiles = glob.glob(os.path.join(tcdir, "*.ktest"))
            assert len(listKtestFiles) == 1, "Error: more than 1 or no ktest from Zesti for tests: "+tc+", zestiout: "+tcdir
            for ktestfile in listKtestFiles:
                zestKtests.append(ktestfile)
        sym_args_param, test2semudirMap = getSymArgsFromKtests (zestKtests, test2zestidirMap.keys(), semuworkdir)
        dumpJson([sym_args_param, stripRootTest2Dir(outDir, test2semudirMap)], test2semudirMapFile)
    else:
        print "## Loading parametrized tests mapping from cache"
        assert os.path.isdir(semuworkdir), "Error: semuworkdir absent when TEST-GEN mode skipped"
        sym_args_param, test2semudirMap = loadJson(test2semudirMapFile)
        test2semudirMap = prependRootTest2Dir(outDir, test2semudirMap)

    # Get all test samples before starting experiment
    ## TODO TODO: Fix this when supporting other testSampleModes
    print "# Getting Test Samples .."
    invalid_ktests = set(test2zestidirMap) - set (test2semudirMap)
    testSamples, alltestsObj, unwrapped_testlist = getTestSamples(testList, testSamplePercent, matrix, discards=invalid_ktests) 
    dumpJson([testSamples, alltestsObj, unwrapped_testlist], os.path.join(cacheDir, "testsamples.json"))
    alltests = alltestsObj["DEVTESTS"] + alltestsObj["GENTESTS"]

    if testSampleMode == 'DEV':
        testSamples = {'DEV': alltestsObj['DEVTESTS']}
    elif testSampleMode == 'KLEE':
        testSamples = {'KLEE': alltestsObj['GENTESTS']}

    semuOutputs = os.path.join(cacheDir, "semu_outputs")
    if not os.path.isdir(semuOutputs):
        os.mkdir(semuOutputs)

    # process and analyse for each test Sample with each approach
    for ts_size in testSamples:
        # Make temporary outdir for test sample size
        outFolder = "out_testsize_"+str(ts_size)
        thisOut = os.path.join(outDir, outFolder)

        semuoutput = os.path.join(semuOutputs, outFolder)

        # Execute SEMU
        if SEMU_EXECUTION in toExecute: 
            if martOut is not None:
                executeSemu (semuworkdir, semuoutput, testSamples[ts_size], test2semudirMap, kleeSemuInBCLink, candidateMutantsFile, sym_args_param, semu_exe_dir, semuTuning, mode=runMode) 

        # process with each approach
        if COMPUTE_TASK in toExecute: 
            print "# Procesing for test size", ts_size, "..."

            if martOut is not None or matrix is not None:
                if os.path.isdir(thisOut):
                    shutil.rmtree(thisOut)
                os.mkdir(thisOut)

            # process for matrix
            if matrix is not None:
                processMatrix (matrix, alltests, 'groundtruth', groundConsideredMutant_covtests, thisOut) 
                processMatrix (matrix, testSamples[ts_size], 'classic', groundConsideredMutant_covtests, thisOut) 

            # process for SEMU
            if martOut is not None:
                processSemu (semuoutput, "semu", thisOut)

        # Analysing for each test Sample 
        if ANALYSE_TASK in toExecute:
            print "# Analysing for test size", ts_size, "..."

            # Make final Analysis and plot
            if martOut is not None and matrix is not None:
                analysis_plot(thisOut, None) #groundConsideredMutant_covtests.keys()) # None to not plot random

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
