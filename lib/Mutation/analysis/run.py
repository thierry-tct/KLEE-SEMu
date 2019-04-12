#! /usr/bin/env python

#######################
# This script takes as input:
# - The Strong Mutation matrix (passing this enable selecting by matrix)
# - The Mart output directory (passing this enable semu selection)
# - The test list file
# - The test running script
# - The path to executable in project
# - topDir for output (required)
########################
#XXX INFO: In case there are some tests that fail with zesti and want to skip them,
#XXX INFO: Just rerun passing the environment variable: SEMU_ZESTI_RUN_SKIP_FAILURE=on

import os, sys, stat
import json, re
import shutil, glob
import tarfile
import argparse
import random, time
import math
import pandas as pd
import struct
import itertools

from multiprocessing.pool import ThreadPool

# Other files
import matrixHardness
import rankSemuMutants
import analyse
import ktest_tool

sys.path.insert(0, os.path.expanduser("~/mytools/MFI-V2.0/REFACTORING"))
import magma.common.fs as magma_common_fs  # for compress and decmpress dir
import magma.statistics.algorithms as magma_stats_algo # for Venn

SEMU_ZESTI_RUN_SKIP_FAILURE = "SEMU_ZESTI_RUN_SKIP_FAILURE"

OutFolder = "OUTPUT"
KleeSemuBCSuff = ".MetaMu.bc"
ZestiBCSuff = ".Zesti.bc"
WRAPPER_TEMPLATE = None
SEMU_CONCOLIC_WRAPPER = "wrapper-call-semu-concolic.in"
ZESTI_CONCOLIC_WRAPPER = "wrapper-call-zesti-concolic.in"
MY_SCANF = None
mutantInfoFile = "mutantsInfos.json"
fdupesDuplicatesFile = "fdupes_duplicates.json"

KLEE_TESTGEN_SCRIPT_TESTS = "MFI_KLEE_TOPDIR_TEST_TEMPLATE.sh"

FilterHardToKill = "FilterHardToKill"
GenTestsToKill = "GenTestsToKill"

STDIN_KTEST_DATA_FILE = "stdin-ktest-data"

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

def ktest_fdupes(*args):
    """
    The function compute the fdupes of the klee ktest directories 
    and ktest files given as arguments. 
    It requires that the files and directories passed as arguments exist

    :param *args: each argument is either a file or a directory that exists

    :return: returns two values: 
            - The first is a python list of tuples is returned. 
                each tuple represent files duplicates with each 
                other and rank by their age (modified time) the oldest 
                first (earliest modified to latest modified).
            - The second is the list of files that are not valid
                ktest files.
    """
    ret_fdupes = []
    invalid = []
    file_set = set()
    for file_dir in args:
        if os.path.isfile(file_dir):
            file_set.add(file_dir)
        elif os.path.isdir(file_dir):
            # get ktest files recursively
            for root, directories, filenames in os.walk(file_dir):
                for filename in filenames:
                    file_set.add(os.path.join(root, filename))
        else:
            error_exit("Invalid file or dir passed (inexistant): "+file_dir)

    # apply fdupes: load all ktests and strip the non uniform data 
    # (.bc file used) then compare the remaining data
    kt2used_dat = {}
    for kf in file_set:
        try:
            b = ktest_tool.KTest.fromfile(kf)
            kt2used_dat[kf] = (b.args[1:], b.objects)
        except:
            invalid.append(kf)
        
    # do fdupes
    dup_dict = {} 
    keys = kt2used_dat.keys()
    for ktest_file in keys:
        if ktest_file in kt2used_dat:
            ktest_file_dat = kt2used_dat[ktest_file]
            del kt2used_dat[ktest_file]
            for other_file in kt2used_dat:
                if kt2used_dat[other_file] == ktest_file_dat:
                    if ktest_file not in dup_dict:
                        dup_dict[ktest_file] = []
                    dup_dict[ktest_file].append(other_file)
            if ktest_file in dup_dict:
                for dup_of_kt_file in dup_dict[ktest_file]:
                    del kt2used_dat[dup_of_kt_file]

    # Finilize
    for ktest_file in dup_dict:
        tmp = [ktest_file] + dup_dict[ktest_file]
        # sort by decreasing modified age
        tmp.sort(key=lambda x: os.path.getmtime(x))
        ret_fdupes.append(tuple(tmp))

    return ret_fdupes, invalid
#~ def ktest_fdupes()

'''
def compressDir (inDir, out_tar_filename=None, remove_inDir=False):
    if out_tar_filename is None:
        out_tar_filename = inDir + ".tar.gz"
    with tarfile.open(out_tar_filename, "w:gz") as tar_handle:
        tar_handle.add(inDir)
    assert tarfile.is_tarfile(out_tar_filename), "tar created is invalid: "+ out_tar_filename

    if remove_inDir:
        shutil.rmtree(inDir)
#~ def compressDir()

def decompressDir (in_tar_filename, outDir=None):
    if (in_tar_filename.endswith(".tar.gz")):
        if outDir is None:
            outDir = in_tar_filename[:-len('.tar.gz')]
        if os.path.isdir(outDir):
            shutil.rmtree(outDir)
        tar = tarfile.open(in_tar_filename, "r:gz")
        tar.extractall()
        tar.close()
    elif (in_tar_filename.endswith(".tar")):
        if outDir is None:
            outDir = in_tar_filename[:-len('.tar')]
        if os.path.isdir(outDir):
            shutil.rmtree(outDir)
        tar = tarfile.open(in_tar_filename, "r:")
        tar.extractall()
        tar.close()
    else:
        error_exit("Invalid tar file: "+in_tar_filename)
#~ def decompressDir()

def getCommonSetsSizes_venn (setsElemsDict, setsize_from=None, setsize_to=None, name_delim='&'):
    res_set = {}
    if setsize_from is None:
        setsize_from = 1
    if setsize_to is None:
        setsize_to = len(setsElemsDict)
    ordered_keys = list(setsElemsDict)
    for setsize in range(setsize_from, setsize_to+1):
        for set_pos in itertools.combinations(range(len(ordered_keys)), setsize):
            name_key = name_delim.join([ordered_keys[i] for i in set_pos])
            assert name_key not in res_set
            res_set[name_key] = None
            #print set_pos, len(set_pos)
            for i in set_pos:
                if res_set[name_key] is None:
                    res_set[name_key] = set(setsElemsDict[ordered_keys[i]])
                else:
                    res_set[name_key] &= setsElemsDict[ordered_keys[i]]
    #print res_set
    res_num = {}
    for v in res_set:
        res_num[v] = len(res_set[v])
    
    return res_num
#~ def getCommonSetsSizes_venn()
'''

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

def runZestiOrSemuTC (unwrapped_testlist, devtests, exePath, runtestScript, kleeZestiSemuInBCLink, outpdir, zestiexedir, llvmgcc_exe_dir, llvm27_exe_dir): #, mode="zesti+symbex"):
    test2outdirMap = {}

    if not os.path.isfile(kleeZestiSemuInBCLink):
        error_exit("Error: Zesti .bc file is missing. maybe should rerun 'prepare'. missing file is: "+kleeZestiSemuInBCLink)

    # copy bc
    kleeZestiSemuInBC = os.path.basename(kleeZestiSemuInBCLink) 
    #if mode == "zesti+symbex":
    print "# Extracting tests Infos with ZESTI\n"
    cmd = "llvm-gcc" if llvmgcc_exe_dir is None else os.path.join(llvmgcc_exe_dir, "llvm-gcc")
    cmd += " -c -std=c89 -emit-llvm "+MY_SCANF+" -o "+MY_SCANF+".bc"
    ec = os.system(cmd)
    if ec != 0:
        error_exit("Error: failed to compile my_scanf to llvm for Zesti. Returned "+str(ec)+".\n >> Command: "+cmd)
    ec = os.system(("llvm-link" if llvm27_exe_dir is None else os.path.join(llvm27_exe_dir, "llvm-link"))+' '+kleeZestiSemuInBCLink+" "+MY_SCANF+".bc -o "+os.path.join(outpdir, kleeZestiSemuInBC))
    if ec != 0:
        error_exit("Error: failed to link myscanf and zesti bc. Returned "+str(ec))
    os.remove(MY_SCANF+".bc")

    # Verify that Zesti is accessible
    zextmp = "klee"
    if zestiexedir is not None:
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
        semuExecLog = os.path.join(outpdir, "semu.out")
        # Run Semu with tests (wrapper is installed)
        print "# Running Tests", tc, "..."
        zestCmd = " ".join(["bash", runtestScript, tc, testrunlog])
        retCode = os.system(zestCmd)
        nNew = len(glob.glob(os.path.join(outpdir, "klee-out-*")))
        if nNew == nKleeOut:
            print ">> problem with command: "+ zestCmd

            if SEMU_ZESTI_RUN_SKIP_FAILURE in os.environ and os.environ[SEMU_ZESTI_RUN_SKIP_FAILURE].strip().lower() == 'on' :
                # avoid failure by creating an invalid test that will be skipped
                fix_dir = os.path.join(outpdir, "klee-out-"+str(nNew))
                os.mkdir(fix_dir)
                with open(os.path.join(fix_dir, 'test000001.ktest'), 'w') as f:
                    f.write(">> There was a failure: "+"Test execution failed for test case '"+tc+"', retCode was: "+str(retCode)+'\n')
                # update klee-last
                if os.path.islink(os.path.join(outpdir, 'klee-last')):
                    os.remove(os.path.join(outpdir, 'klee-last'))
                os.symlink(fix_dir, os.path.join(outpdir, 'klee-last'))
                # update nNew
                nNew += 1
            else:
                error_exit ("Test execution failed for test case '"+tc+"', retCode was: "+str(retCode))
        assert nNew > nKleeOut, "Test was not run: "+tc

        # Premissions on changing output klee stuffs
        os.chmod(outpdir, 0o777)
        #for kfile in glob.iglob(outpdir+"/klee-*"):
        #    os.chmod(kfile, 0o777)

        for devtid, kleetid in enumerate(range(nKleeOut, nNew)):
            kleeoutdir = os.path.join(outpdir, 'klee-out-'+str(kleetid))
            # Check that the kleeoutdir has right ownership, otherwise set
            if os.stat(kleeoutdir).st_uid != os.geteuid():
                if os.system(" ".join(["sudo chown -R --reference", outpdir, kleeoutdir])) != 0:
                    error_exit ("Failed to set ownership of kleeoutdir of roottest")
            # Remove everything from kleeoutdir, but the ktest
            for fname in glob.iglob(kleeoutdir+'/*'):
                if not (fname.endswith('.ktest') or os.path.basename(fname) == STDIN_KTEST_DATA_FILE):
                    os.remove(fname)
            wrapTestName = os.path.join(tc.replace('/', '_') + "-out", "Dev-out-"+str(devtid), "devtest.ktest")
            if not len(glob.glob(kleeoutdir+'/*.ktest')) > 0:
                print "## Did not find ktest in folder", kleeoutdir 
                #wait_creation = raw_input(". can you see it manually? [y/n]")
                time.sleep(5)
            if len(glob.glob(kleeoutdir+'/*.ktest')) <= 0:
                assert os.path.isfile(semuExecLog), "Semu exec log not found"
                with open(semuExecLog) as f:
                    cantgentest = False
                    kleedone = False
                    for line in f:
                        if line.strip() == "KLEE: WARNING: unable to get symbolic solution, losing test case":
                            cantgentest = True
                        elif line.strip().startswith("KLEE: done: generated tests = 1"):
                            kleedone = True
                    if cantgentest and kleedone:
                        shutil.copy2 (semuExecLog, os.path.join(kleeoutdir, 'test000001.ktest'))

            if len(glob.glob(kleeoutdir+'/*.ktest')) <= 0:
                if SEMU_ZESTI_RUN_SKIP_FAILURE in os.environ and os.environ[SEMU_ZESTI_RUN_SKIP_FAILURE].strip().lower() == 'on' :
                    with open(os.path.join(kleeoutdir, 'test000001.ktest'), 'w') as f:
                        f.write(">> There was a failure: "+"No ktest was generated for "+wrapTestName+". Folder is: "+kleeoutdir+". ZEST CMD: "+zestCmd+'\n')
                else:
                    assert False, "No ktest was generated for "+wrapTestName+". Folder is: "+kleeoutdir+". ZEST CMD: "+zestCmd

            test2outdirMap[wrapTestName] = kleeoutdir
        # update
        nKleeOut = nNew

        if os.path.isfile(semuExecLog):
            os.remove(semuExecLog)
    for wtc in devtests:
        if wtc not in test2outdirMap:
            if SEMU_ZESTI_RUN_SKIP_FAILURE in os.environ and os.environ[SEMU_ZESTI_RUN_SKIP_FAILURE].strip().lower() == 'on' :
                # avoid failure by creating an invalid test that will be skipped
                fix_dir = os.path.join(outpdir, "klee-out-"+str(nKleeOut))
                os.mkdir(fix_dir)
                with open(os.path.join(fix_dir, 'test000001.ktest'), 'w') as f:
                    f.write(">> There was a failure: "+"test not in Test2SemuoutdirMap: \nMap: "+str(test2outdirMap)+"\nTest: "+wtc+'\n')
                test2outdirMap[wtc] = fix_dir
                nKleeOut += 1
            else:
                print "Error:", "test not in Test2SemuoutdirMap: \nMap: "+str(test2outdirMap)+"\nTest: "+wtc
                print "\n Could run with SEMU_ZESTI_RUN_SKIP_FAILURE=on env var to neglect the error."
                assert False 
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
    where each argument is represented by a pair of argtype (argv or file or stdin and the corresponding sizes)
    IN KLEE, sym-files is taken into account only once (the last one)
'''
def parseZestiKtest(filename, test2zestidirMap_arg=None):

    datalist = []
    b = ktest_tool.KTest.fromfile(filename)
    # get the object one at the time and obtain its stuffs
    # the objects are as following: [model_version, <the argvs> <file contain, file stat>]
    # Note that files appear in argv. here we do not have n_args because concrete (from Zesti)

    # XXX Watch this: TMP: make sure all files are at the end
    firstFile_ind = -1
    postFileArgv_ind = []
    assert b.objects[0][0] == 'model_version', "Invalid model_version position for file: "+filename
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
            if len(indexes_ia) > 0 or name != "argv": # filename in args, the corresponding position in datalist is indexes_ia, or name is not argv but is containe. example "--file=in1" TODO
                # in case the same name appears many times in args, let the user manually verify
                if len(indexes_ia) > 1:
                    if test2zestidirMap_arg is not None:
                        actual_test = None
                        for at in test2zestidirMap_arg:
                            if os.path.dirname(filename).endswith(test2zestidirMap_arg[at]):
                                actual_test = at
                                break
                        print "\n>> CONFLICT: the file object at position ",ind,"with name","'"+name+"'","in ktest",filename,"appears several times in args list (The actual test is:", actual_test,")."
                    else:
                        print "\n>> CONFLICT: the file object at position ",ind,"with name","'"+name+"'","in ktest",filename,"appears several times in args list (Check OUTPUT/caches/test2zestidirMap.json for actual test)."
                    print "    >> Please choose its space separated position(s), (",indexes_ia,"):"
                    raw = raw_input()
                    indinargs = [int(v) for v in raw.split()]
                    assert len(set(indinargs) - set(indexes_ia)) == 0, "input wrong indexes. do not consider program name"
                elif len(indexes_ia) == 1:
                    indinargs = indexes_ia
		else: # name != "argv"
                    indexes_ia = [i for i,x in enumerate(b.args[1:]) if name in x]
                    if len(indexes_ia) <= 0:
                        if SEMU_ZESTI_RUN_SKIP_FAILURE in os.environ and os.environ[SEMU_ZESTI_RUN_SKIP_FAILURE].strip().lower() == 'on' :
                            pass
                        else:
                            print "Error: Must have at least one argv containing filename in its data"
                            print "\n Could run with SEMU_ZESTI_RUN_SKIP_FAILURE=on env var to neglect the error."
                            assert False
                    if len(indexes_ia) > 1:
                        if test2zestidirMap_arg is not None:
                            actual_test = None
                            for at in test2zestidirMap_arg:
                                if os.path.dirname(filename).endswith(test2zestidirMap_arg[at]):
                                    actual_test = at
                                    break
                            print "\n>> HINT NEEDED: the file object at position ",ind,"with name",name,"in ktest",filename,"has file with complex argv (The actual test is:", actual_test,")."
                        else:
                            print "\n>> HINT NEEDED: the file object at position ",ind,"with name",name,"in ktest",filename,"has file with complex argv (Check OUTPUT/caches/test2zestidirMap.json for actual test)."
                        print "    >> Please choose its space separated position(s), (",indexes_ia,"):"
                        raw = raw_input()
                        indinargs = [int(v) for v in raw.split()]
                        assert len(set(indinargs) - set(indexes_ia)) == 0, "input wrong indexes. do not consider program name"
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
                assert name == "argv", "name ("+name+") not in args and not argv: "+filename
                datalist.append(('ARGV', len(data))) #XXX

    if len(filesNstatsIndex) > 0:
        assert filesNstatsIndex == range(filesNstatsIndex[0], filesNstatsIndex[-1]+1), "File objects are not continuous: (File "+filename+"): "+str(filesNstatsIndex)+str(range(filesNstatsIndex[0], filesNstatsIndex[-1]+1))

    if model_version_pos == 0:
        #for ii in range(len(fileargsposinObj_remove)):
        #    for iii in range(len(fileargsposinObj_remove[ii])):
        #        fileargsposinObj_remove[ii][iii] += 1
        # Do bothing for fileargsposinObj_remove because already indexed to not account for model version XXX
        if len(fileargsposinObj_remove) > 0:
            if len(fileargsposinObj_remove[-1]) > 0:
                assert max(fileargsposinObj_remove[-1]) < filesNstatsIndex[0], "arguments do not all come before files in object"
        filesNstatsIndex = [(v - 1) for v in filesNstatsIndex] #-1 for model_versio which will be move to end later

        # stdin and after files obj
        if stdin is not None:
            afterLastFilenstatObj = max(filesNstatsIndex) + 1 if len(filesNstatsIndex) > 0 else (len(b.objects) - 2 -1) # -2 because of stdin and stdin-stat, -1 for mode_version
            datalist.append(stdin)
        else:
            afterLastFilenstatObj = max(filesNstatsIndex) + 1 if len(filesNstatsIndex) > 0 else (len(b.objects) - 1) # -1 for model_version
            # shadow-zesti have problem with stdin, use our hack on wrapper to capture that
            stdin_file = os.path.join(os.path.dirname(filename), STDIN_KTEST_DATA_FILE)
            assert os.path.isfile(stdin_file), "The stdin exported in wrapper is missing for test: "+filename
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

def is_sym_args_having_nargs(sym_args, check_good=False):
    key_w, min_n_arg, max_n_arg, maxlen = sym_args.strip().split()
    min_n_arg, max_n_arg, maxlen = map(int, (min_n_arg, max_n_arg, maxlen))
    if check_good:
        assert "-sym-args" in key_w, "Invalid key_w, must be having '-sym-args '"
        assert min_n_arg <= max_n_arg, "error: min_n_arg > max_n_arg. (bug)"
    if min_n_arg < max_n_arg:
        return True
    return False
# def is_sym_args_having_nargs()

def getSymArgsFromZestiKtests (ktestFilesList, test2zestidirMap_arg, argv_becomes_arg_i=False, add_sym_stdout=False):
    testNamesList = test2zestidirMap_arg.keys()
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
        b_tmp = ktest_tool.KTest.fromfile(ktestfile)
        if len(b_tmp.objects) == 0:
            print "@WARNING: Skipping test because Zesti generated empty KTEST file:", ktestfile
            continue

        # sed because Zesti give argv, argv_1... while sym args gives arg0, arg1,...
        ktestdat, testArgs, fileNstatInd, maxFsize, fileargind, afterFnS = parseZestiKtest(ktestfile, test2zestidirMap_arg)
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
                    # [:-1] in ktdat.objects[fai][1][-1] because of last '\0'
                    if ktdat.objects[fai][1][:-1] == ktdat.objects[filenstatsInObj[ktpos][2*ind_fai]][0]:
                        ktdat.objects[fai] = (ktdat.objects[fai][0], shortFnames[ind_fai])
                    else:
                        #print len(ktdat.objects[fai][1]) ,len(ktdat.objects[filenstatsInObj[ktpos][2*ind_fai]][0])
                        #print list(ktdat.objects[fai][1]) ,list(ktdat.objects[filenstatsInObj[ktpos][2*ind_fai]][0])
			print "\n>> MANUAL REPLACE: the final file name is '"+shortFnames[ind_fai]+"'.","initial name is '"+ktdat.objects[filenstatsInObj[ktpos][2*ind_fai]][0]+"'."
			print "  >> Test name is:", ktestContains["CORRESP_TESTNAME"][ktpos]
			print "    >> Please replace initial file name with new in '"+ktdat.objects[fai][1]+"' :"
			raw = raw_input().strip()
                        ktdat.objects[fai] = (ktdat.objects[fai][0], raw)

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

    if add_sym_stdout:
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
                maxlen = int(commonArgs[apos].strip().split()[-1])
                for sharedarg_i in range(commonArgsNumPerTest[t][apos]):
                    curlen = len(ktestContains["KTEST-OBJ"][t].objects[objpos + sharedarg_i][1])
                    curval = ktestContains["KTEST-OBJ"][t].objects[objpos + sharedarg_i]
                    ktestContains["KTEST-OBJ"][t].objects[objpos + sharedarg_i] = (curval[0], curval[1]+'\0'*(maxlen-curlen+1)) #+1 Because last zero added after sym len

                # Insert n_args
                ## XXX No insertion of n_args if min_n_arg and max_n_arg are equal
                if is_sym_args_having_nargs(commonArgs[apos], check_good=True):
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
                pass #TODO handle the case of files (NB: check above how the files are recognized from zesti tests (size may not be 0)

    # Change the args list in each ktest object with the common symb args 
    for ktdat in ktestContains["KTEST-OBJ"]:
        ktdat.args = ktdat.args[:1]
        for s in commonArgs:
            ktdat.args += s.strip().split(' ') 

    # Change all argv keywords into arg<i>
    if argv_becomes_arg_i:
        for ktdat in ktestContains["KTEST-OBJ"]:
            i_ = 0
            for objpos, (name, data) in enumerate(ktdat.objects):
                if name != "argv":
                    continue
                ktdat.objects[objpos] = ('arg'+str(i_), data)
                i_ += 1

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

    # list_new_sym_args = [(<nmin>,<nmax>,<size>),...]
    def old2new_cmdargs(objSegment, list_new_sym_args, old_has_nargs):
        res = []
        nargs = len(objSegment) - int(old_has_nargs) # First obj may be n_args object
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
            if is_sym_args_having_nargs(" ".join(['-sym-args']+list(map(str, list_new_sym_args[i])))):
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
            else:
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
                    assert is_sym_args_having_nargs(sa), "min_n_args and max_n_arg must be different here"
                    kt_obj.insert(obj_ind, ("n_args", struct.pack('<i', 0)))
                    obj_ind += 1
            else:
                assert isCmdArg(kt_obj[obj_ind][0]), "Supposed to be CMD arg: "+str(kt_obj[obj_ind][0])
                if '-sym-arg ' in argvinfo['old'][arg_ind]:
                    assert len(argvinfo['new'][arg_ind]) == 1, "must be one sym-args here"
                    if is_sym_args_having_nargs(argvinfo['new'][arg_ind][0]):
                        kt_obj.insert(obj_ind, ("n_args", struct.pack('<i', 1)))
                        obj_ind += 1 #Go after n_args

                    # Update the object len to the given in sym-args
                    old_len_ = len(kt_obj[obj_ind][1])
                    new_len_ = argvinfo_new_extracted[arg_ind][0][2] + 1
                    if old_len_ < new_len_:
                        kt_obj[obj_ind] = (kt_obj[obj_ind][0], kt_obj[obj_ind][1] + '\0'*(new_len_ - old_len_))
                    else:
                        assert old_len_ == new_len_, "Error: new arg len lower than pld len (BUG)"
                    obj_ind += 1 #Go after argv(arg)
                else: #sym-args
                    if is_sym_args_having_nargs(argvinfo['old'][arg_ind]):
                        assert kt_obj[obj_ind][0] == 'n_args', "must be n_args here"
                        nargs = struct.unpack('<i', kt_obj[obj_ind][1])[0]
                        old_has_nargs = True
                    else:
                        nargs = int(argvinfo['old'][arg_ind].strip().split()[1]) #n_args == min_n_arg == max_n_arg
                        old_has_nargs = False

                    if len(argvinfo['new'][arg_ind]) > 1 or argvinfo['old'][arg_ind] != argvinfo['new'][arg_ind][0]:
                        # put the args enabled as match to the right most in new
                        tmppos_last = obj_ind + nargs 
                        replacement = old2new_cmdargs(kt_obj[obj_ind:tmppos_last+1], argvinfo_new_extracted[arg_ind], \
                                                                                        old_has_nargs=old_has_nargs)
                        kt_obj[obj_ind:tmppos_last+1] = replacement
                        obj_ind += len(replacement)
                    else:
                        obj_ind += nargs + int(old_has_nargs) #+1 for n_args obj

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
                    argv_zest['new'][z_ind:] = [[v] for v in ineqs]
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
                    argv_klee['new'][k_ind:] = [[v] for v in ineqs]
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
def executeSemu (semuOutDirs, semuSeedsDir, metaMutantBC, candidateMutantsFiles, symArgs, semuexedir, tuning, mergeThreadsDir=None, exemode=FilterHardToKill): #="zesti+symbex"):
    # Prepare the seeds to use
    threadsOutTop = os.path.dirname(semuOutDirs[0])
    if os.path.isdir(threadsOutTop):
        shutil.rmtree(threadsOutTop)
    os.mkdir(threadsOutTop)

    assert len(candidateMutantsFiles) == len(semuOutDirs), "Missmatch between number of candidate mutant files and number of outputs folders: "+str(len(candidateMutantsFiles))+" VS "+str(len(semuOutDirs))

    if tuning['name'] == '_pureklee_' and 'SEMU' not in tuning:
        assert exemode == GenTestsToKill, "Must be Test generation mode for _pureklee_"
        isPureKLEE = True
    else:
        isPureKLEE = False

    if isPureKLEE:
        # Use simple bc file
        metaMutantBC = metaMutantBC[:-len('.MetaMu.bc')]+'.bc'
        nMutants = sum([sum(1 for line_ in open(mlist)) for mlist in candidateMutantsFiles])
        nThreads = 1
    else:
        nThreads = len(candidateMutantsFiles)

    filter_mutestgen = "" if exemode == FilterHardToKill else " -semu-max-tests-gen-per-mutant="+str(tuning['EXTRA']['MaxTestsPerMutant']) # num of test per mutant
    if tuning['EXTRA']['-semu-testsgen-only-for-critical-diffs']:
        filter_mutestgen += " -semu-testsgen-only-for-critical-diffs"
    if tuning['EXTRA']['-semu-continue-mindist-out-heuristic']:
        filter_mutestgen += " -semu-continue-mindist-out-heuristic"
    if tuning['EXTRA']['-semu-disable-statediff-in-testgen']:
        filter_mutestgen += " -semu-disable-statediff-in-testgen"


    # Copy the metaMutantBC file into semu semuSeedsDir (will be remove when semuSeedsDir is removed bellow)
    # Avoid case where klee modifies the BC file and don't have backup
    metaMutantBCFilePath = os.path.join(semuSeedsDir, os.path.basename(metaMutantBC))
    shutil.copy2(metaMutantBC, metaMutantBCFilePath)


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

    pending_threads = range(nThreads)
    enable_semu_fork_externalcall = False
    while len(pending_threads) > 0:
        runSemuCmds = []
        for thread_id in pending_threads:
            candidateMutantsFile = candidateMutantsFiles[thread_id]
            semuOutDir = semuOutDirs[thread_id]
            logFile = semuOutDir+".log"

            if os.path.isdir(semuOutDir):
                shutil.rmtree(semuOutDir)

            kleeArgs = "-allow-external-sym-calls -libc=uclibc -posix-runtime -search=bfs"
            kleeArgs += ' ' + " ".join([par+'='+str(tuning['KLEE'][par]) for par in tuning['KLEE']])  #-max-time=50000 -max-memory=9000 --max-solver-time=300
            kleeArgs += " -max-sym-array-size=4096 --max-instruction-time=10. -use-cex-cache " # -watchdog"
            kleeArgs += " --output-dir="+semuOutDir
            semukleearg = "-seed-out-dir="+semuSeedsDir
            if isPureKLEE:
                #semuArgs = ""
                semuArgs = "-stop-after-n-tests="+str(tuning['EXTRA']['MaxTestsPerMutant'] * nMutants)
                semuArgs += " -only-output-states-covering-new"
                semuExe = "klee"
            else:
                semukleearg += " -only-replay-seeds" #make sure that the states not of seed are removed
                semuArgs = " ".join([par+'='+str(tuning['SEMU'][par]) for par in tuning['SEMU']])  #" ".join(["-semu-precondition-length=3", "-semu-mutant-max-fork=2"])
                #semuArgs += " " + " ".join(["-semu-precondition-file="+prec for prec in symbexPreconditions])
                semuArgs += filter_mutestgen
                if candidateMutantsFile is not None:
                    semuArgs += " -semu-candidate-mutants-list-file " + candidateMutantsFile
                if enable_semu_fork_externalcall:
                    semuArgs += " -semu-forkprocessfor-segv-externalcalls"
                semuExe = "klee-semu"

            semuExe = semuExe if semuexedir is None else os.path.join(semuexedir, semuExe)
            runSemuCmd = " ".join([semuExe, kleeArgs, semukleearg, semuArgs, metaMutantBCFilePath, " ".join(symArgs), "> /dev/null"]) #,"2>&1"])
            #sretcode = os.system(runSemuCmd)
            # Timeout (Since watchdog is not use, ensure timeout with timeout (Future, use subprocess.call... of puthon3))
            max_time_argument = str(float(tuning['KLEE']['-max-time']) + 30) #3600)
            runSemuCmd = " ".join(["timeout --foreground --kill-after=600s", max_time_argument, runSemuCmd])
            runSemuCmd += " 2>"+logFile
            runSemuCmds.append(runSemuCmd)

        print '['+time.strftime("%c")+']', "## Executing", "pure KLEE" if isPureKLEE else "SEMU", '('+tuning['name']+')', "with", len(pending_threads), \
                                                                                                "parallel threads. Execution log in <semu_outputs/Thread-<i>.log>"
        if len(pending_threads) > 1:
            threadpool = ThreadPool(len(pending_threads))
            sretcodes = threadpool.map(os.system, runSemuCmds)
            threadpool.terminate()
            threadpool.close()
            threadpool.join()
        else:
            sretcodes = map(os.system, runSemuCmds)

        # get the actual returned code (os.system return not only the code but also the signal)
        sretcodes = [os.WEXITSTATUS(os_sys_ret) for os_sys_ret in sretcodes]

        failed_thread_executions = []
        for thread_id, sretcode in zip(pending_threads, sretcodes):
            if sretcode != 0 :#and sretcode != 256: # 256 for watchdog timeout
                if sretcode != 124 and sretcode != 137: # timeout and kill(9)
                    failed_thread_executions.append((thread_id, sretcode))

        if len(failed_thread_executions) > 0:
            common_err = failed_thread_executions[0][1]
            for v in failed_thread_executions:
                if v[1] != common_err:
                    common_err = None
                    break
            if enable_semu_fork_externalcall or common_err is None or common_err != 11: 
                # already reran or errors are not all same and not sigsegv(11)
                print "# Execution failed for", len(failed_thread_executions), "threads:"
                for thread_id, sretcode in failed_thread_executions:
                    print "-- Returned Code:", sretcode, ", for thread", thread_id,". Command: ", runSemuCmds[thread_id]
                #error_exit("Error: klee-semu symbex failled! (name is: "+tuning['name']+")") # with code "+str(sretcode))
                return ("@@Error: klee-semu symbex failled! (name is: "+tuning['name']+")") # with code "+str(sretcode))
            else:
                # The execution failed for some threads, probably due to Seg Fault during external call.
                # Rerun those thread in mode where a new process is forked everytime
                pending_threads = [v[0] for v in failed_thread_executions]
                enable_semu_fork_externalcall = True
                print '['+time.strftime("%c")+']', "##", "pure klee" if isPureKLEE else "klee-semu", '('+tuning['name']+')', "failed for the threads:", pending_threads, \
                                                                                            "\n    >> re-executing them by forking process for external calls"
        else:
            # All sucessfull, end
            pending_threads = []
            print '['+time.strftime("%c")+']', "##", "pure klee" if isPureKLEE else "klee-semu", '('+tuning['name']+')', "execution is done!"

    # In case of pureKLEE (testgen mode), we need the file containing mutants tests
    if isPureKLEE: 
        for semuOutDir in semuOutDirs:
            mktlistfile = os.path.join(semuOutDir, "mutant-0.ktestlist")
            df_ktl = []
            inittime = os.path.getctime(os.path.join(semuOutDir, "assembly.ll"))
            for ktp in glob.glob(os.path.join(semuOutDir, "*.ktest")):
                kttime = os.path.getctime(ktp) - inittime
                assert kttime >= 0, "negative kttime w.r.t assembly.ll, for ktest: "+ ktp
                df_ktl.append({"MutantID":0, 'ktest': os.path.basename(ktp), "ellapsedTime(s)": kttime})

            # Handle case where not test is generated
            if len(df_ktl) == 0:
                df_ktl = {"MutantID":[], 'ktest':[], "ellapsedTime(s)":[]}

            df_ktl = pd.DataFrame(df_ktl)
            df_ktl.to_csv(mktlistfile, index=False)


    if mergeThreadsDir is not None:
        if os.path.isdir(mergeThreadsDir):
            shutil.rmtree(mergeThreadsDir)
        os.mkdir(mergeThreadsDir)
        thread_data_map = {}

        if exemode == FilterHardToKill:
            thread_data_map_filename = os.path.join(mergeThreadsDir, "thread_data_map.jsom")
            for thread_id in range(nThreads):
                thread_data_map[thread_id] = []
                for mutoutfp in glob.glob(os.path.join(semuOutDirs[thread_id],"mutant-*.semu")):
                    thread_data_map[thread_id].append(os.path.basename(mutoutfp))
                    merg_mutoutfp = os.path.join(mergeThreadsDir, os.path.basename(mutoutfp))
                    assert not os.path.isfile(merg_mutoutfp), "Same mutant was treated in different threads (BUG). Mutant id file is: "+os.path.basename(merg_mutoutfp)
                    # copy into merge, adding thread id to state (which is an address) to avoid considering different states with same address du to difference in execution threads, as one
                    with open(merg_mutoutfp, "w") as f_out:
                        with open(mutoutfp, "r") as f_in:
                            for line in f_in:
                                f_out.write(line.replace(',0x', ','+str(thread_id)+'_0x'))
            # store thread data map
            dumpJson (thread_data_map, thread_data_map_filename)

            # Remove threads data dirs
            for thread_id in range(nThreads):
                shutil.rmtree(semuOutDirs[thread_id])
        else:
            error_exit ("Merge thread is set within semuExecution only for FilterHardToKill mode")

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

    return None
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
    take the ktests in the folders of semuoutputs, then put then together removing duplicates
    The result is put in newly created dir mfi_ktests_dir. The .ktestlist files of each mutant are updated
'''
def fdupeGeneratedTest (mfi_ktests_dir_top, mfi_ktests_dir, semuoutputs, seeds_dir=None):
    assert mfi_ktests_dir_top in mfi_ktests_dir, "mfi_ktests_dir_top not in mfi_ktests_dir"
    if os.path.isdir(mfi_ktests_dir_top):
        shutil.rmtree(mfi_ktests_dir_top)
    os.makedirs(mfi_ktests_dir)
    ktests = {}
    for fold in semuoutputs:
        kt_fold = glob.glob(fold+"/*.ktest")
        mut_fold = glob.glob(fold+"/mutant-*.ktestlist")
        for ktp in kt_fold:
            assert ktp not in ktests, "ktp not in ktests. fold is "+fold+" ktp is "+ktp
            ktests[ktp] = []

        # XXX handle case where klee-semu could not output a test case
        bad2good_ktest_map = {}
        ktest_goodid_list = sorted([int(os.path.basename(kt).replace('.ktest', '').replace('test', '')) for kt in ktests])
        for bad_id_1, good_id in enumerate(ktest_goodid_list):
            bad_id = bad_id_1 + 1
            bad_kt = 'test%06d.ktest' % bad_id
            good_kt = 'test%06d.ktest' % good_id
            bad2good_ktest_map[bad_kt] = good_kt

        # get data for mutants
        for minf in mut_fold:
            df = pd.read_csv(minf)
            for index, row in df.iterrows():
                assert row["ktest"].endswith(".ktest"), "Invalid minf file: "+minf
                et = row["ellapsedTime(s)"]
                mid = row["MutantID"]
                if row["ktest"] in bad2good_ktest_map:
                    ktp = os.path.join(fold, bad2good_ktest_map[row["ktest"]])
                    assert ktp in ktests, "test not in ktests: "+str(ktests)+",\n test (not in ktests): "+ktp+"; minf is: "+minf
                    ktests[ktp].append((mid, et))
                    

    # Verify that each ktest in ktests has corresponding mutant
    if len([kt for kt in ktests if len(ktests[kt]) == 0]) > 0:
        error_exit("Some ktests are not present as belonging to mutant: "+str([kt for kt in ktests if len(ktests[kt]) == 0]))
                  
    # Use fdupes across the dirs in semuoutputs to remove duplicates and update test infos
    seed_dup_kts = []
    dup_result, non_kt_files = ktest_fdupes(seeds_dir, *semuoutputs)
    for la in dup_result:
        #assert la[0] not in dupmap, "fdupe line: "+la[0]+", is not in dupmap: "+str(dupmap)
        val_in_ktd = []
        val_in_seed = []
        for ii in range(len(la)):
            if la[ii].endswith('.ktest'):
                if os.path.abspath(seeds_dir) == os.path.dirname(os.path.abspath(la[ii])):
                    val_in_seed.append(la[ii])
                else:
                    val_in_ktd.append(la[ii])
        if len(val_in_seed) > 0:
            seed_dup_kts += val_in_ktd
            # discard tests dup of seeds
            for dpkt in val_in_ktd:
                del ktests[dpkt]
        else:
            if len(val_in_ktd) > 0:
                remain = val_in_ktd[0]
                dups = val_in_ktd[1:]
                assert remain in ktests, "remain not in ktests. remain is "+remain
                for dpkt in dups:
                    ktests[remain] += ktests[dpkt]
                    del ktests[dpkt]
    
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

    # copy 'info' file
    if len(semuoutputs) > 0:
        shutil.copy2(os.path.join(semuoutputs[0], "info"), mfi_ktests_dir)

    etdf = pd.DataFrame(etimeObj)
    etdf.to_csv(os.path.join(mfi_ktests_dir, "tests_by_ellapsedtime.csv"), index=False)
    dumpJson(finalObj, os.path.join(mfi_ktests_dir, "mutant_ktests_mapping.json"))
    dumpJson(seed_dup_kts, os.path.join(mfi_ktests_dir, "seed_dup_ktests.json"))

    # Handle semu exec info files
    info_dats = []
    out_id = -1
    for fold in semuoutputs:
        out_id += 1
        semu_exec_info_file = os.path.join(fold, 'semu_execution_info.csv') 
        if os.path.isfile(semu_exec_info_file):
            df = pd.read_csv(semu_exec_info_file)
            for index, row in df.iterrows():
                info_dats.append((out_id, row))

    info_dats.sort(key=lambda x: float(x[1]['ellapsedTime(s)']))
    
    final_info_file = os.path.join(mfi_ktests_dir, 'semu_execution_info.csv')
    if len(info_dats) == 0:
        final_info_list = {
                        "ellapsedTime(s)":[], "stateCompareTime(s)":[],
                        "#MutStatesForkedFromOriginal":[],"#MutStatesEqWithOrigAtMutPoint":[]
                    }
    else:
        final_info_list = []

    pointers = {v: {
                    "ellapsedTime(s)":None, "stateCompareTime(s)":0,
                    "#MutStatesForkedFromOriginal":0,"#MutStatesEqWithOrigAtMutPoint":0
                    } for v in range(out_id+1)
                }
    for out_id, row in info_dats:
        pointers[out_id] = row
        tmp_obj = {"ellapsedTime(s)": row["ellapsedTime(s)"]}
        for metric in ["stateCompareTime(s)", "#MutStatesForkedFromOriginal", \
                                                "#MutStatesEqWithOrigAtMutPoint"]:
            tmp_obj[metric] = sum(pointers[v][metric] for v in pointers)
        if len(final_info_list) > 0:
            if final_info_list[-1]["ellapsedTime(s)"] == tmp_obj["ellapsedTime(s)"]:
                final_info_list[-1] = tmp_obj
            else:
                final_info_list.append(tmp_obj)
        else:
            final_info_list.append(tmp_obj)

    info_df = pd.DataFrame(final_info_list)
    info_df.to_csv(final_info_file, index=False)
#~ def fdupeGeneratedTest ()

def fdupesAggregateKtestDirs (mfi_ktests_dir_top, mfi_ktests_dir, inKtestDirs, names):
    assert len(inKtestDirs) == len(names)
    assert len(set(names)) == len(names), "There should be no redundancy in names"
    assert mfi_ktests_dir_top in mfi_ktests_dir
    if os.path.isdir(mfi_ktests_dir_top):
        shutil.rmtree(mfi_ktests_dir_top)
    os.makedirs(mfi_ktests_dir)

    ktestsPre2Post = {ktd: {} for ktd in inKtestDirs}

    # Use fdupes across the dirs in semuoutputs to remove duplicates and update test infos
    redundancesMap = {}
    fdupesout = mfi_ktests_dir+".tmp"
    fdupcmd = " ".join(["fdupes -1"]+inKtestDirs+[">",fdupesout])
    if os.system(fdupcmd) != 0:
        error_exit ("fdupes failed. cmd: "+fdupcmd)
    assert os.path.isfile(fdupesout), "Fdupes failed to produce output"
    with open(fdupesout) as fp:
        for line in fp:
            la = line.strip().split()
            if la[0].endswith('.ktest'): # do not consider other possible non ktest duplicates
                redundancesMap[la[0]] = la[1:]
    os.remove(fdupesout)

    nonredundances = set()
    for iktd in inKtestDirs:
        nonredundances |= set(glob.glob(os.path.join(iktd, '*.ktest')))
    for v in redundancesMap:
        nonredundances -= set(redundancesMap[v])

    oldnewnamemap = {}
    # Copy non duplicates into mfi_ktests_dir
    testid = 1
    for ktp in nonredundances:
        newtname = "test"+str(testid)+".ktest"
        testid += 1
        shutil.copy2(ktp, os.path.join(mfi_ktests_dir, newtname))
        oldnewnamemap[ktp] = newtname

    # update the redundant keys
    for v in redundancesMap:
        for rv in redundancesMap[v]:
            assert rv not in oldnewnamemap
            oldnewnamemap[rv] = oldnewnamemap[v]

    # transform oldnewnamemap into ktestsPre2Post
    for vv in oldnewnamemap:
        iktd = os.path.dirname(vv)
        kk = os.path.basename(vv)
        ktestsPre2Post[iktd][kk] = oldnewnamemap[vv]

    # Finalize metadata
    for i in range(len(inKtestDirs)):
        etdf = pd.read_csv(os.path.join(inKtestDirs[i], "tests_by_ellapsedtime.csv"))
        in_finalObj = loadJson(os.path.join(inKtestDirs[i], "mutant_ktests_mapping.json"))
        in_seed_dup_kts = loadJson(os.path.join(inKtestDirs[i], "seed_dup_ktests.json"))
        semu_info_df = pd.read_csv(os.path.join(inKtestDirs[i], "semu_execution_info.csv"))
        finalObj = {}

        tmp_ktest_col = []
        for index, row in etdf.iterrows():
            tmp_ktest_col.append(ktestsPre2Post[inKtestDirs[i]][row['ktest']])
        etdf['ktest'] = tmp_ktest_col
        for kt in in_finalObj:
            assert ktestsPre2Post[inKtestDirs[i]][kt] not in finalObj
            finalObj[ktestsPre2Post[inKtestDirs[i]][kt]] = in_finalObj[kt]

        assert set(finalObj) == set(etdf['ktest']), "BUG: mismatch between test and by muts: "+\
                                str(set(finalObj))+" VS "+str(set(etdf['ktest']))

        # copy 'info' file
        if os.path.isfile(os.path.join(inKtestDirs[i], 'info')):
            shutil.copy2(os.path.join(inKtestDirs[i], "info"), os.path.join(mfi_ktests_dir, names[i]+'-info'))

        etdf.to_csv(os.path.join(mfi_ktests_dir, names[i]+"-tests_by_ellapsedtime.csv"), index=False)
        dumpJson(finalObj, os.path.join(mfi_ktests_dir, names[i]+"-mutant_ktests_mapping.json"))
        dumpJson({"Number of seed duplicates removed": len(in_seed_dup_kts)}, os.path.join(mfi_ktests_dir, names[i]+"-seed_dup_ktests.json"))
        semu_info_df.to_csv(os.path.join(mfi_ktests_dir, names[i]+"-semu_execution_info.csv"), index=False)
#~ def fdupesAggregateKtestDirs()

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
def mutantsOfFunctions (candidateFunctionsJson, mutinfo, fdupes_dupfile, create=False):
    assert os.path.isfile(mutinfo), "mutant info file do not exist: "+mutinfo
    # load mutants info and get the list of mutants per function
    mInf = loadJson(mutinfo)
    if fdupes_dupfile is not None:
        fdupesObj = loadJson(fdupes_dupfile)
        # remove fdupes dups from mut info
        for remain_m in fdupesObj:
            for todel_m in fdupesObj[remain_m]:
                del mInf[todel_m]
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

    pathfiles = glob.glob(os.path.join(kleetestsdir, "*.sym.path"))
    #pathfiles = [v for v in pathfiles if not v.endswith(".sym.path")] #Only consider path files, not .sym.path files
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

ZESTI_DEV_TASK = 'ZESTI_DEV_TASK'
TEST_GEN_TASK = 'TEST_GEN_TASK'
SEMU_EXECUTION = 'SEMU_EXECUTION'
COMPUTE_TASK = "COMPUTE_TASK"
ANALYSE_TASK = "ANALYSE_TASK"
tasksList = [ZESTI_DEV_TASK, TEST_GEN_TASK, SEMU_EXECUTION, COMPUTE_TASK, ANALYSE_TASK]

ALIVE_ALL = 'aliveall'  # All alive mutants
ALIVE_RANDOM = 'aliverandom'  # Random set of N mutants
ALIVE_COVERED_RAND = 'alivecoveredrand'  # Random set of N mutants covered by existing tests
ALIVE_COVERED_MOST = 'alivecoveredmost'  # Top N most covered mutants (by highest number of tests)
ALIVE_COVERED_LEAST = 'alivecoveredleast'  # Top N least covered mutants (by lowest number of tests)
FIXED_MUTANT_NUMBER_STRATEGIES = [ALIVE_ALL, ALIVE_COVERED_RAND, ALIVE_COVERED_MOST, ALIVE_COVERED_LEAST]

def applyFixedMutantFiltering(groundConsideredMutant_covtests, afterFuncFilter_byfunc, fixedmutanttarget, fixedmutantnumber, continuesemu=False):
    if fixedmutantnumber is None or fixedmutanttarget == ALIVE_ALL:
        return
    if len(groundConsideredMutant_covtests) <= fixedmutantnumber:
        return
    if fixedmutanttarget == ALIVE_RANDOM:
        if continuesemu:
            error_exit("cannot continue unfinished semu with alive random filtering (mutants may not be same as others that ran before)")
        selected = random.sample(list(groundConsideredMutant_covtests), fixedmutantnumber)
    elif fixedmutanttarget == ALIVE_COVERED_RAND:
        if continuesemu:
            error_exit("cannot continue unfinished semu with alive covered random filtering (mutants may not be same as others that ran before)")
        selected = random.sample([m for m in groundConsideredMutant_covtests if len(groundConsideredMutant_covtests[m]) > 0], fixedmutantnumber)
    elif fixedmutanttarget == ALIVE_COVERED_MOST:
        selected = sorted(groundConsideredMutant_covtests.keys(), reverse=True, key=lambda x: len(groundConsideredMutant_covtests[x]))[:fixedmutantnumber]
    elif fixedmutanttarget == ALIVE_COVERED_LEAST:
        selected = sorted(groundConsideredMutant_covtests.keys(), reverse=False, key=lambda x: len(groundConsideredMutant_covtests[x]))[:fixedmutantnumber]
    else:
        error_exit ("invalit target strategy")

    selected = set(selected)

    # remove non selected
    for m in set(groundConsideredMutant_covtests) - selected:
        del groundConsideredMutant_covtests[m]
    for func in afterFuncFilter_byfunc.keys():
        afterFuncFilter_byfunc[func] = list(selected & set(afterFuncFilter_byfunc[func]))
        if len(afterFuncFilter_byfunc[func]) == 0:
            del afterFuncFilter_byfunc[func]
#~ def applyFixedMutantFiltering()

def encode_tech_conf_name(preconditionlength, mutantmaxfork, gentestfordiscardedfrom, \
                          postcheckpointcontinueproba, mutantcontinuestrategy, maxtestsgenpermutants, \
                          disablestatediffintestgen, testgenonlycriticaldiffs):
    name = "_".join([str(preconditionlength), str(mutantmaxfork), str(gentestfordiscardedfrom), \
                    str(postcheckpointcontinueproba), str(mutantcontinuestrategy), str(maxtestsgenpermutants), \
                    str(disablestatediffintestgen), 'crit' if testgenonlycriticaldiffs else 'nocrit'])
    return name
#~ def encode_tech_conf_name()

def decode_tech_conf_name(namestr):
    ret_obj = {}
    if namestr == '_pureklee_':
        ret_obj['_precondLength'] = '-'
        ret_obj['_mutantMaxFork'] = '-'
        ret_obj['_genTestForDircardedFrom'] = '-'
        ret_obj['_postCheckContProba'] = '-'
        ret_obj['_mutantContStrategy'] = '-'
        ret_obj['_maxTestsGenPerMut'] = '-'
        ret_obj['_disableStateDiffInTestgen'] = '-'
        ret_obj['_testGenOnlyCriticalDiffs'] = '-'
    else:
        vals = namestr.strip().split('_')
        if len(vals) != 8:
            error_exit("invalid name passed to decode_tech_conf_name. must have 8 fields")

        ret_obj['_precondLength'] = int(vals[0])
        ret_obj['_mutantMaxFork'] = int(vals[1])
        ret_obj['_genTestForDircardedFrom'] = int(vals[2])
        ret_obj['_postCheckContProba'] = float(vals[3])
        ret_obj['_mutantContStrategy'] = vals[4]
        ret_obj['_maxTestsGenPerMut'] = int(vals[5])

        assert vals[6].lower() in ['on', 'off'], "invalid state diff in testgen"
        ret_obj['_disableStateDiffInTestgen'] = (vals[6].lower() == 'on')

        assert vals[7].lower() in ['crit', 'nocrit'], "invalid crit/nocrit"
        ret_obj['_testGenOnlyCriticalDiffs'] = (vals[7].lower() == 'crit')

    assert len(ret_obj) == 8, "BUG"
    return ret_obj
#~ def decode_tech_conf_name()

# TODO: (1) PASS test sampling, (2) KLEE seeded test gen, (3) Merge generated tests and filter existing (keeping map), (4) Report MS and FD post test execution
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

    parser = argparse.ArgumentParser()
    parser.add_argument("outTopDir", help="topDir for output (required)")
    parser.add_argument("--executionMode", type=str, default=GenTestsToKill, choices=[FilterHardToKill, GenTestsToKill], help="The execution mode for this script (Find hard mutants of generate test to kill mutants.)")
    parser.add_argument("--exepath", type=str, default=None, help="The path to executable in project")
    parser.add_argument("--runtest", type=str, default=None, help="The test running script")
    parser.add_argument("--testlist", type=str, default=None, help="The test list file")
    parser.add_argument("--martout", type=str, default=None, help="The Mart output directory (passing this enable semu selection)")
    parser.add_argument("--matrix", type=str, default=None, help="The Strong Mutation matrix (passing this enable selecting by matrix)")
    parser.add_argument("--passfail", type=str, default=None, help="The Pass Fail matrix")
    parser.add_argument("--coverage", type=str, default=None, help="The mutant Coverage matrix")
    parser.add_argument("--candidateFunctionsJson", type=str, default=None, help="List of Functions to consider (for scalability). Json File,  Empty list of considered means consider all functions")
    parser.add_argument("--zesti_exe_dir", type=str, default=None, help="The Optional directory containing the zesti executable (named klee). if not specified, the default klee must be zesti")
    parser.add_argument("--llvm27_exe_dir", type=str, default=None, help="The Optional directory containing llvm2.7 executable. useful for zesti to compile extra headers and link")
    parser.add_argument("--llvmgcc_exe_dir", type=str, default=None, help="The Optional directory containing llvm-gcc executable. useful for zesti to compile extra headers and link")
    parser.add_argument("--semu_exe_dir", type=str, default=None, help="The Optional directory containing the SEMu executable (named klee-semu). if not specified, must be available on the PATH")
    parser.add_argument("--klee_tests_topdir", type=str, default=None, help="The Optional directory containing the extra tests separately generated by KLEE")
    parser.add_argument("--covTestThresh", type=str, default='10%', help="Minimum number(percentage) of tests covering a mutant for it to be selected for analysis")
    parser.add_argument("--skip_completed", action='append', default=[], choices=tasksList, help="Specify the tasks that have already been executed")
    parser.add_argument("--testSampleMode", type=str, default="DEV", choices=["DEV", "KLEE", "NUM", "PASS"], help="choose how to sample subset for evaluation. DEV means use Developer test, NUM, mean a percentage of all tests, PASS mean all passing tests")
    parser.add_argument("--testSamplePercent", type=float, default=10, help="Specify the percentage of test suite to use for analysis") #, (require setting testSampleMode to NUM)")
    parser.add_argument("--semusolver", type=str, default='z3', choices=['stp', 'z3'], help="Specify the solver to use for klee/semu")
    parser.add_argument("--semutimeout", type=int, default=86400, help="Specify the timeout for semu execution")
    parser.add_argument("--semumaxmemory", type=int, default=9000, help="Specify the max memory for semu execution")
    parser.add_argument("--semupreconditionlength", type=str, default='2', help="Specify space separated list of precondition length semu execution (same number as 'semumutantmaxfork')")
    parser.add_argument("--semumutantmaxfork", type=str, default='2', help="Specify space separated list of hard checkpoint for mutants (or post condition checkpoint) as PC length, in semu execution")
    parser.add_argument("--semugentestfordiscardedfrom", type=str, default='0', help="Specify space separated list of positive integer values representing the number of checkpoint to see before generating tests for non continuing mutant states, in semu execution")
    parser.add_argument("--semupostcheckpointcontinueproba", type=str, default='0.0', help="Specify space separated list of positive integer values representing the ratio of mutant states of each mutant that is allowed to pass checkpoint to the next one, in semu execution")
    parser.add_argument("--semumutantcontinuestrategy", type=str, default='mdo', help="Specify the space separated list of the trategies to use to continue mutants after watchpoint in test generation mode. Currently the strategies are min distance to output(mdt) and random(rnd).")
    parser.add_argument("--semumaxtestsgenpermutants", type=str, default='5', help="Specify the space separated list of the  maximum number of tests to generate for each mutant in test generation mode")
    parser.add_argument("--semudisablestatediffintestgen", type=str, default='off', help="Disable the inclusion of state diff in the test generation of mutant(only use mutant PC)")
    parser.add_argument("--semutestgenonlycriticaldiffs", action="store_true", help="Enable only critical diff when test generated")
    parser.add_argument("--semuloopbreaktimeout", type=float, default=120.0, help="Specify the timeout delay for ech mutant execution on a test case (estimation), to avoid inifite loop")
    parser.add_argument("--nummaxparallel", type=int, default=50, help="Specify the number of parallel executions (the mutants will be shared accross at most this number of treads for SEMU)")
    parser.add_argument("--by_function_parallelism", action="store_true", help="Enable parallelism by function. (mutants of different functions are explored in parallel)")
    parser.add_argument("--disable_pureklee", action="store_true", help="Disable doing computation for pureklee")
    parser.add_argument("--fixedmutantnumbertarget", type=str, default=ALIVE_ALL, help="Specify the how the mutants to terget are set (<mode>[:<#Mutants>]): "+str(FIXED_MUTANT_NUMBER_STRATEGIES))
    parser.add_argument("--semucontinueunfinishedtunings", action="store_true", help="enable reusing previous semu execution and computation result (if available). Useful when execution fail for some tunings and we do not want to reexecute other completed tunings")
    parser.add_argument("--disable_subsuming_mutants", action="store_true", help="Disable considering subsuming mutants in reporting")

    parser.add_argument("--semuanalysistimesnapshots_min", type=str, default=(' '.join([str(x) for x in range(1, 240, 1)])), help="Specify the space separated list of the considered time snapshots to compare the approaches in analyse")

    args = parser.parse_args()

    outDir = os.path.join(args.outTopDir, OutFolder)
    exePath = args.exepath
    runtestScript = args.runtest
    testList = args.testlist
    martOut = args.martout
    matrix = args.matrix
    passfail = args.passfail
    coverage = args.coverage
    candidateFunctionsJson = args.candidateFunctionsJson
    klee_tests_topdir = args.klee_tests_topdir
    zesti_exe_dir = args.zesti_exe_dir
    llvm27_exe_dir = args.llvm27_exe_dir
    llvmgcc_exe_dir = args.llvmgcc_exe_dir
    semu_exe_dir = args.semu_exe_dir

    # get abs path in case not
    outDir = os.path.abspath(outDir)
    executionMode = args.executionMode
    exePath = os.path.abspath(exePath) if exePath is not None else None 
    runtestScript = os.path.abspath(runtestScript) if runtestScript is not None else None 
    testList = os.path.abspath(testList) if testList is not None else None 
    martOut = os.path.abspath(martOut) if martOut is not None else None 
    matrix = os.path.abspath(matrix) if matrix is not None else None
    passfail = os.path.abspath(passfail) if passfail is not None else None
    coverage = os.path.abspath(coverage) if coverage is not None else None
    candidateFunctionsJson = os.path.abspath(candidateFunctionsJson) if candidateFunctionsJson is not None else None
    klee_tests_topdir = os.path.abspath(klee_tests_topdir) if klee_tests_topdir is not None else None
    zesti_exe_dir = os.path.abspath(zesti_exe_dir) if zesti_exe_dir is not None else None
    llvm27_exe_dir = os.path.abspath(llvm27_exe_dir) if llvm27_exe_dir is not None else None
    llvmgcc_exe_dir = os.path.abspath(llvmgcc_exe_dir) if llvmgcc_exe_dir is not None else None
    semu_exe_dir = os.path.abspath(semu_exe_dir) if semu_exe_dir is not None else None

    covTestThresh = args.covTestThresh
    testSampleMode = args.testSampleMode
    if testSampleMode in ["KLEE", "NUM", "PASS"]:
        assert klee_tests_topdir is not None, "klee_tests_topdir not give with KLEE or NUM test Smaple Mode"
    if testSampleMode == "PASS":
        assert passfail is not None, "must give passfail matrix for test Sample mode 'PASS'"
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

    semupreconditionlength_list = args.semupreconditionlength.strip().split()
    semumutantmaxfork_list = args.semumutantmaxfork.strip().split()
    semugentestfordiscardedfrom_list = args.semugentestfordiscardedfrom.strip().split()
    semupostcheckpointcontinueproba_list = args.semupostcheckpointcontinueproba.strip().split()
    semumutantcontinuestrategy_list = args.semumutantcontinuestrategy.strip().split()
    semumaxtestsgenpermutants_list = args.semumaxtestsgenpermutants.strip().split()
    semudisablestatediffintestgen_list = args.semudisablestatediffintestgen.strip().split()
    all_lists = [semupreconditionlength_list, semumutantmaxfork_list , semugentestfordiscardedfrom_list, \
                semupostcheckpointcontinueproba_list, semumutantcontinuestrategy_list, semumaxtestsgenpermutants_list, \
                semudisablestatediffintestgen_list]
    for sl in range(len(all_lists)):
        for ll in range(sl+1, len(all_lists)):
            assert len(all_lists[sl]) == len(all_lists[ll]), \
                "inconsistency between number of elements. "+str(sl)+"th list and "+\
                str(ll)+"th list. They must match (pair wise space separated)" 
    if len(set(zip(*all_lists))) < len(all_lists[0]):
        assert False, "The precondition, mutant maxfork,... tuple apperas more than once: "+ \
                str(set([x for x in zip(*all_lists) if zip(*all_lists).count(x) > 1]))

    fmnt = args.fixedmutantnumbertarget.split(':')
    if len(fmnt) == 2:
        fixedmutanttarget = fmnt[0]
        assert fmnt[1].isdigit(), "expect an integer after the column"
        fixedmutantnumber = int(fmnt[1])
        assert fixedmutantnumber >= 1, "The specified number of mutant must be >= 1"
    elif len(fmnt) == 1:
        fixedmutanttarget = fmnt[0]
        fixedmutantnumber = None
    else:
        error_exit ("invalid fixedmutantnumbertarget parameter format. expect <targetmode>[:<#Mutants>]")
    assert fixedmutanttarget in FIXED_MUTANT_NUMBER_STRATEGIES, "invalit target strategy for fixedmutantnumbertarget param"

    # Parameter tuning for Semu execution (timeout, to precondition depth)
    ## get precondition and mutantmaxfork from klee tests if specified as percentage
    semuTuningList = []
    for semupreconditionlength, semumutantmaxfork, semugentestfordiscardedfrom, \
            semupostcheckpointcontinueproba , semumutantcontinuestrategy, \
            semumaxtestsgenpermutants, semudisablestatediffintestgen in zip(*all_lists):
        if semupreconditionlength[-1] == '%' or semumutantmaxfork[-1] == '%':
            minpath_len, max_pathlen = getPathLengthsMinMaxOfKLeeTests(klee_tests_topdir, "Expecting path file for longest ktest path extraction in klee-test-dir")
        
        if semupreconditionlength[-1] == '%':
            args_semupreconditionlength = int(float(semupreconditionlength[:-1]) * max_pathlen / 100.0)
        else:
            args_semupreconditionlength = int(semupreconditionlength)

        if semumutantmaxfork[-1] == '%':
            args_semumutantmaxfork = int(float(semumutantmaxfork[:-1]) * max_pathlen / 100.0)
        else:
            args_semumutantmaxfork = int(semumutantmaxfork)
        args_semumutantmaxfork = max(0, args_semumutantmaxfork)
        print "#>> SEMU Symbex - Precondition Param:", args_semupreconditionlength, \
                ", Checkpoint Param:", args_semumutantmaxfork, \
                ", GenTestDiscardFrom:", semugentestfordiscardedfrom, \
                ", Post checkpoint Continu proba:", semupostcheckpointcontinueproba, \
                ", mutant continue strategy:", semumutantcontinuestrategy, \
                ", max testgen per mutant:", semumaxtestsgenpermutants, \
                ", disable state dif in testgen:", semudisablestatediffintestgen
            
        assert int(semugentestfordiscardedfrom) >= 0, \
                                        'invalid semugentestfordiscardedfrom'+str(semugentestfordiscardedfrom)
        assert float(semupostcheckpointcontinueproba) >= 0.0 and float(semupostcheckpointcontinueproba) <= 1.0, \
                                        'invalid semupostcheckpointcontinueproba'+str(semupostcheckpointcontinueproba)
        assert semumutantcontinuestrategy.lower() in ['mdo', 'rnd'], "invalid mutant continue strategy"
        assert int(semumaxtestsgenpermutants) >= 0, \
                                        'invalid semumaxtestsgenpermutants'+str(semumaxtestsgenpermutants)
        assert semudisablestatediffintestgen.lower() in ['off', 'on'], "invalid mutant gentest no diff enable disable value"
        semuTuningList.append({
                        'name': encode_tech_conf_name(semupreconditionlength, semumutantmaxfork, semugentestfordiscardedfrom, \
                                            semupostcheckpointcontinueproba, semumutantcontinuestrategy, \
                                            semumaxtestsgenpermutants, semudisablestatediffintestgen, \
                                            args.semutestgenonlycriticaldiffs),
                        'KLEE':{'-max-time':args.semutimeout, '-max-memory':args.semumaxmemory, 
                                '-solver-backend': args.semusolver, '-max-solver-time':300}, 
                        'SEMU':{"-semu-precondition-length":args_semupreconditionlength, 
                                "-semu-mutant-max-fork":args_semumutantmaxfork, 
                                "-semu-checknum-before-testgen-for-discarded": semugentestfordiscardedfrom,
                                "-semu-mutant-state-continue-proba": semupostcheckpointcontinueproba,
                                "-semu-loop-break-delay":args.semuloopbreaktimeout},
                        'EXTRA':{'MaxTestsPerMutant': int(semumaxtestsgenpermutants), 
                                 "-semu-testsgen-only-for-critical-diffs":args.semutestgenonlycriticaldiffs,
                                 "-semu-continue-mindist-out-heuristic":(semumutantcontinuestrategy.lower()=="mdo"),
                                 "-semu-disable-statediff-in-testgen":(semudisablestatediffintestgen.lower()=='on')}
                     })

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
        zestioutdir_targz = zestioutdir+".tar.gz"
        inBCFilePath = zestiInBCLink #if runMode == "zesti+symbex" else kleeSemuInBCLink
        test2zestidirMapFile = os.path.join(cacheDir, "test2zestidirMap.json")
        if ZESTI_DEV_TASK in toExecute:
            # Prepare outdir and copy bc
            if os.path.isdir(zestioutdir):
                shutil.rmtree(zestioutdir)
            if os.path.isfile(zestioutdir_targz):
                os.remove(zestioutdir_targz)
            os.mkdir(zestioutdir)

            unused, alltestsObj, unwrapped_testlist = getTestSamples(testList, 0, matrix)   # 0 to not sample

            if testSampleMode in ['DEV', 'NUM', 'PASS']:
                test2zestidirMap = runZestiOrSemuTC (unwrapped_testlist, alltestsObj['DEVTESTS'], exePath, runtestScript, inBCFilePath, zestioutdir, zesti_exe_dir, llvmgcc_exe_dir, llvm27_exe_dir) #, mode=runMode) #mode can also be "semuTC"
            else:
                # No need to run zesti on KLEE only mode
                test2zestidirMap = {}

            dumpJson(stripRootTest2Dir(outDir, test2zestidirMap), test2zestidirMapFile)

            # Compress zestioutdir and delete the directory, keeping only the tar.gz file
            errmsg = magma_common_fs.compressDir(zestioutdir, zestioutdir_targz, remove_in_directory=True)
            if errmsg is not None:
                error_exit(errmsg)
        else:
            print "## Loading zesti test mapping from Cache"
            #assert os.path.isdir(zestioutdir), "Error: zestioutdir absent when ZESTI_DEV mode skipped"
            assert os.path.isfile(zestioutdir_targz), "Error: zestioutdir.tar.gz absent when ZESTI_DEV mode skipped"
            test2zestidirMap = loadJson(test2zestidirMapFile)
            test2zestidirMap = prependRootTest2Dir(outDir, test2zestidirMap)

    # TODO: TEST GEN part here. if klee_tests_topdir is not None, means use the tests from klee to increase baseline and dev test to evaluate aproaches
    # prepare seeds and extract sym-args. Then store it in the cache
    semuworkdir = os.path.join(cacheDir, "SemuWorkDir")
    semuworkdir_targz = semuworkdir+".tar.gz"
    test2semudirMapFile = os.path.join(cacheDir, "test2semudirMap.json")
    if TEST_GEN_TASK in toExecute:
        print "# Doing TEST_GEN_TASK ..."
        #assert os.path.isdir(zestioutdir), "Error: "+zestioutdir+" not existing. Please make sure to collect Dev tests with Zesti"
        assert os.path.isfile(zestioutdir_targz), "Error: "+zestioutdir_targz+" not existing. Please make sure to collect Dev tests with Zesti"

        # Decompress zestioutdir_targz
        if os.path.isdir(zestioutdir):
            shutil.rmtree(zestioutdir)
        errmsg = magma_common_fs.decompressDir(zestioutdir_targz, zestioutdir)
        if errmsg is not None:
            error_exit(errmsg)
        
        if os.path.isdir(semuworkdir):
            shutil.rmtree(semuworkdir)
        os.mkdir(semuworkdir)

        zest_sym_args_param = None
        zestKTContains = None
        klee_sym_args_param = None
        kleeKTContains = None

        if testSampleMode in ['DEV', 'NUM', 'PASS']:
            zestKtests = []
            for tc in test2zestidirMap.keys():
                tcdir = test2zestidirMap[tc]
                listKtestFiles = glob.glob(os.path.join(tcdir, "*.ktest"))
                assert len(listKtestFiles) == 1, "Error: more than 1 or no ktest from Zesti for tests: "+tc+", zestiout: "+tcdir
                for ktestfile in listKtestFiles:
                    zestKtests.append(ktestfile)
            # refactor the ktest fom zesti and put in semu workdir, together with the sym
            zest_sym_args_param, zestKTContains = getSymArgsFromZestiKtests (zestKtests, test2zestidirMap)

        if testSampleMode in ['KLEE', 'NUM', 'PASS']:
            unused, alltestsObj, unwrapped_testlist = getTestSamples(testList, 0, matrix)   # 0 to not sample
            klee_sym_args_param, kleeKTContains = loadAndGetSymArgsFromKleeKTests (alltestsObj['GENTESTS'], klee_tests_topdir)
            
        sym_args_param, test2semudirMap = mergeZestiAndKleeKTests (semuworkdir, zestKTContains, zest_sym_args_param, kleeKTContains, klee_sym_args_param)
        dumpJson([testSampleMode, sym_args_param, stripRootTest2Dir(outDir, test2semudirMap)], test2semudirMapFile)

        # remove temporary decompressed zestioutdir 
        shutil.rmtree(zestioutdir)

        # Compress semuworkdir and delete the directory, keeping only the tar.gz file
        errmsg = magma_common_fs.compressDir(semuworkdir, semuworkdir_targz, remove_in_directory=True)
        if errmsg is not None:
            error_exit(errmsg)

    else:
        print "## Loading parametrized tests mapping from cache"
        #assert os.path.isdir(semuworkdir), "Error: semuworkdir absent when TEST-GEN mode skipped"
        assert os.path.isfile(semuworkdir_targz), "Error: semuworkdir.tar.gz absent when TEST-GEN mode skipped"
        tmpSamplMode, sym_args_param, test2semudirMap = loadJson(test2semudirMapFile)
        assert tmpSamplMode == testSampleMode, "Given test Sample Mode ("+testSampleMode+") is different from caches ("+tmpSamplMode+"); should not skip TEST_GET_TASK!"
        test2semudirMap = prependRootTest2Dir(outDir, test2semudirMap)

    # Get all test samples before starting experiment
    ## XXX: Fix this when supporting other testSampleModes
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
        elif testSampleMode == 'PASS':
            cand_pass = set()
            # get passing
            with open(passfail) as f:
                for line_ in f:
                    tc_, v_ = line_.strip().split()
                    if v_ == '0':
                        cand_pass.add(tc_)
            # get passing and considered
            cand_pass &= set(alltestsObj['DEVTESTS']) | set(alltestsObj['GENTESTS'])

            sampl_size = int(max(1, testSamplePercent * len(cand_pass) / 100))
            testSamples = {'PASS_'+str(testSamplePercent): random.sample(cand_pass, sampl_size)}
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
    testgen_mode_initial_numtests = None
    testgen_mode_initial_muts = None
    testgen_mode_initial_killmuts = None
    if matrix is not None:
        groundConsideredMutant_covtests = matrixHardness.getCoveredMutants(coverage, os.path.join(martOut, mutantInfoFile), os.path.join(martOut, fdupesDuplicatesFile), testTresh_str = covTestThresh)

        if executionMode == FilterHardToKill:
            ground_KilledMutants = set(matrixHardness.getKillableMutants(matrix)) 
            toremoveMuts = set(groundConsideredMutant_covtests) - ground_KilledMutants
        else:
            assert len(testSamples.keys()) == 1, "TestSamples must have only one key (correspond to one experiment - one set of mutants and tests)"
            ground_KilledMutants = set(matrixHardness.getKillableMutants(matrix, testset=set(testSamples[testSamples.keys()[0]]))) 
            toremoveMuts = ground_KilledMutants

            testgen_mode_initial_numtests = len(testSamples[testSamples.keys()[0]])
            testgen_mode_initial_muts = list(groundConsideredMutant_covtests)
            testgen_mode_initial_killmuts = list(toremoveMuts)
            
        # keep only covered by treshold at least, and killed
        for mid in toremoveMuts:
            if mid in groundConsideredMutant_covtests:
                del groundConsideredMutant_covtests[mid]
        print "# Number of Mutants after coverage filtering:", len(groundConsideredMutant_covtests)
        
        # consider the specified functions
        afterFuncFilter_byfunc = mutantsOfFunctions (candidateFunctionsJson, os.path.join(martOut, mutantInfoFile), os.path.join(martOut, fdupesDuplicatesFile), create=False)
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

        # Apply filtering of mutants if specified (Which mutants to focus on)
        applyFixedMutantFiltering(groundConsideredMutant_covtests, afterFuncFilter_byfunc, fixedmutanttarget, fixedmutantnumber, args.semucontinueunfinishedtunings)
        print "# Number of Mutants after filtering specified:", len(groundConsideredMutant_covtests)
        if len(groundConsideredMutant_covtests) < 1:
            error_exit ("error(done): No mutant Left to analyze")

        paraAssign = assignSemuJobs(afterFuncFilter_byfunc, args.nummaxparallel if args.by_function_parallelism else 1)
        for pj in paraAssign:
            pjs = set(pj)
            list_groundConsideredMutant_covtests.append({mid: groundConsideredMutant_covtests[mid] for mid in groundConsideredMutant_covtests if mid in pjs})
        
        print "# Number of Mutants Considered:", sum([len(x) for x in list_groundConsideredMutant_covtests]), ". With", len(paraAssign), "Semu executions in parallel"

        minMutNum = 10 if executionMode == FilterHardToKill else 1

        assert sum([len(x) for x in list_groundConsideredMutant_covtests]) >=  minMutNum, " ".join(["We have only", str(sum([len(x) for x in list_groundConsideredMutant_covtests])), "mutants fullfiling testcover treshhold",str(covTestThresh),"(Expected >= "+str(minMutNum)+")"])
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

        semuSeedsDir = semuOutputsTop+".seeds.tmp"

        '''
            function to run in parallel for different configs (pre, post conditions pairs and pure KLEE)
        '''
        def configParallel(semuTuning):
            # Make temporary outdir for test sample size
            outFolder = "out_testsize_"+str(ts_size)+'.'+semuTuning['name']
            this_Out = os.path.join(outDir, outFolder)
            thisOut_list = [this_Out+"-"+str(i) for i in range(len(list_candidateMutantsFiles))]
            mergeSemuThisOut = this_Out+".semumerged"

            se_output = os.path.join(semuOutputsTop, outFolder, "Thread")
            semuoutputs = [se_output+"-"+str(i) for i in range(len(list_candidateMutantsFiles))]
            # No merge on Test Gen mode
            mergeSemuThreadsDir = se_output+".semumerged" if executionMode == FilterHardToKill else None

            # XXX Help not to execute finished if some failed
            semuWasNotExecuted = True
            # Test gen mode
            if executionMode != FilterHardToKill:
                mfi_mutants_list = os.path.join(this_Out, "mfirun_mutants_list.txt")
                mfi_ktests_dir_top = os.path.join(this_Out, "mfirun_ktests_dir")
                mfi_ktests_dir = os.path.join(mfi_ktests_dir_top, KLEE_TESTGEN_SCRIPT_TESTS+"-out", "klee-out-0")
                mfi_execution_output = os.path.join(this_Out, "mfirun_output")
                if args.semucontinueunfinishedtunings:
                    # check whether it was finished
                    if os.path.isfile(mfi_mutants_list) and os.path.isdir(mfi_ktests_dir):
                        for semuoutput in semuoutputs:
                            if not os.path.isdir(semuoutput):
                                semuWasNotExecuted = False
                            else:
                                semuWasNotExecuted = True
                                break
                if semuWasNotExecuted:
                    if os.path.isfile(mfi_mutants_list):
                        os.remove(mfi_mutants_list)
                    if os.path.isdir(mfi_ktests_dir_top):
                        shutil.rmtree(mfi_ktests_dir_top)
                    if os.path.isdir(mfi_execution_output):
                        shutil.rmtree(mfi_execution_output)

            # Execute SEMU
            if SEMU_EXECUTION in toExecute:
                if semuWasNotExecuted: 
                    if martOut is not None:
                        ret = executeSemu (semuoutputs, semuSeedsDir, kleeSemuInBCLink, list_candidateMutantsFiles, sym_args_param, semu_exe_dir, semuTuning, mergeThreadsDir=mergeSemuThreadsDir, exemode=executionMode) 
                        if ret is not None:
                            print '\n'+ret+'\n'
                            return None
                else:
                    print '['+time.strftime("%c")+']', "# Semu was already executed (just reading) for ts_size:", ts_size, "; name: "+semuTuning['name'], "..."

            if executionMode == FilterHardToKill:
                if len(thisOut_list) == 1 and os.path.isdir(mergeSemuThreadsDir): #only have one thread, only process that
                    zips = [(groundConsideredMutant_covtests, mergeSemuThisOut, mergeSemuThreadsDir)]
                else:
                    zips = zip(list_groundConsideredMutant_covtests+[groundConsideredMutant_covtests], thisOut_list+[mergeSemuThisOut], semuoutputs+[mergeSemuThreadsDir])
                for groundConsMut_cov, thisOutSe, semuoutput in zips:
                    # process with each approach
                    if COMPUTE_TASK in toExecute: 
                        print "# Procesing for test size", ts_size, "; name: "+semuTuning['name'], "..."

                        if martOut is not None or matrix is not None:
                            if os.path.isdir(thisOutSe):
                                shutil.rmtree(thisOutSe)
                            os.mkdir(thisOutSe)

                        # process for matrix
                        if matrix is not None:
                            processMatrix (matrix, alltests, 'groundtruth', groundConsMut_cov, thisOutSe) 
                            processMatrix (matrix, testSamples[ts_size], 'classic', groundConsMut_cov, thisOutSe) 

                        # process for SEMU
                        if martOut is not None:
                            processSemu (semuoutput, "semu", thisOutSe)

                    # Analysing for each test Sample 
                    if ANALYSE_TASK in toExecute:
                        print "# Analysing for test size", ts_size,  "; name: "+semuTuning['name'], "..."

                        # Make final Analysis and plot
                        if martOut is not None and matrix is not None:
                            analysis_plot(thisOutSe, None) #groundConsideredMutant_covtests.keys()) # None to not plot random
            else:
                # Test gen mode
                if COMPUTE_TASK in toExecute: 
                    atMergeStage = False
                    for semuoutput in semuoutputs:
                        if not os.path.isdir(semuoutput):
                            atMergeStage = True
                        else:
                            assert not atMergeStage, "some semuoutputs are deleted but not "+semuoutput+". Must delete and rerun Semu Execution!"

                    if atMergeStage:
                        print '['+time.strftime("%c")+']', "# -- Already ready for merge in Test Gen Compute. Do nothing (for "+this_Out+')'
                    else:
                        print '['+time.strftime("%c")+']', "# Compute task Procesing for test size", ts_size,  "; name: "+semuTuning['name'], "..."
                        if not os.path.isdir(this_Out):
                            os.mkdir(this_Out)
                        if os.path.isdir(mfi_execution_output):
                            print ""
                            choice = raw_input("Are you sure you want to clean existing mfi_execution_output? [y/N] ")
                            if choice.lower() in ['y', 'yes']:
                                shutil.rmtree(mfi_execution_output)
                                print "## Previous outdir removed"
                            else:
                                print "\n#> Not deleting previous out, use skip compute task if just want to analyse", "; name: "+semuTuning['name'] 
                                exit(1)

                        if not os.path.isdir(mfi_execution_output):
                            with open(mfi_mutants_list, "w") as fp:
                                for mid in groundConsideredMutant_covtests:
                                    fp.write(str(mid)+'\n')
                            #TODO consider removing seed tests (already executed). The merge of thread should be done in semuexec. (Not necessary, because only live mutant are executed)
                            fdupeGeneratedTest (mfi_ktests_dir_top, mfi_ktests_dir, semuoutputs, semuSeedsDir) 
                        # remove semuoutputs dirs
                        for semuoutput in semuoutputs:
                            if os.path.isdir(semuoutput):
                                shutil.rmtree(semuoutput)

                return (mfi_mutants_list, mfi_ktests_dir_top, mfi_ktests_dir, mfi_execution_output, this_Out, semuTuning['name'])
        #~ def configParallel():

        # Prepare the seeds to use
        if SEMU_EXECUTION in toExecute:
            if os.path.isdir(semuSeedsDir):
                shutil.rmtree(semuSeedsDir)
            os.mkdir(semuSeedsDir)

            # Decompress semuworkdir_targz
            errmsg = magma_common_fs.decompressDir(semuworkdir_targz, semuworkdir)
            if errmsg is not None:
                error_exit(errmsg)
            
            # copy needed tests
            for tc in testSamples[ts_size]:
                shutil.copy2(test2semudirMap[tc], semuSeedsDir)

            # delete the decompressed temporary semuworkdir 
            shutil.rmtree(semuworkdir)

            # Fdupes the seedDir.
            if os.system(" ".join(["fdupes -r -d -N", semuSeedsDir, '> /dev/null'])) != 0:
                error_exit ("fdupes failed on semuSeedDir")

            # In case of clean run, clean semuTuningList semuOutputsTop 
            if not args.semucontinueunfinishedtunings:
                for d in os.listdir(semuOutputsTop):
                    d_tmp = os.path.join(semuOutputsTop, d)
                    if os.path.isdir(d_tmp):
                        shutil.rmtree(d_tmp)

        # In the case of tests generation, add pure klee
        if executionMode == GenTestsToKill and not args.disable_pureklee:
            assert type(semuTuningList[-1]) == dict
            purekleetune = semuTuningList[-1].copy()
            purekleetune['name'] = "_pureklee_"
            del purekleetune['SEMU']
            semuTuningList.append(purekleetune)

        # Actual Semu execution and compute
        nparallel_for_tunings = min( \
                            len(semuTuningList), \
                            max(1, args.nummaxparallel / len(list_candidateMutantsFiles)) \
                            )
        if nparallel_for_tunings > 1:
            conf_threadpool = ThreadPool(nparallel_for_tunings)
            ctp_return = conf_threadpool.map(configParallel, semuTuningList)
            conf_threadpool.terminate()
            conf_threadpool.close()
            conf_threadpool.join()
        else:
            ctp_return = map(configParallel, semuTuningList)

        if None in ctp_return:
            error_exit("error: There were some failures (see above)!")

        # Remove used seeds
        if SEMU_EXECUTION in toExecute:
            shutil.rmtree(semuSeedsDir)

        if executionMode == GenTestsToKill:
            agg_Out = os.path.join(outDir, "TestGenFinalAggregated"+str(ts_size))
            mfi_mutants_list = os.path.join(agg_Out, "mfirun_mutants_list.txt")
            killed_non_mfi_mutants_list = os.path.join(agg_Out, "killed_non_mfirun_mutants_list.txt")
            mfi_ktests_dir_top = os.path.join(agg_Out, "mfirun_ktests_dir")
            mfi_ktests_dir = os.path.join(mfi_ktests_dir_top, KLEE_TESTGEN_SCRIPT_TESTS+"-out", "klee-out-0")
            mfi_execution_output = os.path.join(agg_Out, "mfirun_output")
            killed_non_mfi_execution_output = os.path.join(agg_Out, "killed_non_mfirun_output")

            ## TODO: Remove this when all migrated
            if "MFI_SEMU_SUBSUMING_MIGRATE_TMP" in os.environ and os.environ["MFI_SEMU_SUBSUMING_MIGRATE_TMP"] == "on":
                with open(killed_non_mfi_mutants_list, "w") as f:
                    for m in ground_KilledMutants:
                        f.write(str(m)+"\n")
                exit(0)
            ######~~~~~~~~~~~

            if COMPUTE_TASK in toExecute: 
                ktdirs = []
                d_names = []
                present_thisOut = []
                for part_mfi_mutants_list, part_mfi_ktests_dir_top, part_mfi_ktests_dir, part_mfi_execution_output, part_this_Out, part_d_name in ctp_return:
                    ktdirs.append(part_mfi_ktests_dir)
                    d_names.append(part_d_name)
                    if os.path.isdir(part_this_Out):
                        present_thisOut.append(part_this_Out)

                if len(present_thisOut) == 0:
                    print "# -- Already merged in Test Gen Compute. Do nothing (merging)" 
                elif len(present_thisOut) < len(ctp_return):
                    assert os.path.isdir(agg_Out), "Must have started deleting merge thread after merging tests here and stopped"
                    print "# -- problem occured when deleting merge threads, after agg merge here. just verify and delete these manually: "+str(present_thisOut)
                else:
                    if os.path.isdir(agg_Out):
                        shutil.rmtree(agg_Out)
                    os.mkdir(agg_Out)

                    # All mutant list must be same
                    part_mfi_mutants_list, part_mfi_ktests_dir_top, part_mfi_ktests_dir, part_mfi_execution_output, part_this_Out, part_d_name = ctp_return[0]
                    shutil.copy2(part_mfi_mutants_list, mfi_mutants_list)

                    with open(killed_non_mfi_mutants_list, "w") as f:
                        for m in ground_KilledMutants:
                            f.write(str(m)+"\n")

                    # Merge tests.  
                    fdupesAggregateKtestDirs (mfi_ktests_dir_top, mfi_ktests_dir, ktdirs, d_names)

                    # delete the separate data stuffs
                    for part_mfi_mutants_list, part_mfi_ktests_dir_top, part_mfi_ktests_dir, part_mfi_execution_output, part_this_Out, part_d_name in ctp_return:
                        shutil.rmtree(part_this_Out) # This implicitely remove part_mfi_mutants_list, part_mfi_ktests_dir_top, ...


            # TODO: considere separate executions and Fault detection in Analysis
            if ANALYSE_TASK in toExecute:
                time_snapshots_minutes_list = [int(math.ceil(float(v))) for v in args.semuanalysistimesnapshots_min.strip().split()]
                max_time_minutes = int(math.ceil(float(args.semutimeout)/60.0))
                time_snapshots_minutes_list.append(max_time_minutes) # semutimeout is in seconds
                time_snapshots_minutes_list = sorted(list(set(time_snapshots_minutes_list)))
                for v in time_snapshots_minutes_list:
                    assert v >= 0, "invalid value for time snapshot: "+str(v)

                # discard higher than specified max
                maxtime_ind = 0
                for stv in time_snapshots_minutes_list:
                    if stv >= max_time_minutes:
                        break
                    maxtime_ind += 1
                time_snapshots_minutes_list = time_snapshots_minutes_list[:maxtime_ind+1]

                initial_dats_json = os.path.join(agg_Out, "Initial-dat.json")
                outcsvfile = os.path.join(agg_Out, "Results.csv")
                funcscsvfile = os.path.join(agg_Out, "Results-byfunctions.csv")
                outjsonfile_common = os.path.join(agg_Out, "Techs-relation.json")
                if os.path.isfile(outcsvfile):
                    os.remove(outcsvfile)
                for outjsonfile in glob.glob(outjsonfile_common+'*'):
                    if os.path.isfile(outjsonfile):
                        os.remove(outjsonfile)

                mutant_info_obj = loadJson(os.path.join(martOut, mutantInfoFile))
                considered_mutants_by_functions = {}
                for m in groundConsideredMutant_covtests:
                    funcname = mutant_info_obj[str(m)]['FuncName']
                    if funcname not in considered_mutants_by_functions:
                        considered_mutants_by_functions[funcname] = []
                    considered_mutants_by_functions[funcname].append(m) 

                nGenTests_ = len(glob.glob(mfi_ktests_dir+"/*.ktest"))
                if nGenTests_ > 0 and not os.path.isdir(mfi_execution_output):
                    print "\n--------------------------------------------------"
                    print ">> There are a total of", nGenTests_, "tests to run"
                    print ">> You now need to execute the generated tests using MFI (externale mode)"
                    print ">> For MFI, use the following:"
                    print ">>   mfi_mutants_list:", mfi_mutants_list
                    print ">>   mfi_ktests_dir_top:", mfi_ktests_dir_top
                    print ">>   mfi_execution_output:", mfi_execution_output
                    print "...................................."
                    print "echo 5 > mfi_executing.state"
                    print "MFI_OVERRIDE_OUTPUT="+mfi_execution_output,
                    print "MFI_OVERRIDE_MUTANTSLIST="+mfi_mutants_list,
                    print "MFI_OVERRIDE_GENTESTSDIR="+mfi_ktests_dir_top, 
                    print "<Run MFI command>"
                    print "--------------------------------------------------"
                    print "@ Rexecute this when done (skipping every task except analyse task)."
                else:
                    nMutants = len(groundConsideredMutant_covtests)
                    sm_file = os.path.join(mfi_execution_output, "data", "matrices", "SM.dat")
                    mcov_file = os.path.join(mfi_execution_output, "data", "matrices", "MCOV.dat")
                    pf_file = os.path.join(mfi_execution_output, "data", "matrices", "ktestPassFail.txt")
                    gen_on_killed_sm = os.path.join(killed_non_mfi_execution_output, "data", "matrices", "SM.dat")

                    # get subsuming mutants
                    subsuming_mutants_clusters = None
                    if not args.disable_subsuming_mutants:
                        subsuming_mutants_clusters = get_subsuming_mutants(matrix, sm_file, gen_on_killed_sm)

                    out_df_parts = []
                    funcs_out_df_parts = []
                    for time_snapshot_minute in time_snapshots_minutes_list:
                        outjsonfile = outjsonfile_common+"-"+str(time_snapshot_minute)+'min.json'
                        outobj_ = {}
                        killedMutsPerTuning = {}
                        killedSubsumingMutsPerTuning = {}
                        time_snap_dfs_dats = []
                        funcs_time_snap_dfs_dats = []
                        for semuTuning in semuTuningList:
                            nameprefix = semuTuning['name']
                            cons_kt_df = pd.read_csv(os.path.join(mfi_ktests_dir, nameprefix+"-tests_by_ellapsedtime.csv"))
                            cons_kt_df = cons_kt_df[cons_kt_df['ellapsedTime(s)'] <= time_snapshot_minute * 60.0]
                            testsOfThis = cons_kt_df['ktest']
                            test2mutsDS = loadJson(os.path.join(mfi_ktests_dir, nameprefix+"-mutant_ktests_mapping.json"))
                            #test2mutsDS = {kt: test2mutsDS[kt] for kt in test2mutsDS if kt in testsOfThis}
                            mutants2ktests = {}
                            targeted_mutants = set()
                            for kt in test2mutsDS:
                                for mutant_,ellapsedtime in test2mutsDS[kt]:
                                    if float(ellapsedtime) <= time_snapshot_minute * 60.0:
                                        targeted_mutants.add(mutant_)
                                        if mutant_ not in mutants2ktests:
                                            mutants2ktests[mutant_] = []
                                        mutants2ktests[mutant_].append(kt)
                            assert set([kt for m in mutants2ktests for kt in mutants2ktests[m]]) == set(testsOfThis), \
                                    "Error (BUG?): Mismatch betweem values in tests_by_ellapsedtime and mutant_ktest_map" + \
                                        " for "+nameprefix

                            if nameprefix != '_pureklee_':
                                assert len(targeted_mutants - set(groundConsideredMutant_covtests)) == 0, "more mutants were used to gen tests: "+str(len(targeted_mutants - set(groundConsideredMutant_covtests)))

                            # TODO loop over here for whole proj and by each function
                            for filtering_func in [None] + list(considered_mutants_by_functions):
                                # Compute testsOfThis
                                if filtering_func is None:
                                    filt_mutants = set(groundConsideredMutant_covtests)
                                    filt_nMutants = nMutants
                                    filt_targeted_mutants = targeted_mutants
                                    filt_testsOfThis = testsOfThis
                                else:
                                    assert filtering_func in considered_mutants_by_functions, "BUG: invalid func:"+filtering_func
                                    filt_mutants = set(considered_mutants_by_functions[filtering_func])
                                    filt_nMutants = len(filt_mutants)
                                    filt_targeted_mutants = list(set(targeted_mutants) & set(considered_mutants_by_functions[filtering_func]))
                                    filt_testsOfThis = []
                                    for m in filt_targeted_mutants:
                                        filt_testsOfThis += mutants2ktests[m]
                                    filt_testsOfThis = list(set(filt_testsOfThis) & set(testsOfThis))
                                # subsuming
                                if subsuming_mutants_clusters is not None:
                                    subsuming_filt_mutants_clust = subs_clusters_of(subsuming_mutants_clusters, filt_mutants)
                                    subsuming_filt_nMutants_clust = len(subsuming_filt_mutants_clust)
                                    subsuming_filt_targeted_mutants_clust = subs_clusters_of(subsuming_mutants_clusters, filt_targeted_mutants)

                                filt_testsOfThis = set([os.path.join(KLEE_TESTGEN_SCRIPT_TESTS+"-out", "klee-out-0", kt) for kt in filt_testsOfThis])
                                testsKillingOfThis = []
                                if len(filt_testsOfThis) > 0:
                                    newKilled = set(filt_mutants) & set(matrixHardness.getKillableMutants(sm_file, filt_testsOfThis, testkillinglist=testsKillingOfThis))
                                    newCovered = set(filt_mutants) & set(matrixHardness.getListCoveredMutants(mcov_file, filt_testsOfThis))
                                    nnewFailing = len(set(matrixHardness.getFaultyTests(pf_file, filt_testsOfThis)))
                                else:
                                    newKilled = []
                                    newCovered = []
                                    nnewFailing = 0
                                nnewKilled = len(newKilled)
                                nnewCovered = len(newCovered)

                                # subsuming
                                if subsuming_mutants_clusters is not None:
                                    subsuming_newKilled_clust = subs_clusters_of(subsuming_mutants_clusters, newKilled)
                                    subsuming_newCovered_clust = subs_clusters_of(subsuming_mutants_clusters, newCovered)
                                    nsubsuming_newKilled_clust = len(subsuming_newKilled_clust)
                                    nsubsuming_newCovered_clust = len(subsuming_newCovered_clust)

                                semu_info_df = pd.read_csv(os.path.join(mfi_ktests_dir, nameprefix+"-semu_execution_info.csv"))
                                semu_info_stateCmpTime = '-'
                                semu_info_numMutstatesForkedFromOrig = '-'
                                semu_info_numMutstatesEqWithOrigAtMutPoint = '-'
                                for index, row in semu_info_df.iterrows():
                                    if float(row["ellapsedTime(s)"]) <= time_snapshot_minute * 60.0 :
                                        semu_info_stateCmpTime = float(row["stateCompareTime(s)"])
                                        semu_info_numMutstatesForkedFromOrig = int(row["#MutStatesForkedFromOriginal"])
                                        semu_info_numMutstatesEqWithOrigAtMutPoint = int(row["#MutStatesEqWithOrigAtMutPoint"])
                                    else:
                                        break
                                if time_snapshot_minute == max_time_minutes:
                                    seed_dup_kts_num = loadJson(os.path.join(mfi_ktests_dir, nameprefix+"-seed_dup_ktests.json"))["Number of seed duplicates removed"]
                                else:
                                    seed_dup_kts_num = '-'

                                tmp_data = {
                                                "TimeSnapshot(min)": time_snapshot_minute,
                                                "Tech-Config": nameprefix,
                                                "#Mutants": filt_nMutants, 
                                                "#Targeted": len(filt_targeted_mutants), 
                                                "#Covered": nnewCovered, 
                                                "#Killed": nnewKilled, 
                                                "#GenTests":len(filt_testsOfThis), 
                                                "#GenTestsKilling":len(testsKillingOfThis), 
                                                "#FailingTests":nnewFailing, 
                                                "MS-INC":(nnewKilled * 100.0 / filt_nMutants), 
                                                "#AggregatedTestGen": nGenTests_,
                                                "#SeedDuplicatedGenTests": seed_dup_kts_num,
                                                "StateComparisonTime(s)": semu_info_stateCmpTime,
                                                "#MutStatesForkedFromOriginal": semu_info_numMutstatesForkedFromOrig,
                                                "#MutStatesEqWithOrigAtMutPoint": semu_info_numMutstatesEqWithOrigAtMutPoint,
                                            }

                                # Subsuming
                                if subsuming_mutants_clusters is not None:
                                    tmp_data["#SubsMutantsClusters"] = subsuming_filt_nMutants_clust
                                    tmp_data["#SubsTargetedClusters"] = len(subsuming_filt_targeted_mutants_clust)
                                    tmp_data["#SubsCoveredClusters"] = nsubsuming_newCovered_clust 
                                    tmp_data["#SubsKilledClusters"] = nsubsuming_newKilled_clust
                                    tmp_data["MS_SUBSUMING-INC"] = (nsubsuming_newKilled_clust * 100.0 / subsuming_filt_nMutants_clust)

                                if filtering_func is None:
                                    time_snap_dfs_dats.append(tmp_data)
                                    killedMutsPerTuning[nameprefix] = set(newKilled)
                                    if subsuming_mutants_clusters is not None:
                                        killedSubsumingMutsPerTuning[nameprefix] = set(subsuming_newKilled_clust)
                                else:
                                    tmp_data["FunctionName"] = filtering_func
                                    funcs_time_snap_dfs_dats.append(tmp_data) 

                        for _snap_dfs_dats in (time_snap_dfs_dats, funcs_time_snap_dfs_dats):
                            _snap_agg_ntest = sum([v["#GenTests"] for v in _snap_dfs_dats])
                            for v_ind in range(len(_snap_dfs_dats)):
                                _snap_dfs_dats[v_ind]["#AggregatedTestGen"] = _snap_agg_ntest
                        out_df_parts += time_snap_dfs_dats
                        funcs_out_df_parts += funcs_time_snap_dfs_dats

                        extra_res = {}
                        subsuming_extra_res = {}
                        #venn_killedMutsInCommon, _ = magma_stats_algo.getCommonSetsSizes_venn (killedMutsPerTuning, setsize_from=1, setsize_to=len(killedMutsPerTuning), name_delim='&')
                        venn_killedMutsInCommon, _ = magma_stats_algo.getCommonSetsSizes_venn (killedMutsPerTuning, setsize_from=2, setsize_to=2, \
                                                                                                name_delim='&', not_common=extra_res)
                        if subsuming_mutants_clusters is not None:
                            subsuming_venn_killedMutsInCommon, _ = magma_stats_algo.getCommonSetsSizes_venn (\
                                                                                                killedSubsumingMutsPerTuning, setsize_from=2, setsize_to=2, \
                                                                                                name_delim='&', not_common=subsuming_extra_res)

                        #extra_keys_outobj = []
                        assert "MAX_SEMU_TIMEOUT" not in outobj_
                        outobj_["MAX_SEMU_TIMEOUT"] = args.semutimeout
                        #extra_keys_outobj.append('MAX_SEMU_TIMEOUT')
                        overlap_nonoverlap = [("MUTANTS", venn_killedMutsInCommon, extra_res)]
                        # subsuming
                        if subsuming_mutants_clusters is not None:
                            overlap_nonoverlap.append(("SUBSUMING_CLUSTERS", subsuming_venn_killedMutsInCommon, subsuming_extra_res))
                        for mut_mode, _venn, _extra in overlap_nonoverlap:
                            outobj_[mut_mode] = {}
                            assert "OVERLAP_VENN" not in outobj_[mut_mode]
                            outobj_[mut_mode]["OVERLAP_VENN"] = _venn
                            #extra_keys_outobj.append('OVERLAP_VENN')
                            assert "NON_OVERLAP_VENN" not in outobj_[mut_mode]
                            outobj_[mut_mode]["NON_OVERLAP_VENN"] = _extra
                            #extra_keys_outobj.append('NON_OVERLAP_VENN')

                        dumpJson(outobj_, outjsonfile)
                        print "Kill Mutant TestGen Analyse Result:"
                        #for k in sorted(outobj_.keys(), reverse=True, key=lambda x:x in extra_keys_outobj):
                        #    print ">>> '"+k+"':", outobj_[k]

                    initial_json_obj = { \
                                "Initial#Mutants": len(testgen_mode_initial_muts), \
                                "Initial#KilledMutants": len(testgen_mode_initial_killmuts), \
                                "Inintial#Tests": testgen_mode_initial_numtests, \
                                "Initial-MS": ((len(testgen_mode_initial_killmuts)) * 100.0 / len(testgen_mode_initial_muts)), \
                                "TestSampleMode": args.testSampleMode, \
                                "MaxTestGen-Time(min)": max_time_minutes, \
                                "By-Functions": {} \
                                } 
                    # subsuming
                    if subsuming_mutants_clusters is not None:
                        initial_json_obj["Initial#SubsumingMutants"] = len(subs_clusters_of(subsuming_mutants_clusters, testgen_mode_initial_muts))
                        initial_json_obj["Initial#SubsumingKilledMutants"] = len(subs_clusters_of(subsuming_mutants_clusters, testgen_mode_initial_killmuts))
                        initial_json_obj["Initial-MS_Subsuming"] = \
                                                            initial_json_obj["Initial#SubsumingKilledMutants"] * 100.0 / initial_json_obj["Initial#SubsumingMutants"]

                    for mlist_list, mlist_key in [(testgen_mode_initial_muts, "Initial#Mutants"), \
                                                (testgen_mode_initial_killmuts, "Initial#KilledMutants")]:
                        for m in mlist_list:
                            funcname = mutant_info_obj[str(m)]['FuncName']
                            if funcname in considered_mutants_by_functions:
                                if funcname not in initial_json_obj["By-Functions"]:
                                    initial_json_obj["By-Functions"][funcname] = {}
                                    initial_json_obj["By-Functions"][funcname]["Initial#Mutants"] = 0 
                                    initial_json_obj["By-Functions"][funcname]["Initial#KilledMutants"] = 0 
                                initial_json_obj["By-Functions"][funcname][mlist_key] += 1
                    for funcname in initial_json_obj["By-Functions"]:
                        initial_json_obj["By-Functions"][funcname]["Initial-MS"] = 100.0 \
                                                * initial_json_obj["By-Functions"][funcname]["Initial#KilledMutants"] \
                                                / initial_json_obj["By-Functions"][funcname]["Initial#Mutants"] 
                    dumpJson(initial_json_obj, initial_dats_json)

                    res_df = pd.DataFrame(out_df_parts)
                    funcs_res_df = pd.DataFrame(funcs_out_df_parts)
                    ordered_df_cols = ["TimeSnapshot(min)", "Tech-Config", 
                                        "#SubsMutantsClusters", "#SubsTargetedClusters", "#SubsCoveredClusters", "#SubsKilledClusters", "MS_SUBSUMING-INC", \
                                        "#Mutants", "#Targeted", "#Covered", "#Killed", "#GenTests", "#GenTestsKilling", "#FailingTests", "MS-INC", \
                                        "#AggregatedTestGen", "#SeedDuplicatedGenTests", \
                                        "StateComparisonTime(s)", "#MutStatesForkedFromOriginal", "#MutStatesEqWithOrigAtMutPoint"]
                    funcs_ordered_df_cols = ["FunctionName"]+ordered_df_cols
                    if set(res_df) != set(ordered_df_cols):
                        error_exit ("(BUG), need to update ordered_df_cols: inconsistent")
                    if set(funcs_res_df) != set(funcs_ordered_df_cols):
                        error_exit ("(BUG), need to update funcs_ordered_df_cols: inconsistent")
                    res_df = res_df[ordered_df_cols]
                    funcs_res_df = funcs_res_df[funcs_ordered_df_cols]

                    # Add decoded name fields
                    nf_df = pd.DataFrame([decode_tech_conf_name(v) for v in res_df['Tech-Config']])
                    funcs_nf_df = pd.DataFrame([decode_tech_conf_name(v) for v in funcs_res_df['Tech-Config']])
                    
                    #print res_df.to_string()

                    res_df = res_df.join(nf_df)
                    funcs_res_df = funcs_res_df.join(funcs_nf_df)
                    res_df.to_csv(outcsvfile, index=False)
                    funcs_res_df.to_csv(funcscsvfile, index=False)

    print "\n@ DONE!"

#~ def main()

def get_subsuming_mutants (initial_sm, surviving_sm, killed_on_surviving_ms):
    """ Return subsuming clusters
    """
    # Loading
    agg_mat_dat = matrixHardness.loadMatrix(initial_sm, None)
    for m_f in [surviving_sm, killed_on_surviving_sm]:
        tmp = matrixHardness.loadMatrix(m_f, None) 
        for m in tmp:
            if m in agg_mat_dat:
                agg_mat_dat[m] += tmp[m]
            else:
                agg_mat_dat[m] = tmp[m]
    # Compute subsumption
    eq, subs_clusters = magma_stats_algo.getSubsumingMutants (agg_mat_dat)
    return subs_clusters
#~ def get_subsuming_mutants()

def subs_clusters_of(subs_clusters, killed_mutants):
    k_m_set = set(killed_mutants)
    clusters_of = []
    for c_id, c in enumerate(subs_clusters):
        if len(set(c) & k_m_set) > 0:
            clusters_of.append(c_id)
    return clusters_of
#~ def subs_clusters_of()

def subsuming_ms(subs_clusters, killed_mutants):
    return 100.0 * len(subs_clusters_of(subs_clusters, killed_mutants)) / len(subs_clusters)
#~ def subsuming_ms()

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
