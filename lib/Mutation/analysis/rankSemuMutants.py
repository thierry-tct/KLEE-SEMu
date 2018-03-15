
# This Module, for one project:
#    - load the data obtained after running SEMU
#    - sort the mutant from harder to kill to easier to kill, and relative equivalent
# It will return a json file having:
#    * list of relative equivalent mutants, Note the mutants not in the JSON are also all relative equivalent
#    * a map having as key a score, and value th elist of mutant ids having that hard to kill score    

############

import os, sys
import argparse
import json, re

import glob, shutil

import pandas as pd
import numpy as np

class DIFF_CODES:
    def __init__(self):
        self.NO_DIFF = 0x00
        self.VARS_DIFF = 0x01
        self.RETCODE_DIFF_OTHERFUNC = 0x02
        self.RETCODE_DIFF_ENTRYFUNC = 0x04
        self.RETCODE_DIFF_MAINFUNC = 0x08
        self.OUTENV_DIFF = 0x10
        self.SYMBOLICS_DIFF = 0x20
        self.PC_DIFF = 0x40
        
    def hasAnyCodes (self, val, codeUnion):
        return (val & codeUnion != 0) 
    def hasAllCodes (self, val, codeUnion):
        return (val & codeUnion == codeUnion) 
    def isEqualTo (self, val, codeUnion):
        return (val == codeUnion) 
#~ class DIFF_CODES

ks = DIFF_CODES()

def getOrigState_MaxDepth(pc):
    return os.path.dirname(pc)
#~ def getOrigState_MaxDepth():

def loadData(indir, maxtime):
    outObj = {}
    for mutfile in glob.glob(os.path.join(indir,"mutant-*.semu")):
        mutID = int(re.findall('\d+', os.path.basename(mutfile))[0])
        df = pd.read_csv(mutfile)
        # each file should haev at least one row of data
        assert mutID == df.loc[0, 'MutantID'], "Problem with input file, Mutant id mismatch - "+mutFile
        df = df.drop('MutantID', axis=1)

        # Filter according to maxtime, only keep row written withing the maxtime
        df = df.loc[df['ellapsedTime(s)'] <= maxtime]
        if len(df) == 0: # At this time, this mutant was not 'semu killed'
            continue
        df = df.drop('ellapsedTime(s)', axis=1)

        tmpRowList = df.to_dict("records")
        outObj[mutID] = {}
        for row in tmpRowList:
            pc = os.path.join(str(row['OrigState']), str(row['MaxDepthID']), str(row['WatchPointID']))
            if pc not in outObj[mutID]:
                outObj[mutID][pc] = []
            del row['OrigState']
            del row['MaxDepthID']
            del row['WatchPointID']
            outObj[mutID][pc].append(row)

    return outObj
# ~ def loadData()

'''
    return True if mutId is harder than compID or if the two are incomparable
'''
def RightEasierOrIncomparableToLeft(mutID, compID, inData):
    # check whether the two are equal (same PCs and diffr-types per PCs)
    if len(inData[mutID]) == len(inData[compID]):
        if len(set(inData[mutID]) - set(inData[compID])) != 0:
            # Incomparable (different pcs)
            return True
        else:
            lscore = rscore = 0.0
            for pc in inData[mutID]:
                for ffl in inData[mutID][pc]:
                    for ffr in inData[compID][pc]:
                        if ffl['Diff_Type'] == ffr['Diff_Type']:
                            if ks.hasAnyCodes (ffl['Diff_Type'], ks.VARS_DIFF):
                                if float(ffl['nMaxFeasibleDiffs']) / ffl['nSoftClauses'] < float(ffr['nMaxFeasibleDiffs']) / ffr['nSoftClauses']: 
                                    lscore += 1
                                else:
                                    rscore += 1
                        else:
                            if ks.hasAllCodes (ffl['Diff_Type'], ffr['Diff_Type']):  # all codes of ffr are also in ffl (ffr is harder)
                                rscore += 1
                            if ks.hasAllCodes (ffr['Diff_Type'], ffl['Diff_Type']):  # all codes of ffl are also in ffr (ffl is harder)
                                lscore += 1
                            '''if int(ffl['Diff_Type']) < int(ffr['Diff_Type']):  # TODO: consider main/entry func, outenv, symbolic, pc diffs as equals
                                lscore += 1
                            else:
                                rscore += 1'''
                            
            return (lscore >= rscore)
    else:
        # Not having same number of PCs
        intersect = set(inData[mutID]) & set(inData[compID])
        if len(intersect) == len(inData[mutID]):
            return True # PCs of mutID are all in PCs of compID, thus compID is easier.
        elif len(intersect) == len(inData[compID]):
            return False # MutID is easier than compID
        else:
            # incomparable no inclusion TODO: maybe count the number of PCs and compare to see which is harder
            return True

#~ def RightEasierOrIncomparableToLeft()

'''
    Approximate the mutant hardness with regard to another mutants
'''
def computeScoresPairwise(inData):
    outData = {'Relative-Equivalent': [], 'Hardness': {}}
    for mutID in inData:
        # compute mutant hardness (0~1)
        m_hardness = 0.0

        for compID in inData:
            if mutID == compID:
                continue
            # Make sure that the number of mutants less hard or not comparable to mutID is in m_hardness. (non comparable will have same score)
            if RightEasierOrIncomparableToLeft(mutID, compID, inData):
                m_hardness += 1

        m_hardness /= len(inData)
        ## XXX For now, just use proportions
        #pc_diffs = 

        # add mutant to output
        if m_hardness == 1.0:
            #pass # Impossible
            outData['Relative-Equivalent'].append(mutID)
        else:
            if m_hardness not in outData['Hardness']:
                outData['Hardness'][m_hardness] = list()
            outData['Hardness'][m_hardness].append(mutID)
    return outData
#~ def computeScoresPairwise()


def getUnique (mutantsymbinfos):
    # Since we have two types of checking in SEMU (Out env check and watch point check), both happend on the same path thus we need to merge them
    # Furthermore, for a same sate, a mutant must appear only once for watchpoint (all corresponding mutants are terminated after watchpoint)
    # Therefore, we merge the instances by corresponding original state's value (OrigState) and MaxDepthID
    uniq_mutantsymbinfos = {}
    for e_pc in mutantsymbinfos:
        pc = getOrigState_MaxDepth(e_pc)
        if pc not in uniq_mutantsymbinfos:
            uniq_mutantsymbinfos[pc] = []
        uniq_mutantsymbinfos[pc] += mutantsymbinfos[e_pc]
    return uniq_mutantsymbinfos
#~ def getUnique ():


'''
    approximate the hardness of a Mutant. Take as input the mutant symbolic info Dict. containing diff per path.
    -------------------------------
    hardness = 1 - (kill / covers)
    ------------------------------
        kill -> diff
        covers -> diff or no-diff
    ------------------------------
    hardness = sum_{p in Paths}(percentage of tests killing M in p) / (Number of paths) 
    -----------------------------
        Approximate percentage of killing with state diffs.
            - Out Env diff -> 100%
            - Entry/main func ret code diff -> 100%
            - PC(program counter) diff -> 50% 
            - No Diff -> 0%
            - Symbolic diff -> 100%
            - state variable diff -> (num var diff) / (total num vars)

'''
def independentApproximateHardness(mutantsymbinfos):
    maxKillProba = 100.0
    weights = { ks.NO_DIFF: 0.0,
                ks.VARS_DIFF: 1.0, # Will be computed later
                ks.RETCODE_DIFF_OTHERFUNC: 1.0, #Unexpected
                ks.RETCODE_DIFF_ENTRYFUNC: maxKillProba,
                ks.RETCODE_DIFF_MAINFUNC: maxKillProba, 
                ks.OUTENV_DIFF: maxKillProba/2, 
                ks.SYMBOLICS_DIFF: maxKillProba, 
                ks.PC_DIFF: maxKillProba #/10 
            }
    surediffsSet = [ks.RETCODE_DIFF_ENTRYFUNC, ks.RETCODE_DIFF_MAINFUNC, ks.SYMBOLICS_DIFF] + [ks.PC_DIFF]
    unsureSet = set(weights) - set(surediffsSet)
    surediffsDiff = None
    if len(surediffsSet) > 0:
        surediffsDiff = surediffsSet[0]
        for d in surediffsSet[1:]:
            surediffsDiff |= d
    
    # XXX
    # Since we have two types of checking in SEMU (Out env check and watch point check), both happend on the same path thus we need to merge them
    # Furthermore, for a same sate, a mutant must appear only once for watchpoint (all corresponding mutants are terminated after watchpoint)
    # Therefore, we merge the instances by corresponding original state's value (OrigState) and MaxDepthID
    uniq_mutantsymbinfos = getUnique(mutantsymbinfos)  #mutantsymbinfos
    
    nPaths = len(uniq_mutantsymbinfos)
    killProba = 0.0
    for pc in uniq_mutantsymbinfos:
        local_proba = 0.0
        for instance in uniq_mutantsymbinfos[pc]:  # Multiple instance because we may have multiple environment calls
            dt = instance['Diff_Type']

            if surediffsDiff is not None and ks.hasAnyCodes(dt, surediffsDiff):
                local_proba = weights[surediffsSet[0]]
                # certainly kill, we are done with this path (break)
                break
            else:
                for subdifftype in unsureSet:
                    if ks.hasAnyCodes(dt, ks.VARS_DIFF):
                        local_proba += weights[subdifftype] * float(instance['nMaxFeasibleDiffs']) / instance['nSoftClauses']
                    else:
                        local_proba += weights[subdifftype]
        killProba += local_proba if local_proba <= maxKillProba else maxKillProba

    return (1 - killProba / nPaths / maxKillProba) #value betwen 0 and 1
#~ def independentApproximateHardness()

def breakTies (inData, mutlist):
    uniq_data = {}
    for mid in mutlist:
       uniq_data[mid] = getUnique(inData[mid])
    
    # use unique data to separate the mutants
    pairw = computeScoresPairwise(uniq_data)
    assert len(pairw['Relative-Equivalent']) == 0, "Must not be equivalent here"
    res = []
    for a_hn in sorted(pairw['Hardness'], reverse=True):
        res.append(sorted(pairw['Hardness'][a_hn])) #Sort only for easy to see

    assert len(mutlist) == sum([len(m_l) for m_l in res]), "Some mutants were omitted in above coputations"
    
    return res
#~ def breakTies ():

'''
    Use the state difference to approximate the hardness of each mutant regardless independently
'''
def computeScoresStateApproximate(inData):
    outData = {'Relative-Equivalent': [], 'Hardness': {}}
    for mutID in inData:
        # compute mutant hardness (0~1)
        m_hardness = independentApproximateHardness(inData[mutID])

        # add mutant to output
        if m_hardness == 1.0:
            #pass # Impossible
            outData['Relative-Equivalent'].append(mutID)
        else:
            if m_hardness not in outData['Hardness']:
                outData['Hardness'][m_hardness] = list()
            outData['Hardness'][m_hardness].append(mutID)

    breakTiesEnabled = True
    if breakTiesEnabled:
        # Break Tie as much as possible using a sort of pairwise comparison
        mutGroupsbyDechardness = []
        for h in sorted(outData['Hardness'], reverse=True):
            broken = breakTies (inData, outData['Hardness'][h])
            mutGroupsbyDechardness += broken

        newHardnesses = np.linspace(0.0, 1.0, len(mutGroupsbyDechardness),endpoint=False)[::-1]  #[::-1] reverses the array
        outData['Hardness'] = {v: muts for v, muts in zip(newHardnesses, mutGroupsbyDechardness) }
    return outData
#~ def computeScoresStateApproximate()

'''
Combine pairwise with state approximation
'''
def pairwise_state_Scores(inData):
    outData = {'Relative-Equivalent': [], 'Hardness': {}}
    intermDat = computeScoresPairwise(inData)
    outData['Relative-Equivalent'] = intermDat['Relative-Equivalent']
    ordered = []
    for ph in sorted(intermDat['Hardness'].keys(), reverse=True):
        state_hn = {}
        for mut in intermDat['Hardness'][ph]:
            m_h = independentApproximateHardness(inData[mut])
            if m_h not in state_hn:
                state_hn[m_h] = []
            state_hn[m_h].append(mut)
        ordered += [state_hn[mhv] for mhv in sorted(state_hn, reverse=True)]
    for pos,uniform_hn in enumerate(np.linspace(0.0, 1.0, len(ordered),endpoint=False)[::-1]): #[::-1] reverses the array
        outData['Hardness'][uniform_hn] = ordered[pos]
    assert len(outData['Hardness']) == len(ordered)
    return outData
#~ def pairwise_state_Scores()

def libMain(semuOutDir, outFilename, maxtime=float('inf')):
    assert (outFilename is not None), "Must specify output file"

    print "# Starting", semuOutDir, "..."

    inDataObj = loadData(semuOutDir, maxtime=maxtime)

    outDataObj = computeScoresPairwise(inDataObj)

    with open(outFilename+'-pairwise.json', "w") as fp:
        json.dump(outDataObj, fp)

    outDataObj = computeScoresStateApproximate(inDataObj)

    with open(outFilename+'-approxH.json', "w") as fp:
        json.dump(outDataObj, fp)

    outDataObj = pairwise_state_Scores(inDataObj)

    with open(outFilename+'-merged_PairApprox.json', "w") as fp:
        json.dump(outDataObj, fp)

    print "# Done"
#~ libMain()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="SEMU's output directory (klee-last), containing mutant difference data")
    parser.add_argument("-o", "--outfile", type=str, default=None, help="JSON output file for mutant classement")
    args = parser.parse_args()

    outFile = args.outfile
    inDir = args.indir

    libMain(inDir, outFile)
#~ def main()

if __name__ == "__main__":
    main()
