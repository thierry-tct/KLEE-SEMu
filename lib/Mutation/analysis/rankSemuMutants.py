
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
    def hasAllCodes (self, va2yyl, codeUnion):
        return (val & codeUnion == codeUnion) 
    def isEqualTo (self, va2yyl, codeUnion):
        return (val == codeUnion) 
#~ class DIFF_CODES

ks = DIFF_CODES()

def loadData(indir):
    outObj = {}
    for mutfile in glob.glob(os.path.join(indir,"mutant-*.semu")):
        mutID = int(re.findall('\d+', os.path.basename(mutfile))[0])
        df = pd.read_csv(mutfile)
        # each file should haev at least one row of data
        assert mutID == df.loc[0, 'MutantID'], "Problem with input file, Mutant id mismatch - "+mutFile
        df = df.drop('MutantID', axis=1)
        tmpRowList = df.to_dict("records")
        outObj[mutID] = {}
        for row in tmpRowList:
            pc = os.path.join(str(row['OrigState']), str(row['WatchPointID']))
            if pc not in outObj[mutID]:
                outObj[mutID][pc] = []
            del row['OrigState']
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
                            if float(ffl['nMaxFeasibleDiffs']) / ffl['nSoftClauses'] < float(ffr['nMaxFeasibleDiffs']) / ffr['nSoftClauses']:
                                lscore += 1
                            else:
                                rscore += 1
                        else:
                            if int(ffl['Diff_Type']) > int(ffr['Diff_Type']):
                                lscore += 1
                            else:
                                rscore += 1
                            
            return (lscore > rscore)
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

def computeScores(inData):
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
            pass # Impossible
            #outData['Relative-Equivalent'].append(mutID)
        else:
            if m_hardness not in outData['Hardness']:
                outData['Hardness'][m_hardness] = list()
            outData['Hardness'][m_hardness].append(mutID)
    return outData
#~ def computeScores()

def libMain(semuOutDir, outFilename):
    assert (outFilename is not None), "Must specify output file"

    print "# Starting", semuOutDir, "..."

    inDataObj = loadData(semuOutDir)

    outDataObj = computeScores(inDataObj)

    with open(outFilename+'.json', "w") as fp:
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
