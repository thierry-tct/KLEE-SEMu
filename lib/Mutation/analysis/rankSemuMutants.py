
# This Module, for one project:
#    - load the data obtained after running SEMU
#    - sort the mutant from harder to kill to easier to kill, and relative equivalent
# It will return a json file having:
#    * list of relative equivalent mutants, Note the mutants not in the JSON are also all relative equivalent
#    * a map having as key a score, and value th elist of mutant ids having that hard to kill score    

############

import os, sys
import argparse
import json

import glob, shutil

import pandas as pd

class DIFF_CODES(self):
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
    for mutfile in glob.glob(os.path.join(indir,"mutant-*.semu"):
        mutID = int(re.findall('\d+', os.path.basename(mutFile))[0])
        df = pd.read_csv(mutFile)
        # each file should haev at least one row of data
        assert mutID == df.loc[0, 'MutantID'], "Problem with input file, Mutant id mismatch - "+mutFile
        df = df.drop('MutantID', axis=1)
        outObj[mutID] = df.to_dict("records")
    return outObj
# ~ def loadData()

def computeScores(inData):
    outData = {'Relative-Equivalent': [], 'Hardness': {}}
    for mutID in inData:
        # compute mutant hardness (0~1)
        m_hardness = 0.0

        ## XXX For now, just use proportions
        pc_diffs = 

        # add mutant to output
        if m_hardness not in outData['Hardness']:
            outData['hardness'][m_hardness] = set()
        outData['hardness'][m_hardness].add(mutID)
    return outData
#~ def computeScores()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="SEMU's output directory (klee-last), containing mutant difference data")
    parser.add_argument("-o", "--outfile", type=str, default=None, help="JSON output file for mutant classement")
    args = parser.parse_args()

    outFile = args.outfile
    inDir = args.indir

    assert (outFile is not None), "Must specify output file"

    print "# Starting", inDir, "..."

    inDataObj = loadData(inDir)

    outDataObj = computeScores(inDataObj)

    with open(outFile, "w") as fp:
        json.dump(outDataObj, fp)

    print "# Done"
#~ def main()

if __name__ == "__main__":
    main()
