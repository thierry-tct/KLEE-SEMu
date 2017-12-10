#! /usr/bin/python

import os, sys
import json
import argparse
import random

import matplotlib.pyplot as plt

def loadJson(filename):
    with open(filename) as f:
        return json.load(f)
#~ loadJson()

def average(xarrlist, yarrlist):
    for k in range(1, len(xarrlist)):
        assert xarrlist[k] == xarrlist[0], "All the x must be same"
    y = []
    for j in range (len(xarrlist[0])):
        y.append(0.0)
        for i in range (len(xarrlist)):
            y[-1] += yarrlist[i][j]
        y[-1] /= len(yarrlist)
    return xarrlist[0], y
#~ average()

def plot4 (semuPair, classPair, randPair, refPair, title, figfilename=None, percentage=True):
    plt.style.use('ggplot')
    plt.figure(figsize=(16,9))
    plt.plot(semuPair[0], semuPair[1], 'b-', linewidth=3.0, alpha=0.6, label='semu')
    plt.fill_between(semuPair[0], 0, semuPair[1], facecolor='blue', alpha=0.05)
    plt.plot(classPair[0], classPair[1], 'g-.', linewidth=3.0, alpha=0.6, label='classic')
    plt.fill_between(classPair[0], 0, classPair[1], facecolor='green', alpha=0.05)
    plt.plot(randPair[0], randPair[1], 'r:', linewidth=3.0, alpha=0.6, label='random')
    plt.fill_between(randPair[0], 0, randPair[1], facecolor='red', alpha=0.05)
    plt.plot(refPair[0], refPair[1], 'r--', color='gray', alpha=0.755555, linewidth=2.0, label='ground-truth')
    #plt.fill_between(semuPair[0], 0, semuPair[1], facecolor='gray', alpha=0.05)
    plt.legend(loc='upper center', ncol=1, fontsize='x-large')
    met = "Percentage" if percentage else "Number"
    plt.xlabel(met + " of Selected Mutants")
    plt.ylabel(met + " of Hard to Kill Mutants")
    plt.title(title)
    plt.tight_layout()
    #plt.autoscale(enable=True, axis='x', tight=True)
    #plt.autoscale(enable=True, axis='y', tight=True)
    if figfilename is not None:
        plt.savefig(figfilename+".png")
    else:
        plt.show()
#~ plot4()

def hardnessPlot (xlist, ylist, figfilename=None):
    plt.style.use('ggplot')
    plt.figure(figsize=(16,9))
    plt.plot(xlist, ylist, '-', color='xkcd:dark', linewidth=3.0, alpha=0.5)
    plt.ylim(-0.06, 1)
    plt.xlabel("Selected Mutants percentage position")
    plt.ylabel("Hardness")
    plt.title("Hardness of Mutants according to Ground-Truth")
    plt.tight_layout()
    if figfilename is not None:
        plt.savefig(figfilename+".png")
    else:
        plt.show()
#~ hardnessPlot()

'''
    Make the subject have same points as the reference and pairwise compare
'''
def computePoints(subjObj, refObj, refHardness=None, tieRandom=True, percentage=True):
    ss = []
    nh = []
    checkpoints = [0]
    refMutsOrdered = []
    subjMutsOrdered = []
    for rscore in sorted(refObj, reverse=True, key=lambda x: float(x)):
        checkpoints.append(checkpoints[-1] + len(refObj[rscore]))  #right after last
        refMutsOrdered += list(refObj[rscore])

    if refHardness is not None:
        assert type(refHardness) == list, "must be list, wll have x at pos 0, y at pos 1"
        refHardness.append(range(1, len(refMutsOrdered) + 1))
        refHardness.append([])
        for rscore in sorted(refObj, reverse=True, key=lambda x: float(x)):
            refHardness[-1] += [float(rscore)] * len(refObj[rscore])
        assert len(refHardness[0]) == len(refHardness[1]), "Bug: x and y must have same length"
        if percentage:
            for i in range(len(refHardness[0])):
                refHardness[0][i] *= 100.0 / len(refHardness[0])

    for sscore in sorted(subjObj, reverse=True, key=lambda x: float(x)):
        if tieRandom:
            # RandomSel to break ties
            tmpl = list(subjObj[sscore])
            random.shuffle(tmpl)
            subjMutsOrdered += tmpl
        else:
            # Take first those that are not Hard (worst case)
            subjMutsOrdered += sorted(subjObj[sscore], key=lambda x:int(x in refSelMuts))
    for c in checkpoints:
        ss.append(c)
        nh.append(len(set(refMutsOrdered[:c]) & set(subjMutsOrdered[:c])))

    # Put the value in percentage form, if enabled
    if percentage:
        nMutants = len(refMutsOrdered)
        for i in range(len(ss)):
            ss[i] *= 100.0 / nMutants
            nh[i] *= 100.0 / nMutants
    return ss, nh
#~ computePoints()


SEMU_JSON = "semu.json"
CLASSIC_JSON = "classic.json"
GROUNDTRUTH_JSON = "groundtruth.json"

RAND_REP = 100

def libMain(jsonsdir):
    semuData = loadJson(os.path.join(jsonsdir, SEMU_JSON))['Hardness']
    classicData = loadJson(os.path.join(jsonsdir, CLASSIC_JSON))['Hardness']
    groundtruthData = loadJson(os.path.join(jsonsdir, GROUNDTRUTH_JSON))['Hardness']

    semuSelSizes = [None] * RAND_REP
    semuNHard = [None] * RAND_REP
    classicSelSizes = [None] * RAND_REP
    classicNHard = [None] * RAND_REP
    print "Processing Semu and Classic..."
    for r in range(RAND_REP):
        semuSelSizes[r], semuNHard[r] = computePoints(semuData, groundtruthData)
        classicSelSizes[r], classicNHard[r] = computePoints(classicData, groundtruthData)
    semuSelSizes, semuNHard = average(semuSelSizes, semuNHard)
    classicSelSizes, classicNHard = average(classicSelSizes, classicNHard)

    gtHardness = []
    groundtruthSelSize, groundtruthNHard = computePoints(groundtruthData, groundtruthData, refHardness=gtHardness)

    randSelSizes = [None] * RAND_REP
    randNHard = [None] * RAND_REP
    mutsShuffled = []
    print "Processing Semu and Random..."
    for i in groundtruthData:
        mutsShuffled += list(groundtruthData[i])
    for r in range(RAND_REP):
        random.shuffle(mutsShuffled)
        randSelSizes[r], randNHard[r] = computePoints({float(pos)/len(mutsShuffled): set([mutsShuffled[pos]]) for pos in range(len(mutsShuffled))}, groundtruthData)
    randSelSizes, randNHard = average(randSelSizes, randNHard)

    print "Plotting ..."
    figDir = jsonsdir
    plot4((semuSelSizes,semuNHard), (classicSelSizes, classicNHard), (randSelSizes, randNHard), (groundtruthSelSize, groundtruthNHard), "Hard to Kill Mutant Among Selected", os.path.join(figDir, "comparison"))
    hardnessPlot(gtHardness[0], gtHardness[1], os.path.join(figDir, "hardness"))
    
#~ libMain()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonsdir", help="directory Containing both Json files")
    args = parser.parse_args()

    libMain(args.jsonsdir)
#~ main()

if __name__ == "__main__":
    main()
