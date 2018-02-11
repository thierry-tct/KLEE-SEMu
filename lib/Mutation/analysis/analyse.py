#! /usr/bin/python

import os, sys
import json
import argparse
import random

import matplotlib
matplotlib.use('Agg') # Plot when using ssh -X (avoid server ... error)

import matplotlib.pyplot as plt

def loadJson(filename):
    with open(filename) as f:
        return json.load(f)
#~ loadJson()

def dumpJson(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f)
#~dumpJson()

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

# ROC
def ComputeMLAccuraciesPercent(scores, couplingproba, precisionrecallthresolds, cumulROC):
    PROBA_EQ = -1.0
    PROBA_COUPLED = 1.0
    lowboundscore = -1.0
    
    thresholds = precisionrecallthresolds

    nPositive = 0
    nNegative = 0
    for j in range(len(scores)):
        posj = sum([int(couplingproba[j][i] == PROBA_COUPLED) for i in range(len(scores[j]))])
        nPositive += posj
        nNegative += len(scores[j]) - posj
    # TPR is True Positive Rate, FPR iis False Positive Rate
    minlen = min([len(x) for x in scores])
    ROC = {"TPR":[None]*(1+minlen), "FPR":[None]*(1+minlen), "ThreshScore":[None]*(1+minlen)}
    seenPos = 0
    seenNeg = 0
    previdx = [0]*len(scores)
    for tcount in range(minlen):
        for si,pscores in enumerate(scores):
            curM = int(tcount*len(pscores)/float(minlen))
            for idx in range(previdx[si],curM):
                seenPos += int(couplingproba[si][idx] == PROBA_COUPLED)
                seenNeg += int(couplingproba[si][idx] < PROBA_COUPLED)
            previdx[si] = curM
        ROC["TPR"][tcount] = seenPos / float(nPositive)
        ROC["FPR"][tcount] = seenNeg / float(nNegative)
        ROC["ThreshScore"][tcount] = numpy.nan
    ROC["TPR"][minlen] = 1.0
    ROC["FPR"][minlen] = 1.0
    ROC["ThreshScore"][tcount] = numpy.nan

    if cumulROC is not None:
        cumulROC["TPR"] += ROC["TPR"]
        cumulROC["FPR"] += ROC["FPR"]
        cumulROC["ThreshScore"] += ROC["ThreshScore"]

    # Compute AUC
    AUC = 0.0
    for i in range(minlen):
        AUC += (ROC["FPR"][i+1]-ROC["FPR"][i]) * (ROC["TPR"][i+1]+ROC["TPR"][i])
    AUC *= 0.5
    #print "AUC =", AUC

    # Compute Precision and Recall
    precision = []
    recall = []
    F_Measure = []
    Accuracy = []
    randomPrecision = []
    randomRecall = []
    randomF_Measure = []
    randomAccurcy = []
    seenPos = 0
    seenNeg = 0
    previdx = [0]*len(scores)
    for tcount,t in enumerate(thresholds):
        for si,pscores in enumerate(scores):
            curM = int(t*len(pscores)/100.0)
            for idx in range(previdx[si],curM):
                seenPos += int(couplingproba[si][idx] == PROBA_COUPLED)
                seenNeg += int(couplingproba[si][idx] < PROBA_COUPLED)
            previdx[si] = curM
        if seenPos == 0 and seenNeg == 0:
            continue
        precision.append(seenPos / float(seenPos + seenNeg))
        recall.append(seenPos / float(nPositive))
        if precision[-1] > 0 or recall[-1] > 0:
            F_Measure.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1]))
        else:
            F_Measure.append(numpy.nan)
        Accuracy.append(float(seenPos + nNegative - seenNeg) / (nPositive + nNegative))

        randomPrecision.append(nPositive/float((nNegative+nPositive)))
        randomRecall.append(randomPrecision[-1] * (seenPos+seenNeg) / float(nPositive))
        if randomPrecision[-1] > 0 or randomRecall[-1] > 0:
            randomF_Measure.append(2 * randomPrecision[-1] * randomRecall[-1] / (randomPrecision[-1] + randomRecall[-1]))
        else:
            randomF_Measure.append(numpy.nan)
    return {"Precision":precision, "Recall":recall, "F-Measure":F_Measure, "AUC":AUC, "Accuracy":Accuracy, "RandomPrecision":randomPrecision,"RandomRecall":randomRecall,"RandomF-Measure":randomF_Measure} 
#~ def ComputeMLAccuraciesPercent()

def plotROC(semu_cumulROC, classic_cumulROC, ref_cumulROC, rand_cumulROC, title, figfilename=None, percentage=True):
    aucstring = ''
    if len(mlAccuracies) == 1:  #All are merged
        aucstring = " Curve with AUC=%s" % str(mlAccuracies[0]["AUC"])
    rocdf = pandas.DataFrame(cumulROC)
    p = ggplot.ggplot(rocdf, ggplot.aes(x='FPR', y='TPR', label='ThreshScore')) + ggplot.geom_point() #+ ggplot.geom_text(hjust=-0.2)
    #print p
    p += ggplot.geom_abline(linetype='dashed')
    p += ggplot.geom_area(alpha=0.2)
    p += ggplot.ggtitle("ROC"+aucstring) 
    p += ggplot.ylab('True Positive Rate') 
    p += ggplot.xlab('False Positive Rate') 
    p += ggplot.scale_y_continuous(limits=(0,1)) 
    p += ggplot.scale_x_continuous(limits=(0,1))
    if imageout:
        p.save(imageout+"-ROC.png")
    rocdf.to_csv(imageout+"-ROC.csv", index=False)
#def plotROC()



#-------

def plot4 (semuPair, classPair, randPair, refPair, title, figfilename=None, vertline=None, percentage=True):
    if len(SEMU_JSONs) > 1:
        colors = {'colors':['blue', 'cyan', 'magenta','green','red', 'yellow'], 'pos':0}
    else:
        colors = {'colors':['blue', 'green','red', 'yellow', 'cyan', 'magenta'], 'pos':0}
    def getColor(cdict):
        assert cdict['pos'] < len(cdict['colors']), "reached max colors"
        cdict['pos'] += 1
        return cdict['colors'][cdict['pos']-1]
    
    def makeFull(subjvals, refvals):
        res = []
        for i, v in enumerate(subjvals):
            res.append(v / refvals[i] if v>0 else 0)
        return res

    fullPlot = True

    #plt.style.use('ggplot')
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10,6)) #(13,9)) #(16,9)
    plt.gcf().subplots_adjust(bottom=0.27)
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 1.25
    fontsize = 20 #26
    for si in range(len(semuPair[0])):
        cval = getColor(colors)
        label = (os.path.splitext(SEMU_JSONs[si])[0] if len(SEMU_JSONs) > 1 else 'semu').replace("semu", "SymbolicExec")
        ysemu = makeFull(semuPair[1][si], refPair[1][0]) if fullPlot else semuPair[1][si]
        plt.plot(semuPair[0][si], ysemu, '-', color=cval , linewidth=3.0, alpha=0.6, label=label)
        plt.fill_between(semuPair[0][si], 0, ysemu, facecolor=cval, alpha=0.05)
        # Vertical line
        if vertline is not None:
            plt.axvline(x=vertline['semu'], linewidth=1, color=cval, linestyle='--')
    for ci in range(len(classPair[0])):
        cval = getColor(colors)
        label = os.path.splitext(CLASSIC_JSONs[ci])[0].replace('classic', 'Test-Cases')
        yclassic = makeFull(classPair[1][ci], refPair[1][0]) if fullPlot else classPair[1][ci]
        plt.plot(classPair[0][ci], yclassic, '-.', color=cval, linewidth=3.0, alpha=0.6, label=label)
        plt.fill_between(classPair[0][ci], 0, yclassic, facecolor=cval, alpha=0.05)
        # Vertivcal line
        if vertline is not None:
            plt.axvline(x=vertline['classic'], linewidth=1, color=cval, linestyle='--')
    if randPair[0] is not None and randPair[1] is not None:
        assert len(randPair[0]) == 1
        for ri in range(len(randPair[0])):
            cval = getColor(colors)
            plt.plot(randPair[0][ri], randPair[1][ri], ':', color=cval, linewidth=3.0, alpha=0.6, label='random')
            plt.fill_between(randPair[0][ri], 0, randPair[1][ri], facecolor=cval, alpha=0.05)

    if not fullPlot:
        assert len(refPair[0]) == 1
        plt.plot(refPair[0][0], refPair[1][0], 'r--', color='gray', alpha=0.755555, linewidth=2.0, label='ground-truth')
    #plt.fill_between(semuPair[0], 0, semuPair[1], facecolor='gray', alpha=0.05)

    met = "Percentage" if percentage else "Number"
    plt.xlabel(met + " of Selected Mutants (as Hard to Kill)", fontsize=fontsize)

    if fullPlot:
        lgd = plt.legend(bbox_to_anchor=(0., 0.98, 1., .102), loc=2, ncol=3, mode="expand", fontsize=fontsize, borderaxespad=0.)
        plt.ylabel("Precision", fontsize=fontsize)
        plt.ylim(-0.01,1.01)
        plt.xlim(-1,101)
    else:
        lgd = plt.legend(loc='upper center', ncol=1, fontsize='x-large', shadow=True)
        lgd.get_frame().set_facecolor('#FFFFFF')
        plt.ylabel(met + " of Hard to Kill Mutants", fontsize=fontsize)
        plt.title(title)

    plt.tight_layout()
    #plt.autoscale(enable=True, axis='x', tight=True)
    #plt.autoscale(enable=True, axis='y', tight=True)
    if figfilename is not None:
        plt.savefig(figfilename+".pdf", format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(figfilename+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.show()

    plt.close('all')
#~ plot4()

def hardnessPlot (xlist, ylist, figfilename=None):
    color = '#1b2431' #'xkcd:dark'
    plt.style.use('ggplot')
    plt.figure(figsize=(10,6)) #(16,9)
    plt.plot(xlist, ylist, '-', color=color, linewidth=3.0, alpha=0.5)
    plt.ylim(-0.06, 1)
    plt.xlabel("Selected Mutants percentage position")
    plt.ylabel("Hardness")
    plt.title("Hardness of Mutants according to Ground-Truth")
    plt.tight_layout()
    if figfilename is not None:
        plt.savefig(figfilename+".pdf", format='pdf')
        plt.savefig(figfilename+".png")
    else:
        plt.show()

    plt.close('all')
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

def getSemu_AND_OR_Classic(semu_dat, classic_dat, ground_dat, isAndNotOr):
    semu_res = [{} for i in range(len(semu_dat))]
    classic_res = [{} for i in range(len(classic_dat))]
    ground_res = [{} for i in range(len(ground_dat))]
    s_m = set([])
    c_m = set([])
    for si in range(len(semu_dat)):
        for hdness in semu_dat[si]:
            s_m |= set(semu_dat[si][hdness])
    for ci in range(len(classic_dat)):
        for hdness in classic_dat[ci]:
            c_m |= set(classic_dat[ci][hdness])
    intersect = s_m & c_m if isAndNotOr else s_m | c_m
    for tech_dat, tech_res in [(semu_dat[si], semu_res[si]) for si in range(len(semu_dat))] + [(classic_dat[ci], classic_res[ci]) for ci in range(len(classic_dat))] + [(ground_dat[gi], ground_res[gi]) for gi in range(len(ground_dat))]:
        for hdness in tech_dat:
            retain = intersect & set(tech_dat[hdness])
            if len(retain) > 0:
                tech_res[hdness] = list(retain)
    return semu_res, classic_res, ground_res
#~ def getSemu_AND_OR_Classic():

def getAnalyseInfos(semuData_l, classicData_l, groundtruthData_l, infoObj):
    for tech, tdata in [('semu', semuData_l), ('classic', classicData_l), ('ground', groundtruthData_l)]:
        nMuts = 0
        nHardLevels = 0
        for var in tdata:
            nHardLevels += len(var)
            for hdn in var:
                nMuts += len(var[hdn])
        nMuts /= len(tdata)
        nHardLevels /= len(tdata)
        infoObj[tech]["#Mutants"] = nMuts
        infoObj[tech]["#hardnessLevels"] = nHardLevels
#~ def getAnalyseInfos()

SEMU_JSONs = ["semu"+var+".json" for var in ['-pairwise']] #, '-approxH', '-merged_PairApprox']]
CLASSIC_JSONs = ["classic.json"]
GROUNDTRUTH_JSONs = ["groundtruth.json"]

RAND_REP = 100

def libMain(jsonsdir, mutantListForRandom=None, mutantInfoFile=None):
    semuData_all_l = [loadJson(os.path.join(jsonsdir, semu_json))['Hardness'] for semu_json in SEMU_JSONs]
    classicData_all_l = [loadJson(os.path.join(jsonsdir, classic_json))['Hardness'] for classic_json in CLASSIC_JSONs]
    groundtruthData_all_l = [loadJson(os.path.join(jsonsdir, ground_json))['Hardness'] for ground_json in GROUNDTRUTH_JSONs]

    graphTypes = ["all", "semuANDclassic", "semuORclassic"]
    analyseInfo = {gt:{'semu':{}, 'classic':{}, 'ground':{}} for gt in graphTypes}
    for title in graphTypes:
        if title == 'all':
            semuData_l, classicData_l, groundtruthData_l = semuData_all_l, classicData_all_l, groundtruthData_all_l
        elif title == 'semuANDclassic':
            semuData_l, classicData_l, groundtruthData_l = getSemu_AND_OR_Classic(semuData_all_l, classicData_all_l, groundtruthData_all_l, isAndNotOr=True)
        elif title == 'semuORclassic':
            semuData_l, classicData_l, groundtruthData_l = getSemu_AND_OR_Classic(semuData_all_l, classicData_all_l, groundtruthData_all_l, isAndNotOr=False)
        else:
            assert False, "Invalid type of graph: "+title+". Expected from: "+str(graphTypes)

        if len(semuData_l[0]) == 0 or len(classicData_l[0]) == 0:
            print "Analysis@Warning: Skipped Title because empty data."
            continue
        
        # Get analysis information
        getAnalyseInfos(semuData_l, classicData_l, groundtruthData_l, analyseInfo[title])
        dumpJson(analyseInfo, os.path.join(jsonsdir, "analyse-info.json"))

        semuSelSizes = [[None] * RAND_REP] * len(semuData_l)
        semuNHard = [[None] * RAND_REP] * len(semuData_l)
        classicSelSizes = [[None] * RAND_REP] * len(classicData_l)
        classicNHard = [[None] * RAND_REP] * len(classicData_l)
        print "Processing Semu and Classic for", title, "..."
        assert len(groundtruthData_l) == 1, "Mus have one groundtruth"
        for r in range(RAND_REP):
            for si in range(len(semuData_l)):
                semuSelSizes[si][r], semuNHard[si][r] = computePoints(semuData_l[si], groundtruthData_l[0])
            for ci in range(len(classicData_l)):
                classicSelSizes[ci][r], classicNHard[ci][r] = computePoints(classicData_l[ci], groundtruthData_l[0])

        for si in range(len(semuSelSizes)):
            semuSelSizes[si], semuNHard [si]= average(semuSelSizes[si], semuNHard[si])
        for ci in range(len(classicSelSizes)):
            classicSelSizes[ci], classicNHard[ci] = average(classicSelSizes[ci], classicNHard[ci])

        gtHardness = []
        groundtruthSelSize, groundtruthNHard = computePoints(groundtruthData_l[0], groundtruthData_l[0], refHardness=gtHardness)
        groundtruthSelSize = [groundtruthSelSize]
        groundtruthNHard = [groundtruthNHard]

        mutsShuffled = None
        if mutantInfoFile is not None:
            mutsShuffled = [int(m) for m in loadJson(mutantInfoFile).keys()]
        if mutantListForRandom is not None:
            mutsShuffled = list(mutantListForRandom)
        if mutsShuffled is not None:
            randSelSizes = [None] * RAND_REP
            randNHard = [None] * RAND_REP
            print "Processing Random..."
            #for i in groundtruthData:
            #    mutsShuffled += list(groundtruthData[i])
            for r in range(RAND_REP):
                random.shuffle(mutsShuffled)
                randSelSizes[r], randNHard[r] = computePoints({float(pos)/len(mutsShuffled): set([mutsShuffled[pos]]) for pos in range(len(mutsShuffled))}, groundtruthData_l[0])
            randSelSizes, randNHard = average(randSelSizes, randNHard)
            randSelSizes = [randSelSizes]
            randNHard = [randNHard]
        else:
            randSelSizes = None
            randNHard = None

        print "Plotting ", title, "..."
        figDir = jsonsdir
        visitedMutsPercent = {'semu':analyseInfo[title]['semu']['#Mutants']*100.0/analyseInfo[title]['ground']['#Mutants'], 'classic':analyseInfo[title]['classic']['#Mutants']*100.0/analyseInfo[title]['ground']['#Mutants']}
        plot4((semuSelSizes,semuNHard), (classicSelSizes, classicNHard), (randSelSizes, randNHard), (groundtruthSelSize, groundtruthNHard), "Hard to Kill Mutant Among Selected", os.path.join(figDir, "comparison-"+title), vertline=visitedMutsPercent)
        hardnessPlot(gtHardness[0], gtHardness[1], os.path.join(figDir, "hardness-"+title))
    
#~ libMain()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonsdir", help="directory Containing both Json files")
    parser.add_argument("--mutantsinfofile", type=str, default=None, help="Pass the mutant info fie so that random selection can be executed")
    parser.add_argument("--mutantlistforrandom", type=str, default=None, help="Pass the candidate mutants list so that random selection can be executed")
    args = parser.parse_args()
    
    candMutsList = None
    if args.mutantlistforrandom is not None:
        candMutsList = []
        with open(args.mutantlistforrandom) as f:
            for midstr in f:
                if len(midstr.strip()) > 0:
                    candMutsList.append(int(midstr.strip()))

    libMain(args.jsonsdir, mutantListForRandom=candMutsList, mutantInfoFile=args.mutantsinfofile)
#~ main()

if __name__ == "__main__":
    main()
