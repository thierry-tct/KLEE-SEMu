
# Library to compute the stats after run.py

import os
import sys
import shutil
import json
import scipy.stats
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/mytools/MFI-V2.0/Analysis"))
import plotMerge

###### Non Parametic Vargha Delaney A12 ######
# Taken from -- https://gist.github.com/timm/5630491

def a12(lst1,lst2,pairwise=False, rev=True):
    "how often is x in lst1 more than y in lst2?"
    more = same = 0.0
    for i,x in enumerate(lst1):
        second = [lst2[i]] if pairwise else lst2
        for y in second:
            if   x==y : same += 1
            elif rev     and x > y : more += 1
            elif not rev and x < y : more += 1
    return (more + 0.5*same) / (len(lst1) if pairwise else len(lst1)*len(lst2))

def wilcoxon(list1, list2, isranksum=True):
    if isranksum:
        p_value = scipy.stats.ranksums(list1, list2)
    else:
        p_value = scipy.stats.wilcoxon(list1, list2)
    return p_value
#~ def wilcoxon()
#############################################
def compute_auc(in_x_list, in_y_list):
    """ SUM(abs(x2-x1) * abs(y2-y1) / 2 + (x2 - x1) * min(y1, y2))
    """
    # make sure both inlist are sorted by x
    assert len(set(in_x_list)) == len(in_x_list), "duplicate in in_x_list"
    assert len(in_x_list) == len(in_y_list), "X and Y have diffrent lengths"
    assert len(in_x_list) > 1, "At leats 2 elements required"
    x_list = []
    y_list = []
    for v_x, v_y in sorted(zip(in_x_list, in_y_list), key=lambda p: p[0]):
        x_list.append(v_x)
        y_list.append(v_y)
        assert v_y >= 0, "Only supports positive or null Y coordinate values"

    auc = 0.0
    prev_x = 0.0
    prev_y = 0.0
    for p_ind, x_val, y_val in enumerate(zip(x_list, y_list)):
        auc += (x_val - prev_x) * \
                                (min(y_val, prev_y) + abs(y_val - prev_y)/2.0)
        prev_x = x_val
        prev_y = y_val
    return auc

def compute_apfd(in_x_list, in_y_list):
    auc = compute_auc(in_x_list, in_y_list)
    apfd = auc / abs(max(x_list) - min(x_list))
    return apfd
########################

csv_file="Results.csv"
initial_json="Initial-dat.json"

def getProjRelDir():
    Modes = ["DEV", "KLEE", "NUM", "PASS"]
    curMode = "PASS"
    testSamplePercent = "100.0"
    eachIndirPrefix = 'TestGenFinalAggregated'

    assert curMode in Modes, "curMode not in Modes: "+curMode
    eachIndir = eachIndirPrefix + curMode + testSamplePercent

    projreldir = os.path.join('OUTPUT', eachIndir)

    return projreldir

PROJECT_ID_COL = "projectID"
SpecialTechs = {'_pureklee_': 'klee', '50_50_0_0_rnd_5_on_nocrit':'concrete'}
def libMain(outdir, proj2dir, projcommonreldir=None):
    merged_df = None
    all_initial = {}
    if projcommonreldir is None:
        projcommonreldir = getProjRelDir()

    # Load data
    for proj in proj2dir:
        fulldir = os.path.join(proj2dir[proj], projcommonreldir)
        full_csv_file = os.path.join(fulldir, csv_file)
        full_initial_json = os.path.join(fulldir, initial_json)

        tmp_df = pd.read_csv(full_csv_file, index_col=False)
        assert PROJECT_ID_COL not in tmp_df, PROJECT_ID_COL+" is in df"
        tmp_df[PROJECT_ID_COL] = [proj] * len(tmp_df.index)
        if merged_df is None:
            merged_df = tmp_df
        else:
            assert set(merged_df) == set(tmp_df), "Mismatch column for "+proj 
            merged_df = merged_df.concat(tmp_df, ignore_index=True, sort=False)

        with open(full_initial_json) as fp:
            all_initial[proj] = json.load(fp)

    # Compute the merged json
    merged_json_obj = {}
    merged_json_obj["Initial#Mutants"] = sum([int(all_initial[v]["Initial#Mutants"]) for v in all_initial])
    merged_json_obj["Initial#KilledMutants"] = sum([int(all_initial[v]["Initial#KilledMutants"]) for v in all_initial])
    merged_json_obj["Inintial#Tests"] = sum([int(all_initial[v]["Inintial#Tests"]) for v in all_initial])
    merged_json_obj["Initial-MS"] = sum([float(all_initial[v]["Initial-MS"]) for v in all_initial]) / len(all_initial)
    merged_json_obj["TestSampleMode"] = all_initial[all_initial.keys()[0]]["TestSampleMode"]
    merged_json_obj["MaxTestGen-Time(min)"] = all_initial[all_initial.keys()[0]]["MaxTestGen-Time(min)"]

    # save merged json
    with open(os.path.join(outdir, initial_json), 'w') as fp:
        json.dump(merged_json_obj, fp)
    
    # COMPUTATIONS ON DF
    timeCol = "TimeSnapshot(min)"
    config_columns = ["_precondLength","_mutantMaxFork", 
                        "_genTestForDircardedFrom", "_postCheckContProba", 
                        "_mutantContStrategy", "_maxTestsGenPerMut", 
                        "_disableStateDiffInTestgen"
                    ]
    other_cols = ["_testGenOnlyCriticalDiffs" ]

    msCol = "MS-INC"
    targetCol = "#Targeted"
    numMutsCol = "#Mutants"
    techConfCol = "Tech-Config"

    tech_confs = set(merged_df[techConfCol])
    projects = set(merged_df[PROJECT_ID_COL])
    ms_apfds = {p: {t_c: None for t_c in tech_confs} for p in projects}
    for p in ms_apfds:
        p_tmp_df = merged_df[merged_df[PROJECT_ID_COL] == p]
        for t_c in ms_apfds[p]:
            # get the data
            tmp_df = p_tmp_df[p_tmp_df[techConfCol] == t_c]
            ms_apfds[p][t_c] = compute_apfd(tmp_df[timeCol], tmp_df[msCol])
        tmp_df = p_tmp_df = None
    
    only_semu_cfg_df = merged_df[~merged_df[techConfCol].isin(SpecialTechs)]

    vals_by_conf = {}
    for c in config_columns:
        vals_by_conf[c] = list(set(only_semu_cfg_df[c]))

    techConfbyvalbyconf = {}
    for pc in config_columns:
        techConfbyvalbyconf[pc] = {}
        # process param config (get apfds)
        for val in vals_by_conf[pc]:
            keys = \
                set(only_semu_cfg_df[only_semu_cfg_df[pc] == val][techConfCol])
            techConfbyvalbyconf[pc][val] = keys

    def getListAPFDSForTechConf (t_c):
        v_list = []
        for p in ms_apfds:
            assert t_c in ms_apfds[p]
            v_list.append(ms_apfds[p][t_c])
        return v_list

    # XXX process APFDs (max, min, med)
    #proj_agg_func = sum
    proj_agg_func = np.median
    for pc in techConfbyvalbyconf:
        min_vals = {}
        max_vals = {}
        med_vals = {}
        for val in techConfbyvalbyconf[pc]:
            sorted_by_apfd_tmp = sorted(techConfbyvalbyconf[pc][val], \
                    key=lambda x: proj_agg_func(getListAPFDSForTechConf(x)))
            min_vals[val] = sorted_by_apfd_tmp[0]
            max_vals[val] = sorted_by_apfd_tmp[-1]
            med_vals[val] = sorted_by_apfd_tmp[len(sorted_by_apfd_tmp)/2]
        # plot
        plot_out_file = os.path.join(outdir, "perconf_apfd_"+pc)
        data = {val: {"min": getListAPFDSForTechConf(min_vals[val]), \
                        "med": getListAPFDSForTechConf(med_vals[val]), \
                        "max": getListAPFDSForTechConf(max_vals[val])} \
                                                for val in techConfbyvalbyconf}
        for sp in SpecialTechs:
            data[SpecialTechs[sp]] = {em:getListAPFDSForTechConf(sp) \
                                                for em in ['min', 'med','max']}
        # TODO Actual plot with data 
    
    # XXX Find best confs
    apfd_ordered_techconf_list = sorted(list(set(merged_df[techConfCol])), \
                    reverse=True, \
                    key=lambda x: proj_agg_func(getListAPFDSForTechConf(x)))
    best_val_tmp = proj_agg_func(getListAPFDSForTechConf(\
                                                apfd_ordered_techconf_list[0]))
    worse_val_tmp = proj_agg_func(getListAPFDSForTechConf(\
                                                apfd_ordered_techconf_list[-1]))
    best_elems = []
    worse_elems = []
    for i, v in enumerate(apfd_ordered_techconf_list):
        if proj_agg_func(getListAPFDSForTechConf(v)) >= best_val_tmp:
            best_elems.append(v)
        if proj_agg_func(getListAPFDSForTechConf(v)) <= worse_val_tmp:
            worse_elems.append(v)
    # TODO get corresponding param values and save as csv (best and worse)

    # XXX compare MS with comState time, %targeted, #testgen, WM%
    selectedTimes_minutes = [15, 30, 60, 120]
    # TODO get data and plot

def main():
    pass #libMain()

if __name__ == '__main__':
    main()