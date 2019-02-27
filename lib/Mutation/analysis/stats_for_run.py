
# Library to compute the stats after run.py

from __future__ import print_function
import os
import sys
import shutil
import json
import argparse
import shutil
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
    apfd = auc / abs(max(in_x_list) - min(in_x_list))
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
    eachIndir = eachIndirPrefix + curMode + '_' + testSamplePercent

    projreldir = os.path.join('OUTPUT', eachIndir)

    return projreldir
#~deg getProjRelDir()

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
            merged_df = merged_df.append(tmp_df, ignore_index=True, sort=False)

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
    stateCompTimeCol = "StateComparisonTime(s)"
    numGenTestsCol = "#GenTests"
    numForkedMutStatesCol = "#MutStatesForkedFromOriginal"
    mutPointNoDifCol = "#MutStatesEqWithOrigAtMutPoint"

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

    colors_bw = ['white', 'whitesmoke', 'lightgray', 'silver', 'darkgrey', \
                                                    'gray', 'dimgrey', "black"]
    colors = ["green", 'blue', 'red', "black", "maroon", "magenta", "cyan"]
    linestyles = ['solid', 'solid', 'dashed', 'dashed', 'dashdot', 'dotted', \
                                                                    'solid']
    linewidths = [1.75, 1.75, 2.5, 2.5, 3.25, 3.75, 2]

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
        tmp_all_vals = []
        for g in data:
            for m in data[g]:
                tmp_all_vals.append(data[g][m])
        min_y = min(tmp_all_vals)
        max_y = max(tmp_all_vals)
        assert min_y >= 0 and min_y <= 100
        assert max_y >= 0 and max_y <= 100
        # Actual plot with data 
        # TODO arange max_y, min_y and step_y
        if max_y - min_y >= 10:
            max_y = int(max_y) + 2 
            min_y = int(min_y) - 1 
            step_y = (max_y - min_y) / 10
        else:
            step_y = 1
            rem_tmp = 10 - max_y - min_y + 1
            if 100 - max_y < rem_tmp/2:
                min_y = int(min_y - (rem_tmp - (100 - max_y)))
                max_y = 100
            elif min_y < rem_tmp/2:
                max_y = int(max_y + (rem_tmp - min_y)) 
                min_y = 0
            else:
                max_y = int(max_y + rem_tmp/2)
                min_y = int(min_y - rem_tmp/2)
        yticks_range = range(min_y, max_y+1, step_y)
        plotMerge.plot_Box_Grouped(data, plot_out_file, colors_bw, \
                                "AVERAGE MS", yticks_range=yticks_range)
    
    # XXX Find best and worse confs
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
    # get corresponding param values and save as csv (best and worse)
    best_df_obj = []
    worse_df_obj = []
    for elem_list, df_obj_list in [(best_elems, best_df_obj), \
                                                (worse_elems, worse_df_obj)]:
        for v in elem_list:
            row = {}
            for pc in techConfbyvalbyconf:
                for val in techConfbyvalbyconf[pc]:
                    if v in techConfbyvalbyconf[pc][val]:
                        assert pc not in row, "BUG"
                        row[pc] = val
            row[techConfCol] = v
            row['MS_INC_APFD'] = proj_agg_func(getListAPFDSForTechConf(v))
            df_obj_list.append(row)
    best_df = pd.DataFrame(best_df_obj)
    worse_df = pd.DataFrame(worse_df_obj)
    best_df_file = os.path.join(outdir, "best_tech_conf_apfd.csv")
    worse_df_file = os.path.join(outdir, "worse_tech_conf_apfd.csv")
    best_df.to_csv(best_df_file, index=False)
    worse_df.to_csv(worse_df_file, index=False)

    # XXX compare MS with comState time, %targeted, #testgen, WM%
    selectedTimes_minutes = [15, 30, 60, 120]
    # TODO get data and plot
    #for time_snap in selectedTimes_minutes:
    #    time_snap_df = \
    #            only_semu_cfg_df[int(only_semu_cfg_df[timeCol]) == time_snap]
#~ def libMain()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=None, \
            help="Output directory, will be deleted and recreated if exists")
    parser.add_argument("-i", "--intopdir", default=None, \
            help="Top directory where to all projects are"\
                                        +" (will search the finished ones)")
    args = parser.parse_args()

    outdir = args.output
    intopdir = args.intopdir
    assert outdir is not None
    assert intopdir is not None
    assert os.path.isdir(intopdir)

    if os.path.isdir(outdir):
        if raw_input("\nspecified output exists. Clear it? [y/n] ").lower() \
                                                                        == 'y':
            shutil.rmtree(outdir)
        else:
            print("# please specify another outdir")
            return
    os.mkdir(outdir)
    proj2dir = {}
    for f_d in os.listdir(intopdir):
        direct = os.path.join(intopdir, f_d, getProjRelDir())
        if os.path.isfile(os.path.join(direct, csv_file)):
            proj2dir[f_d] = os.path.join(intopdir, f_d)
    if len(proj2dir) > 0:
        print ("# Calling libMain...")
        libMain(outdir, proj2dir)
        print("# DONE")
    else:
        print("# !! No good project found")

if __name__ == '__main__':
    main()