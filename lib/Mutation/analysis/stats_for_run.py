
# Library to compute the stats after run.py
# Example
# python ~/mytools/klee-semu/src/lib/Mutation/analysis/stats_for_run.py -i SEMU_EXECUTION -o RESULTS --maxtimes "5 15 30 60 120"

from __future__ import print_function
import os
import sys
import shutil
import json
import argparse
import shutil
import scipy.stats
import numpy as np
import itertools

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
    if len(in_x_list) == 0:
        assert "Empty list passed to compute_auc. Pass 2 or more elements"

    x_list = []
    y_list = []
    for v_x, v_y in sorted(zip(in_x_list, in_y_list), key=lambda p: p[0]):
        x_list.append(v_x)
        y_list.append(v_y)
        assert v_y >= 0, "Only supports positive or null Y coordinate values"

    if len(x_list) == 1:
        print ("# WARNING: only one element for compute_auc")
        return y_list[0]

    auc = 0.0
    prev_x = None
    prev_y = None
    for p_ind, (x_val, y_val) in enumerate(zip(x_list, y_list)):
        if prev_x is not None:
            auc += (x_val - prev_x) * \
                                (min(y_val, prev_y) + abs(y_val - prev_y)/2.0)
        prev_x = x_val
        prev_y = y_val
    return auc
#~ def compute_auc()
  
def compute_apfd(in_x_list, in_y_list):
    auc = compute_auc(in_x_list, in_y_list)
    if len(in_x_list) > 1:
        apfd = auc / abs(max(in_x_list) - min(in_x_list))
    else:
        apfd = auc
    return apfd
#~ def compute_apfd()
########################


def make_twoside_plot(left_y_vals, right_y_vals, img_out_file=None, \
                                    x_label="X", y_left_label="Y_LEFT", \
                                    y_right_label="Y_RIGHT", separate=True,\
                                    left_stackbar_legends=None,\
                                    right_stackbar_legends=None):

    if separate:
        fig=plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, sharex = ax1)
    else:
        fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel(x_label)

    flierprops = dict(marker='o', markersize=2, linestyle='none')

    ax1.set_ylabel(y_left_label, color=color)
    if left_stackbar_legends is None:
        bp1 = ax1.boxplot(left_y_vals, flierprops=flierprops)
        for element in ['boxes', 'whiskers', 'fliers', \
                                                'means', 'medians', 'caps']:
            plt.setp(bp1[element], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
    else:
        bottoms = [np.array([0]*len(left_y_vals[0]))]
        for v in left_y_vals:
            bottoms.append(bottoms[-1] + np.array(v))
        ind = np.arange(len(left_y_vals[0]))
        p = [None] * len(left_stackbar_legends)
        for i in range(len(left_stackbar_legends)):
            p[i] = ax1.bar(ind, left_y_vals[i], bottom=bottoms[i])
        ax1.legend(p, left_stackbar_legends)

    if not separate:
        # instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()  

    color = 'tab:red'
    ax2.set_ylabel(y_right_label, color=color)  # we already handled the x-label with ax1
    if right_stackbar_legends is None:
        bp2 = ax2.boxplot(right_y_vals, flierprops=flierprops)
        for element in ['boxes', 'whiskers', 'fliers', \
                                                'means', 'medians', 'caps']:
            plt.setp(bp2[element], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    else:
        bottoms = [np.array([0]*len(right_y_vals[0]))]
        for v in right_y_vals:
            bottoms.append(bottoms[-1] + np.array(v))
        ind = np.arange(len(right_y_vals[0]))
        p = [None] * len(right_stackbar_legends)
        for i in range(len(right_stackbar_legends)):
            p[i] = ax2.bar(ind, right_y_vals[i], bottom=bottoms[i])
        ax2.legend(p, right_stackbar_legends)
        #ax.margins(0.05)

    plt.xticks([])

    plt.tight_layout()
    if img_out_file is None:
        plt.show()
    else:
        plt.savefig(img_out_file+".pdf", format="pdf")
    plt.close('all')
#~ def make_twoside_plot()


csv_file="Results.csv"
funcs_csv_file="Results-byfunctions.csv"
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
SPECIAL_TECHS = {'_pureklee_': 'klee', '50_50_0_0_rnd_5_on_nocrit':'concrete'}
CONCRETE_KEY = '50_50_0_0_rnd_5_on_nocrit'
KLEE_KEY = '_pureklee_'
def libMain(outdir, proj2dir, use_func=False, customMaxtime=None, \
                projcommonreldir=None, onlykillable=False, no_concrete=False):
    merged_df = None
    all_initial = {}
    if projcommonreldir is None:
        projcommonreldir = getProjRelDir()

    input_csv = funcs_csv_file if use_func else csv_file

    # Load data
    for proj in proj2dir:
        fulldir = os.path.join(proj2dir[proj], projcommonreldir)
        full_csv_file = os.path.join(fulldir, input_csv)
        full_initial_json = os.path.join(fulldir, initial_json)

        tmp_df = pd.read_csv(full_csv_file, index_col=False)

        # Only killable
        if onlykillable:
            msCol = "MS-INC"
            assert msCol in tmp_df, "no "+msCol+"Column"
            mses = [float(v) for v in tmp_df[msCol]]
            if sum(mses) == 0:
                continue

        assert PROJECT_ID_COL not in tmp_df, PROJECT_ID_COL+" is in df"
        if use_func:
            funcNameCol = "FunctionName"
            assert funcNameCol in tmp_df, \
                                        "invalid func csv file: "+full_csv_file
            tmp_df[PROJECT_ID_COL] = list(\
                    map(lambda x: os.path.join(proj, x), tmp_df[funcNameCol]))
        else:
            tmp_df[PROJECT_ID_COL] = [proj] * len(tmp_df.index)

        if merged_df is None:
            merged_df = tmp_df
        else:
            assert set(merged_df) == set(tmp_df), "Mismatch column for "+proj 
            merged_df = merged_df.append(tmp_df, ignore_index=True)

        with open(full_initial_json) as fp:
            all_initial[proj] = json.load(fp)

    # Compute the merged json
    merged_json_obj = {}
    merged_json_obj["Projects"] = list(all_initial) 
    merged_json_obj["#Projects"] = len(all_initial) 
    merged_json_obj["Initial#Mutants"] = \
            sum([int(all_initial[v]["Initial#Mutants"]) for v in all_initial])
    merged_json_obj["Initial#KilledMutants"] = \
            sum([int(all_initial[v]["Initial#KilledMutants"]) \
                                                        for v in all_initial])
    merged_json_obj["Inintial#Tests"] = \
            sum([int(all_initial[v]["Inintial#Tests"]) for v in all_initial])
    merged_json_obj["Initial-MS"] = \
            sum([float(all_initial[v]["Initial-MS"]) \
                                    for v in all_initial]) / len(all_initial)
    merged_json_obj["TestSampleMode"] = \
                        all_initial[all_initial.keys()[0]]["TestSampleMode"]
    merged_json_obj["MaxTestGen-Time(min)"] = \
                    all_initial[all_initial.keys()[0]]["MaxTestGen-Time(min)"]
    if use_func:
        merged_json_obj["#Functions"] = 0
        merged_json_obj['By-Functions'] = {}
        for proj in all_initial:
            merged_json_obj["#Functions"] += \
                                        len(all_initial[proj]['By-Functions'])
            for func in all_initial[proj]['By-Functions']:
                func_name_merged = os.path.join(proj, func)
                merged_json_obj['By-Functions'][func_name_merged] = \
                                        all_initial[proj]['By-Functions'][func]

    # save merged json
    with open(os.path.join(outdir, initial_json), 'w') as fp:
        json.dump(merged_json_obj, fp, indent=2, sort_keys=True)
    
    # COMPUTATIONS ON DF
    timeCol = "TimeSnapshot(min)"
    config_columns = ["_precondLength","_mutantMaxFork", 
                        "_genTestForDircardedFrom", "_postCheckContProba", 
                        "_mutantContStrategy", "_maxTestsGenPerMut", 
                        "_disableStateDiffInTestgen"
                    ]
    other_cols = ["_testGenOnlyCriticalDiffs" ]

    msCol = "MS-INC"
    numFailTestsCol = "#FailingTests"
    targetCol = "#Targeted"
    numMutsCol = "#Mutants"
    covMutsCol = "#Covered"
    killMutsCol = "#Killed"
    techConfCol = "Tech-Config"
    stateCompTimeCol = "StateComparisonTime(s)"
    numGenTestsCol = "#GenTests"
    numForkedMutStatesCol = "#MutStatesForkedFromOriginal"
    mutPointNoDifCol = "#MutStatesEqWithOrigAtMutPoint"

    propNoDiffOnForkedMutsStatesCol = "percentageNodiffFoundAtMutPoint"
    assert propNoDiffOnForkedMutsStatesCol not in merged_df, \
                                                    "Use different key (BUG)" 
    propNDOFMS = []
    for ind, row in merged_df.iterrows():
        num_denom = []
        for v in [row[mutPointNoDifCol], row[numForkedMutStatesCol]]:
            try:
                num_denom.append(float(v))
            except ValueError:
                if v == '-':
                    num_denom = v
                    break
                else:
                    print ("\n# Error: Invalid number(", v, ") for column:",\
                                    mutPointNoDifCol if len(num_denom) == 0 \
                                                else numForkedMutStatesCol)
                    print ("# ... Tech_conf is:", row[techConfCol], '\n')
                    assert False
        if type(num_denom) == list:
            try:
                num_denom = 100.0 * num_denom[0] / num_denom[1]
            except ZeroDivisionError:
                assert num_denom[0] == 0, \
                            "total is 0 by equivalent not zero(BUG in run.py)"
                num_denom = 0
        propNDOFMS.append(num_denom)
    merged_df[propNoDiffOnForkedMutsStatesCol] = propNDOFMS

    if customMaxtime is not None:
        # filter anything higher than maxtime (minutes)
        assert customMaxtime > 0, "maxtime must be greater than 0"
        merged_df = merged_df[merged_df[timeCol] <= customMaxtime]
        if len(merged_df) == 0:
            print("# The customMaxtime specified is too low. Terminating ...")
            exit(1)

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
    
    only_semu_cfg_df = merged_df[~merged_df[techConfCol].isin(SPECIAL_TECHS)]

    SpecialTechs = dict(SPECIAL_TECHS)
    if no_concrete:
        del SpecialTechs[CONCRETE_KEY]

    vals_by_conf = {}
    for c in config_columns:
        vals_by_conf[c] = list(set(only_semu_cfg_df[c]))
    # add combinations
    for ls_, rs_ in itertools.combinations(vals_by_conf.keys(), 2):
        vals_by_conf[(ls_,rs_)] = list(itertools.product(vals_by_conf[ls_], vals_by_conf[rs_]))

    techConfbyvalbyconf = {}
    for pc in vals_by_conf:
        techConfbyvalbyconf[pc] = {}
        # process param config (get apfds)
        for val in vals_by_conf[pc]:
            if type(val) in (list, tuple):
                keys = set(\
                    only_semu_cfg_df[(only_semu_cfg_df[pc[0]] == val[0]) & \
                            (only_semu_cfg_df[pc[1]] == val[1])][techConfCol])
                val = tuple(val)
            else:
                keys = set(\
                    only_semu_cfg_df[only_semu_cfg_df[pc] == val][techConfCol])
            if len(keys) != 0:
                techConfbyvalbyconf[pc][val] = keys
        if len(techConfbyvalbyconf[pc]) == 0:
            del techConfbyvalbyconf[pc]


    def getListAPFDSForTechConf (t_c):
        v_list = []
        for p in ms_apfds.keys():
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
    #proj_agg_func = np.median
    proj_agg_func = np.average
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
        if type(pc) in (list, tuple):
            plot_out_file = os.path.join(outdir, "perconf_apfd2_"+".".join(pc))
        else:
            plot_out_file = os.path.join(outdir, "perconf_apfd_"+pc)
        data = {str(val): {"min": getListAPFDSForTechConf(min_vals[val]), \
                        "med": getListAPFDSForTechConf(med_vals[val]), \
                        "max": getListAPFDSForTechConf(max_vals[val])} \
                                            for val in techConfbyvalbyconf[pc]}
        emphasis = None
        if len(data) == 2:
            n_projs = len(data[data.keys()[0]]['max'])
            if n_projs >= 2:
                diff_of_projs = {projpos: None for projpos in range(n_projs)}
                for projpos in range(n_projs):
                    diff_of_projs[projpos] = data[data.keys()[0]]['max'][projpos] - data[data.keys()[1]]['max'][projpos]
                avg = np.average([diff_of_projs[projpos] for projpos in diff_of_projs])
                lt_avg = [projpos for projpos in diff_of_projs if diff_of_projs[projpos] < avg]
                gt_avg = [projpos for projpos in diff_of_projs if diff_of_projs[projpos] >= avg]
                emphasis = [{}, {}]
                for val in data:
                    emphasis[0][val] = {}
                    emphasis[1][val] = {}
                    for mmm in data[val]:
                        emphasis[0][val][mmm] = []
                        emphasis[1][val][mmm] = []
                        for v_ind in lt_avg:
                            emphasis[0][val][mmm].append(data[val][mmm][v_ind])
                        for v_ind in gt_avg:
                            emphasis[1][val][mmm].append(data[val][mmm][v_ind])

        for sp in SpecialTechs:
            data[SpecialTechs[sp]] = {em:getListAPFDSForTechConf(sp) \
                                                for em in ['min', 'med','max']}
        tmp_all_vals = []
        for g in data:
            for m in data[g]:
                tmp_all_vals += data[g][m]
        min_y = min(tmp_all_vals)
        max_y = max(tmp_all_vals)
        assert min_y >= 0 and min_y <= 100, "invalid min_y: "+str(min_y)
        assert max_y >= 0 and max_y <= 100, "invalid max_y: "+str(max_y)
        # Actual plot with data 
        # TODO arange max_y, min_y and step_y
        if max_y - min_y >= 10:
            max_y = min(100, int(max_y) + 2) 
            min_y = max(0, int(min_y) - 1)
            step_y = (max_y - min_y) / 10
        else:
            step_y = 1
            rem_tmp = 10 - (max_y - min_y) + 1
            if 100 - max_y < rem_tmp/2:
                min_y = int(min_y - (rem_tmp - (100 - max_y)))
                max_y = 100
            elif min_y < rem_tmp/2:
                max_y = int(max_y + (rem_tmp - min_y)) 
                min_y = 0
            else:
                max_y = int(max_y + rem_tmp/2)
                min_y = int(min_y - rem_tmp/2)
            min_y = max(0, min_y)
            max_y = min(100, max_y)
        yticks_range = range(min_y, max_y+1, step_y)
        plotMerge.plot_Box_Grouped(data, plot_out_file, colors_bw, \
                                "AVERAGE MS (%)", yticks_range=yticks_range, \
                                    selectData=['min', 'med', 'max'])
        
        if emphasis is not None:
            plotMerge.plot_Box_Grouped(emphasis[0], \
                                os.path.join(outdir, "emph_perconf_apfd_"+pc+"_1."+str(len(emphasis[emphasis.keys()[0]]['max']))), \
                                colors_bw, \
                                "AVERAGE MS (%)", yticks_range=yticks_range, \
                                    selectData=['min', 'med', 'max'])
            plotMerge.plot_Box_Grouped(emphasis[1], \
                                os.path.join(outdir, "emph_perconf_apfd_"+pc+"_2."+str(len(emphasis[emphasis.keys()[0]]['max']))), \
                                 colors_bw, \
                                "AVERAGE MS (%)", yticks_range=yticks_range, \
                                    selectData=['min', 'med', 'max'])

    
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
    conf_name_mapping = {
        "_precondLength": "precond_len", 
        "_mutantMaxFork": "max_depth", 
        "_disableStateDiffInTestgen": "no_state_diff", 
        "_maxTestsGenPerMut": "mutant_max_tests", 
        "_postCheckContProba": "continue_prop", 
        "_genTestForDircardedFrom": "disc_gentest_from", 
        "_mutantContStrategy": "continue_strategy", 
    }
    for elem_list, df_obj_list in [(best_elems, best_df_obj), \
                                                (worse_elems, worse_df_obj)]:
        for v in elem_list:
            row = {}
            for pc in techConfbyvalbyconf:
                if type(pc) in (list, tuple):
                    continue
                for val in techConfbyvalbyconf[pc]:
                    if v in techConfbyvalbyconf[pc][val]:
                        assert pc not in row, "BUG"
                        row[conf_name_mapping[pc]] = val
            row[techConfCol] = v
            row['MS_INC_APFD'] = proj_agg_func(getListAPFDSForTechConf(v))
            df_obj_list.append(row)
    best_df = pd.DataFrame(best_df_obj)
    worse_df = pd.DataFrame(worse_df_obj)
    best_df_file = os.path.join(outdir, "best_tech_conf_apfd.csv")
    worse_df_file = os.path.join(outdir, "worse_tech_conf_apfd.csv")
    best_df.to_csv(best_df_file, index=False)
    worse_df.to_csv(worse_df_file, index=False)

    # XXX compare MS with compareState time, %targeted, #testgen, WM%
    if customMaxtime is None:
        assert False, "Must specify a customMaxtime"
        #selectedTimes_minutes = [5, 15, 30, 60, 120]
    else:
        selectedTimes_minutes = [customMaxtime]

    fixed_y = msCol
    changing_ys = [targetCol, covMutsCol, stateCompTimeCol, numGenTestsCol, \
                        propNoDiffOnForkedMutsStatesCol, \
                        numFailTestsCol] # Fail is used for verification purpose
    # get data and plot
    for time_snap in selectedTimes_minutes:
        time_snap_df = merged_df[merged_df[timeCol] == time_snap]
        if time_snap_df.empty:
            continue
        tmp_tech_confs = set(time_snap_df[techConfCol])
        nMuts_here = int(list(time_snap_df[numMutsCol])[0])
        metric2techconf2values = {}
        # for each metric, get per techConf list on values
        for tech_conf in tmp_tech_confs:
            t_c_tmp_df = time_snap_df[time_snap_df[techConfCol] == tech_conf]
            for metric_col in [fixed_y] + changing_ys + [killMutsCol]:
                if metric_col not in metric2techconf2values:
                    metric2techconf2values[metric_col] = {}
                # make sure that every value is a number (to be used in median)
                tmp_vals_list = []
                for v in t_c_tmp_df[metric_col]:
                    try:
                        tmp_vals_list.append(float(v))
                    except ValueError:
                        if v == '-':
                            #continue
                            tmp_vals_list.append(0.0)
                        else:
                            print ("\n# Error: Invalid number for metric_col",\
                                metric_col, ". value is:", v)
                            print ("# ... Tech_conf is:", tech_conf, '\n')
                            assert False
                metric2techconf2values[metric_col][tech_conf] = tmp_vals_list
        
        if len(metric2techconf2values) == 0:
            print ("#WARNING: metric2techconf2values is empty!")
        else:
            sorted_techconf_by_ms = metric2techconf2values[fixed_y].keys()
            sorted_techconf_by_ms.sort(reverse=True, key=lambda x: ( \
                            np.median(metric2techconf2values[fixed_y][x]), \
                            np.average(metric2techconf2values[fixed_y][x])))

            # Compute MS APFD VS MS and compute the fixed_vals
            ms_inc_apfd_y = "Average MS-INC"
            ms_inc_apfd_vals = []
            fix_vals = []
            for tech_conf in sorted_techconf_by_ms:
                if tech_conf in SPECIAL_TECHS:
                    continue
                fix_vals.append(metric2techconf2values[fixed_y][tech_conf])
                ms_inc_apfd_vals.append([ms_apfds[p][tech_conf] for p in ms_apfds])
            plot_img_out_file = os.path.join(outdir, "msapfdVSms-"+ \
                                                                str(time_snap))
            make_twoside_plot(fix_vals, ms_inc_apfd_vals, plot_img_out_file, \
                            x_label="Configuations", y_left_label=fixed_y, \
                                                y_right_label=ms_inc_apfd_y)

            # Make plots of ms and the others
            for chang_y in changing_ys:
                plot_img_out_file = os.path.join(outdir, "otherVSms-"+ \
                                str(time_snap) + "-"+chang_y.replace('#','n'))
                chang_vals = []
                for tech_conf in sorted_techconf_by_ms:
                    if tech_conf in SPECIAL_TECHS:
                        continue
                    chang_vals.append(\
                                    metric2techconf2values[chang_y][tech_conf])

                if chang_y in [targetCol, stateCompTimeCol, covMutsCol]:
                    if chang_y in (targetCol, covMutsCol):
                        chang_y = chang_y.replace('#', '%')
                        m_val = nMuts_here
                        assert m_val != 0, "Max X muts is 0. (PB)"
                    elif chang_y == stateCompTimeCol:
                        chang_y = chang_y.replace('(s)', '(%)')
                        m_val = time_snap * 60
                    else:
                        assert False, "BUG, unreachable"
                    # compute percentage
                    if m_val != 0:
                        for c_ind in range(len(chang_vals)):
                            chang_vals[c_ind] = \
                                    [v*100.0/m_val for v in chang_vals[c_ind]]
                        
                make_twoside_plot(fix_vals, chang_vals, plot_img_out_file, \
                            x_label="Configuations", y_left_label=fixed_y, \
                                                        y_right_label=chang_y)
            

        # XXX Killed mutants overlap
        overlap_data_dict = {}
        non_overlap_obj = {}
        for proj in proj2dir:
            fulldir = os.path.join(proj2dir[proj], projcommonreldir)
            full_overlap_file = os.path.join(fulldir, \
                        "Techs-relation.json-"+str(int(time_snap))+"min.json")
            assert os.path.isfile(full_overlap_file), "file not existing: "+\
                                full_overlap_file
            with open(full_overlap_file) as f:
                fobj = json.load(f)
                non_overlap_obj[proj] = {}
                overlap_data_dict[proj] = {}
                visited = set()
                for pair in fobj["NON_OVERLAP_VENN"]:
                    a_tmp = pair.split('&')
                    if len(a_tmp) != 2:
                        print("@WARNING: non pair overlap found:", pair)
                        continue
                    left_right = tuple(sorted(a_tmp))
                    if left_right not in visited:
                        visited.add(left_right)
                        overlap_data_dict[proj][left_right] = fobj["OVERLAP_VENN"][pair]
                        
                    if left_right not in non_overlap_obj[proj]:
                        non_overlap_obj[proj][left_right] = {v:0 for v in left_right}
                    for win in fobj["NON_OVERLAP_VENN"][pair]:
                        non_overlap_obj[proj][left_right][win] += \
                                    len(fobj["NON_OVERLAP_VENN"][pair][win])
        all_tech_confs = {v for p in non_overlap_obj[non_overlap_obj.keys()[0]] for v in p}
        assert all_tech_confs == set(sorted_techconf_by_ms), \
                            "Inconsistemcy between dataframe and overlap json"
        tech_conf2position = {}
        for pos, tech_conf in enumerate(sorted_techconf_by_ms):
            tech_conf2position[tech_conf] = pos

        x_label = "Winning Technique Configuration"
        y_label = "Other Technique Configuration"
        hue = "special"
        num_x_wins = "# Mutants Killed more by X"

        # Plot klee conf overlap by proj
        klee_n_semu_by_proj = [[], []]
        by_proj_overlap = []
        for proj in non_overlap_obj:
            klee_n_semu_by_proj[0].append(0)
            klee_n_semu_by_proj[1].append(0)
            by_proj_overlap.append(0)
            for left_right in non_overlap_obj[proj]:
                if KLEE_KEY in left_right:
                    s_c_n_o = non_overlap_obj[proj][left_right][list(set(left_right)-{KLEE_KEY})[0]]
                    k_n_o = non_overlap_obj[proj][left_right][KLEE_KEY]
                    if  s_c_n_o - k_n_o > klee_n_semu_by_proj[0][-1] - klee_n_semu_by_proj[1][-1]:
                        klee_n_semu_by_proj[0][-1] = s_c_n_o
                        klee_n_semu_by_proj[1][-1] = k_n_o
                        by_proj_overlap[-1] = overlap_data_dict[proj][left_right]
            if klee_n_semu_by_proj[1] > klee_n_semu_by_proj[0]:
                print(">>>> Klee has higher non overlap that all semu for project", proj, "(", klee_n_semu_by_proj[1], "VS", klee_n_semu_by_proj[0], ")")
        ## plot
        make_twoside_plot(klee_n_semu_by_proj+[by_proj_overlap], klee_n_semu_by_proj, \
                    os.path.join(outdir, "proj_overlap-"+str(time_snap)+"min"), \
                    x_label="Configuations", y_left_label="# Mutants", \
                                    y_right_label="# Non Overlapping Mutants", \
                                left_stackbar_legends=['semu', 'klee', 'overlap'], \
                                right_stackbar_legends=['semu', 'klee'])

                        
        #proj_agg_func2 = np.average
        proj_agg_func2 = np.median
        df_obj = []

        for left_right in non_overlap_obj[non_overlap_obj.keys()[0]]:
            for left, right in [left_right, reversed(left_right)]:
                hue_val = "SEMu"
                if left in SpecialTechs and right in SpecialTechs:
                    hue_val = SpecialTechs[left]+"_wins-"\
                                                +SpecialTechs[right]+"_loses"
                else:
                    if right in SpecialTechs:
                        hue_val = SpecialTechs[right]+"_loses"
                    if left in SpecialTechs:
                        hue_val = SpecialTechs[left]+"_wins"
                df_obj.append({
                        x_label: tech_conf2position[left], 
                        y_label: tech_conf2position[right], 
                        hue: hue_val,
                        num_x_wins: proj_agg_func2([non_overlap_obj[p][left_right][left] for p in non_overlap_obj]), 
                        })
        killed_muts_overlap = pd.DataFrame(df_obj)
        image_out = os.path.join(outdir, "overlap-"+str(time_snap)+"min")
        # plot
        sns.set_style("white", \
                            {'axes.linewidth': 1.25, 'axes.edgecolor':'black'})
        sns.relplot(x=x_label, y=y_label, hue=hue, size=num_x_wins,
                sizes=(0, 300), alpha=.5, palette="muted",
                height=6, data=killed_muts_overlap)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(image_out+".pdf", format="pdf")
        plt.close('all')

        # plot overlap against pure klee
        image_out2 = os.path.join(outdir, \
                                "semu_klee-nonoverlap-"+str(time_snap)+"min")
        chang_y2 = "# Non Overlapping Mutants"
        fix_vals2 = []
        sb_legend = ['semu', 'klee']
        chang_vals2 = [[], []]
        overlap_vals = []
        klee = SPECIAL_TECHS[KLEE_KEY]
        klee_related_df = killed_muts_overlap[killed_muts_overlap[hue].isin(\
                                                [klee+'_wins', klee+'_loses'])]
        assert not klee_related_df.empty
        for tech_conf in sorted_techconf_by_ms:
            if tech_conf in SPECIAL_TECHS:
                continue
            fix_vals2.append(metric2techconf2values[fixed_y][tech_conf])

            tmp_v = list(klee_related_df[klee_related_df[x_label] == \
                                tech_conf2position[tech_conf]][num_x_wins])
            assert len(tmp_v) != 0
            chang_vals2[0].append(tmp_v[0])
            left_right = tuple(sorted([tech_conf, KLEE_KEY]))
            overlap_vals.append(proj_agg_func2([overlap_data_dict[p][left_right] for p in overlap_data_dict.keys()]))

            tmp_v = list(klee_related_df[klee_related_df[y_label] == \
                                tech_conf2position[tech_conf]][num_x_wins])
            assert len(tmp_v) != 0
            chang_vals2[1].append(tmp_v[0])

        make_twoside_plot(fix_vals2, chang_vals2, image_out2, \
                    x_label="Configuations", y_left_label=fixed_y, \
                                                y_right_label=chang_y2, \
                                    right_stackbar_legends=sb_legend)
        # overlap and non overlap
        image_out3 = os.path.join(outdir, \
                                "semu_klee-overlap_all-"+str(time_snap)+"min")
        overlap_non_vals = chang_vals2 + [overlap_vals] 
        make_twoside_plot(overlap_non_vals, chang_vals2, image_out3, \
                    x_label="Configuations", y_left_label="# Mutants", \
                                                y_right_label=chang_y2, \
                                left_stackbar_legends=sb_legend+['overlap'], \
                                right_stackbar_legends=sb_legend)


#~ def libMain()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=None, \
            help="Output directory, will be deleted and recreated if exists")
    parser.add_argument("-i", "--intopdir", default=None, \
            help="Top directory where to all projects are"\
                                        +" (will search the finished ones)")
    parser.add_argument("--usefunctions", action='store_true', \
                help="Enable using by function instead of just by project")
    parser.add_argument("--maxtimes", default=None, \
                help="space separated customMaxtime list to use (in minutes)")
    parser.add_argument("--onlyprojects", default=None, \
                help="space separated project list to use")
    parser.add_argument('--forposter', action='store_true', \
                                                    help="set poster context")
    parser.add_argument('--onlykillable', action='store_true', \
                    help="Only for project/function with a killable mutant")
    parser.add_argument('--with_concrete', action='store_true', \
                    help="compare normal semu only with pure klee")
    args = parser.parse_args()

    if args.forposter:
        sns.set_context("poster")
    else:
        sns.set_context("paper")

    outdir = args.output
    intopdir = args.intopdir
    assert outdir is not None
    assert intopdir is not None
    assert os.path.isdir(intopdir)

    maxtime_list = None
    if args.maxtimes is not None:
        maxtime_list = list(set(args.maxtimes.strip().split()))
    
    onlyprojects_list = None
    if args.onlyprojects is not None:
        onlyprojects_list = list(set(args.onlyprojects.strip().split()))

    if os.path.isdir(outdir):
        if raw_input("\nspecified output exists. Clear it? [y/n] ")\
                                                    .lower().strip() == 'y':
            shutil.rmtree(outdir)
        else:
            print("# please specify another outdir")
            return
    os.mkdir(outdir)
    proj2dir = {}
    for f_d in os.listdir(intopdir):
        direct = os.path.join(intopdir, f_d, getProjRelDir())
        if os.path.isfile(os.path.join(direct, csv_file)) and \
                        os.path.isfile(os.path.join(direct, funcs_csv_file)):
            proj2dir[f_d] = os.path.join(intopdir, f_d)
    if onlyprojects_list is not None:
        for p in set(proj2dir) - set(onlyprojects_list):
            if p in proj2dir:
                del proj2dir[p]
    if len(proj2dir) > 0:
        print ("# Calling libMain on projects", list(proj2dir), "...")
        if maxtime_list is None:
            libMain(outdir, proj2dir, use_func=args.usefunctions, \
                                            onlykillable=args.onlykillable, \
                                        no_concrete=(not args.with_concrete))
        else:
            for maxtime in maxtime_list:
                mt_outdir = os.path.join(outdir, "maxtime-"+maxtime)
                os.mkdir(mt_outdir)
                libMain(mt_outdir, proj2dir, use_func=args.usefunctions, \
                                            customMaxtime=float(maxtime), \
                                            onlykillable=args.onlykillable, \
                                        no_concrete=(not args.with_concrete))

        print("# DONE")
    else:
        print("# !! No good project found")

if __name__ == '__main__':
    main()