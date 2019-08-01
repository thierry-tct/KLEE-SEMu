
# Library to compute the stats after run.py
# Example
## python ~/mytools/klee-semu/src/lib/Mutation/analysis/stats_for_run.py -i SEMU_EXECUTION -o RESULTS --maxtimes "5 15 30 60 120"
#
# TODO:
# 1. Minimal test suite that improve MS and MS*
# 2. Fix intersection in minimal
# a) Highlight on the slide what is our contribution
"""

python ~/mytools/klee-semu/src/lib/Mutation/analysis/stats_for_run.py -i SEMU_EXECUTION -o RESULTS --maxtimes "120" \
&& python ~/mytools/klee-semu/src/lib/Mutation/analysis/stats_for_run.py -i SEMU_EXECUTION -o RESULTS-KILLABLE --maxtimes "120" --onlykillable \
&& python ~/mytools/klee-semu/src/lib/Mutation/analysis/stats_for_run.py -i SEMU_EXECUTION -o RESULTS_FUNCS --maxtimes "120" --usefunctions \
&& python ~/mytools/klee-semu/src/lib/Mutation/analysis/stats_for_run.py -i SEMU_EXECUTION -o RESULTS_FUNCS-KILLABLE --maxtimes "120" --usefunctions --onlykillable

"""


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
import copy

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

def dumpJson (obj, filename, pretty=True):
    with open(filename, "w") as fp:
        if pretty:
            json.dump(obj, fp, indent=2, sort_keys=True)
        else:
            json.dump(obj, fp)
#~ dumpJson()
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

    if len(x_list) == 1:
        auc = y_list[0]
    else:
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


def make_twoside_plot(left_y_vals, right_y_vals, x_vals=None, img_out_file=None, \
                                    x_label=None, y_left_label="Y_LEFT", \
                                    y_right_label="Y_RIGHT", separate=True,\
                                    left_stackbar_legends=None,\
                                    right_stackbar_legends=None, show_grid=True,
                                    left_color_list=None, right_color_list=None):

    fontsize = 16
    if left_y_vals is None or right_y_vals is None:
        separate = True
        fig, ax = plt.subplots(figsize=(13,8))
        if left_y_vals is not None:
            ax1 = ax
        elif right_y_vals is not None:
            ax2 = ax
        else:
            assert False, "Must specify one at least"
    else:
        if separate:
            fig=plt.figure()
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212, sharex = ax1)
        else:
            fig, ax1 = plt.subplots()

    if x_label is not None:
        ax1.set_xlabel(x_label, fontsize=fontsize)

    flierprops = dict(marker='o', markersize=2, linestyle='none')

    if left_y_vals is not None:
        color = 'tab:blue' if right_y_vals is not None else (0.0,0.0,0.0)

        if separate:
            ax1.set_ylabel(y_left_label, fontsize=fontsize)
        else:
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
                if left_color_list is None:
                    p[i] = ax1.bar(ind, left_y_vals[i], bottom=bottoms[i])
                else:
                    if type(left_color_list[i]) == str:
                        p[i] = ax1.bar(ind, left_y_vals[i], bottom=bottoms[i], hatch=left_color_list[i])
                    else:
                        p[i] = ax1.bar(ind, left_y_vals[i], bottom=bottoms[i], color=left_color_list[i])
            ax1.legend(p, left_stackbar_legends, fontsize=fontsize)
            plt.xlim([0,ind.size])

    if not separate:
        # instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()  

    if right_y_vals is not None:
        color = 'tab:red' if left_y_vals is not None else  (0.0,0.0,0.0)

        if separate:
            ax2.set_ylabel(y_right_label, fontsize=fontsize)  # we already handled the x-label with ax1
        else:
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
                if right_color_list is None:
                    p[i] = ax2.bar(ind, right_y_vals[i], bottom=bottoms[i])
                else:
                    if type(right_color_list[i]) == str:
                        p[i] = ax2.bar(ind, right_y_vals[i], bottom=bottoms[i], hatch=right_color_list[i])
                    else:
                        p[i] = ax2.bar(ind, right_y_vals[i], bottom=bottoms[i], color=right_color_list[i])
            ax2.legend(p, right_stackbar_legends, fontsize=fontsize)
            #ax.margins(0.05)
            plt.xlim([0,ind.size])

    if x_vals is None:
        plt.xticks([])
    else:
        #locs, labels = plt.xticks()
        #assert len(locs) == len(x_vals), "labels mismatch: {} VS {}.".format(len(locs), len(x_vals))
        #print("labels mismatch: {} VS {}.".format(len(locs), len(x_vals)))
        plt.xticks(np.arange(len(x_vals)), x_vals, rotation=45, ha='right', fontsize=fontsize-8)
    
    if not show_grid:
        plt.rcParams["axes.grid"] = False

    plt.tight_layout()
    plt.axis('tight')
    plt.autoscale()

    if img_out_file is None:
        plt.show()
    else:
        plt.savefig(img_out_file+".pdf", format="pdf")
    plt.close('all')
#~ def make_twoside_plot()

def plotLines(x_y_lists_pair_dict, order, xlabel, ylabel, imagepath, colors, linestyles, linewidths, fontsize):
    # get median
    plt.figure(figsize=(13, 9))
    plt.gcf().subplots_adjust(bottom=0.27)
    #plt.style.use(u'ggplot')
    #sns.set_style("ticks")
    sns.set_style("whitegrid")
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 1.25
    #sns.set_context("talk")
    fontsize = 26
    if order is None:
        order = x_y_lists_pair_dict.keys()
    #maxx = max([max(x_y_lists_pair_dict[t][0]) for t in order])
    maxx = max([x_y_lists_pair_dict[t][0] for t in order])
    for ti,tech in enumerate(order):
        x, y = x_y_lists_pair_dict[tech]
        plt.plot(x, y, color=colors[ti], linestyle=linestyles[ti], linewidth=linewidths[ti], label=tech, alpha=0.8)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks((range(1, int(maxx+1), int(maxx/10)) if (maxx % 10 == 0 or type(maxx) == int) else np.arange(1,maxx+1, maxx/10.0)), fontsize=fontsize-5)
    plt.yticks(np.arange(0,1.01,0.2), fontsize=fontsize-5)
    legendMode=1 if len(order) <= 3 else 2
    if legendMode==1:
        lgd = plt.legend(bbox_to_anchor=(0., 0.98, 1., .102), loc=2, ncol=3, mode="expand", fontsize=fontsize, borderaxespad=0.)
    elif legendMode==2:
        lgd = plt.legend(bbox_to_anchor=(0., 0.98, 1.02, .152), loc=2, ncol=3, mode="expand", fontsize=fontsize, borderaxespad=0.)
    else:
        assert False, "invalid legend mode (expect either 1 or 2)"
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=3)
    #sns_plot.set_title('APFD - '+allkonly)
    plt.tight_layout()
    ybot, ytop = plt.gca().get_ylim()
    ypad = (ytop - ybot) / 50
    #ypad = 2
    plt.gca().set_ylim(ybot - ypad, ytop + ypad)
    plt.savefig(imagepath+".pdf", format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close('all')
#~ def plotLines()

def get_minimal_conf_set(tech_conf_missed_muts, get_all=True):
    """ the input has this format: 
        {project: {tc: set()<missed muts>}}
        :param get_all: decides whether to get all the minimal sets of a single one
    """
    flatten_tc_missed_muts = {}
    proj_list = []
    for proj, tc2mutset in list(tech_conf_missed_muts.items()):
        proj_list.append(proj)
        for tc, mutset in list(tc2mutset.items()):
            if tc not in flatten_tc_missed_muts:
                flatten_tc_missed_muts[tc] = set()
            flatten_tc_missed_muts[tc] |= set(["#".join([proj, str(m)]) for m in mutset])

    visited = set()
    clusters_list = []
    for tc in flatten_tc_missed_muts:
        if tc in visited:
            continue
        visited.add(tc)
        cluster = [tc]
        for otc in flatten_tc_missed_muts:
            if otc in visited:
                continue
            if flatten_tc_missed_muts[tc] == flatten_tc_missed_muts[otc]:
                cluster.append(otc)
                visited.add(otc)
            elif len(flatten_tc_missed_muts[tc] - flatten_tc_missed_muts[otc]) == 0:
                visited.add(otc)
            elif len(flatten_tc_missed_muts[otc] - flatten_tc_missed_muts[tc]) == 0:
                cluster = None
                break
        if cluster is not None:
            clusters_list.append(cluster)
    tmp = clusters_list
    clusters_list = []

    # case where klee killed but semu couldn't
    all_inter = set()
    if len(tmp) > 0:
        all_inter = set(flatten_tc_missed_muts[tmp[0][0]])
        for c in tmp[1:]:
            all_inter &= flatten_tc_missed_muts[c[0]] 
        if len(all_inter) > 0:
            print("#> Klee or concrete, managed to kill {} extra mutants".format(len(all_inter)))
    
    def greedy_eval(in_mutset):
        use_median = True
        #use_median = False
        if use_median:
            nmuts_by_proj = {proj:0 for proj in proj_list}
            for m in in_mutset:
                proj, raw_m = m.split('#')
                if proj not in nmuts_by_proj:
                    nmuts_by_proj[proj] = 0
                nmuts_by_proj[proj] += 1
            res = np.median([v for _,v in list(nmuts_by_proj.items())])
            res = (res, len(in_mutset))
        else:
            res = len(in_mutset)
        return res
    #~ def greedy_eval()

    # Use greedy Algorithm to find the smallest combination
    selected_pos = []
    min_pos = 0
    min_size = greedy_eval(flatten_tc_missed_muts[tmp[0][0]])
    for i in range(1,len(tmp)):
        attempt_size = greedy_eval(flatten_tc_missed_muts[tmp[i][0]])
        if attempt_size < min_size:
            min_size = attempt_size
            min_pos = i
    selected_pos.append(min_pos)
    sel_missed = set(flatten_tc_missed_muts[tmp[min_pos][0]])
    while sel_missed != all_inter:
        min_size = greedy_eval(sel_missed)
        min_pos = None
        for i in range(len(tmp)):
            if i in selected_pos:
                continue
            attempt_size = greedy_eval(flatten_tc_missed_muts[tmp[i][0]] & sel_missed)
            if attempt_size < min_size:
                min_size = attempt_size
                min_pos = i
        assert min_pos is not None, "Bug: Should have stopped the while loop. "+str(sel_missed)+str(all_inter)
        selected_pos.append(min_pos)
        sel_missed &= flatten_tc_missed_muts[tmp[min_pos][0]] 
        
    if not get_all:
        clusters_list.append([tmp[i] for i in selected_pos])
    else:
        for ncomb in [len(selected_pos)]: #range(1,len(tmp)):
            if len(clusters_list) > 0:
                break
            #print("# dbg: in loop 2", tmp, ncomb)
            print ("# ncomb =", ncomb, "; num cluster for comb is", len(list(itertools.combinations(tmp, ncomb))))
            for tc_comb in itertools.combinations(tmp, ncomb):
                intersect = set(flatten_tc_missed_muts[tc_comb[0][0]])
                for tc_list in tc_comb[1:]:
                    tc = tc_list[0]
                    intersect &= flatten_tc_missed_muts[tc]
                    if intersect == all_inter:
                        clusters_list.append(list(tc_comb))
            if len(clusters_list) == 0:
                #assert False, "Must have something"
                print ('> Warning: Project do not have minimal cluster: '+str(proj))

    return clusters_list, len(all_inter)
#~ def get_minimal_conf_set()

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

# Columns ####
timeCol = "TimeSnapshot(min)"
config_columns = ["_precondLength","_mutantMaxFork", 
                    "_genTestForDircardedFrom", "_postCheckContProba", 
                    "_mutantContStrategy", "_maxTestsGenPerMut", 
                    "_disableStateDiffInTestgen"
                ]
cnm_accronym = True
conf_name_mapping = {
    "_precondLength": 'PL' if cnm_accronym else "precond_len", 
    "_mutantMaxFork": 'CW' if cnm_accronym else "max_depth", 
    "_disableStateDiffInTestgen": 'NSD' if cnm_accronym else "no_state_diff", 
    "_maxTestsGenPerMut": 'NTPM' if cnm_accronym else "mutant_max_tests", 
    "_postCheckContProba": 'PP' if cnm_accronym else "continue_prop", 
    "_genTestForDircardedFrom": 'MPD' if cnm_accronym else "disc_gentest_from", 
    "_mutantContStrategy": 'PSS' if cnm_accronym else "continue_strategy", 
}
other_cols = ["_testGenOnlyCriticalDiffs" ]

numFailTestsCol = "#FailingTests"
techConfCol = "Tech-Config"
stateCompTimeCol = "StateComparisonTime(s)"
numGenTestsCol = "#GenTests"
numForkedMutStatesCol = "#MutStatesForkedFromOriginal"
mutPointNoDifCol = "#MutStatesEqWithOrigAtMutPoint"

propNoDiffOnForkedMutsStatesCol = "percentageNodiffFoundAtMutPoint"
######~

## PLot const
colors_bw = ['white', 'whitesmoke', 'lightgray', 'silver', 'darkgrey', \
                                                'gray', 'dimgrey', "black"]
colors = ["green", 'blue', 'red', "black", "maroon", "magenta", "cyan"]
linestyles = ['solid', 'solid', 'dashed', 'dashed', 'dashdot', 'dotted', \
                                                                'solid']
linewidths = [1.75, 1.75, 2.5, 2.5, 3.25, 3.75, 2]
fontsize = 26

colors_bw += colors_bw*3
colors += colors*3
linestyles += linestyles*3
linewidths += linewidths*3
#####~

# Others
semuBEST = 'semu-best'
infectOnly = 'no-propagation'
######~

goodViewColors = {
    semuBEST: '//', #(0.0, 0.0, 1.0, 0.6), #'blue',
    infectOnly: '\\', #(0.0, 0.39, 0.0, 0.6), #'green',
    'klee': '.', #(0.6, 0.3, 0.0, 0.6), #'maron',
    'overlap': (0.2, 0.2, 0.2, 0.6), #'grey',
    'missed': (1.0, 1.0, 1.0, 0.6), #(1.0, 0.0, 0.0, 0.6), #'red',
    'initial': (0.0, 0.0, 0.0, 0.6), #'black',
}


def loadData(proj2dir, use_func=False, projcommonreldir=None, \
                                                        onlykillable=False):
    merged_df = None
    all_initial = {}

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

    merged_df['_precondLength'].replace(-1,'MD2MS',inplace=True)
    merged_df['_precondLength'].replace(-2,'MD2MSC',inplace=True)

    return merged_df, all_initial
#~ loadData()

def merge_initial(all_initial, outdir, use_func, has_subsuming_data=True):
    # Compute the merged json
    merged_json_obj = {}
    merged_json_obj["Projects"] = list(all_initial) 
    merged_json_obj["#Projects"] = len(all_initial) 
    if has_subsuming_data:
        merged_json_obj["Initial#SubsumingMutants"] = \
                sum([int(all_initial[v]["Initial#SubsumingMutants"]) \
                                                        for v in all_initial])
        merged_json_obj["Initial#SubsumingKilledMutants"] = \
                sum([int(all_initial[v]["Initial#SubsumingKilledMutants"]) \
                                                        for v in all_initial])
        merged_json_obj["Initial-MS_Subsuming"] = \
                sum([float(all_initial[v]["Initial-MS_Subsuming"]) \
                                    for v in all_initial]) / len(all_initial)

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
    return merged_json_obj
#~ def merge_initial()

def preprocessing_merge_df(loaded_merged_df, customMaxtime, no_concrete):
    assert propNoDiffOnForkedMutsStatesCol not in loaded_merged_df, \
                                                    "Use different key (BUG)" 
    merged_df = loaded_merged_df.copy(deep=True)
    SpecialTechs = dict(SPECIAL_TECHS)
    if no_concrete:
        del SpecialTechs[CONCRETE_KEY]
        merged_df = merged_df[merged_df[techConfCol] != CONCRETE_KEY]

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

    return merged_df, SpecialTechs
#~ def preprocessing_merge_df()

def get_ms_apfds(merged_df, msCol):
    tech_confs = set(merged_df[techConfCol])
    projects = set(merged_df[PROJECT_ID_COL])
    ms_by_time = {t_c: {p:None for p in projects} for t_c in tech_confs}
    ms_apfds = {p: {t_c: None for t_c in tech_confs} for p in projects}
    for p in ms_apfds:
        p_tmp_df = merged_df[merged_df[PROJECT_ID_COL] == p]
        for t_c in ms_apfds[p]:
            # get the data
            tmp_df = p_tmp_df[p_tmp_df[techConfCol] == t_c]
            ms_by_time[t_c][p] = [list(tmp_df[timeCol]), list(tmp_df[msCol])]
            ms_apfds[p][t_c] = compute_apfd(tmp_df[timeCol], tmp_df[msCol])
        tmp_df = p_tmp_df = None

    return ms_apfds, ms_by_time
#~ def get_ms_apfds()

def getListAPFDSForTechConf (t_c, ms_apfds):
    v_list = []
    for p in ms_apfds.keys():
        assert t_c in ms_apfds[p]
        v_list.append(ms_apfds[p][t_c])
    return v_list
#~ def getListAPFDSForTechConf ()
        
def get_techConfUtils(only_semu_cfg_df, SpecialTechs):
    vals_by_conf = {}
    for c in config_columns:
        vals_by_conf[c] = list(set(only_semu_cfg_df[c]))
    # add combinations
    for ls_, rs_ in itertools.combinations(vals_by_conf.keys(), 2):
        vals_by_conf[(ls_,rs_)] = list(itertools.product(vals_by_conf[ls_], vals_by_conf[rs_]))

    techConf2ParamVals = {k: {conf_name_mapping[cc]: None for cc in config_columns} for k in SpecialTechs}
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
                for k in keys:
                    if k not in techConf2ParamVals:
                        techConf2ParamVals[k] = {}
                    assert conf_name_mapping[pc] not in techConf2ParamVals[k], "each param once per conf"
                    techConf2ParamVals[k][conf_name_mapping[pc]] = val
            if len(keys) != 0:
                techConfbyvalbyconf[pc][val] = keys
        if len(techConfbyvalbyconf[pc]) == 0:
            del techConfbyvalbyconf[pc]

    return techConf2ParamVals, techConfbyvalbyconf
#~ def get_techConfUtils()

def compute_n_plot_param_influence(techConfbyvalbyconf, outdir, SpecialTechs, \
                                    n_suff, ms_apfds, proj_agg_func=None, \
                                    bests_only=False, use_fixed=False, special_on_per_param=False,\
                                    specific_cmp_best_only=True):
    y_repr = "" if use_fixed else "AVERAGE " # Over time Average

    for pc in techConfbyvalbyconf:
        for_sota = False
        min_vals = {}
        max_vals = {}
        med_vals = {}
        overal_best = None
        overal_best_score = 0
        for val in techConfbyvalbyconf[pc]:
            sorted_by_apfd_tmp = sorted(techConfbyvalbyconf[pc][val], \
                    key=lambda x: proj_agg_func(getListAPFDSForTechConf(x, ms_apfds)))
            min_vals[val] = sorted_by_apfd_tmp[0]
            max_vals[val] = sorted_by_apfd_tmp[-1]
            med_vals[val] = sorted_by_apfd_tmp[len(sorted_by_apfd_tmp)/2]
            if overal_best is None or proj_agg_func(getListAPFDSForTechConf(max_vals[val], ms_apfds)) > overal_best_score:
                overal_best = str(val)
                overal_best_score = proj_agg_func(getListAPFDSForTechConf(max_vals[val], ms_apfds))

        # plot
        if type(pc) in (list, tuple):
            plot_out_file = os.path.join(outdir, "perconf_apfd2_"+".".join(pc))
            sota_pc = {"_postCheckContProba", "_mutantMaxFork"}
            if set(pc) == sota_pc:
                for_sota = True
        else:
            plot_out_file = os.path.join(outdir, "perconf_apfd_"+pc)
        data = {str(val): {"min": getListAPFDSForTechConf(min_vals[val], ms_apfds), \
                        "med": getListAPFDSForTechConf(med_vals[val], ms_apfds), \
                        "max": getListAPFDSForTechConf(max_vals[val], ms_apfds)} \
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

        if special_on_per_param:
            for sp in SpecialTechs:
                data[SpecialTechs[sp]] = {em:getListAPFDSForTechConf(sp, ms_apfds) \
                                                for em in ['min', 'med','max']}

        # BEST VS SOTA VS KLEE
        info_best_sota_klee = None
        if for_sota:
            sota_val = "('0', '0.0')"
            if sota_val in data:
                best_sota_klee_data = {}
                for sp in SpecialTechs:
                    best_sota_klee_data[SpecialTechs[sp]] = {em:getListAPFDSForTechConf(sp, ms_apfds) \
                                                    for em in ['min', 'med','max']}
                best_sota_klee_data[infectOnly] = copy.deepcopy(data[sota_val])
                best_sota_klee_data[semuBEST] = copy.deepcopy(data[overal_best])

                outfile_best_sota_klee = os.path.join(outdir, "bestVSsotaVSklee")
                
                info_best_sota_klee = {}
                for lev, vals in [('max', max_vals), ('med', med_vals), ('min', min_vals)]:
                    info_best_sota_klee[lev] = {}
                    for sp, sp_name in list(SpecialTechs.items()):
                        info_best_sota_klee[lev][sp_name] = {techConfCol: sp,
                                                    'score': proj_agg_func(best_sota_klee_data[sp_name][lev])}
                    for v in vals:
                        v_str = str(v)
                        if v_str == sota_val:
                            info_best_sota_klee[lev][infectOnly] = {
                                                    techConfCol: vals[v], 
                                                    'score': proj_agg_func(best_sota_klee_data[infectOnly][lev])}
                        elif v_str == overal_best:
                            info_best_sota_klee[lev][semuBEST] = {
                                                    techConfCol: vals[v], 
                                                    'score': proj_agg_func(best_sota_klee_data[semuBEST][lev])}
                dumpJson(info_best_sota_klee, outfile_best_sota_klee+'.info.json')

                bsk_sel_dat = ['min', 'med', 'max']
                if specific_cmp_best_only:
                    bsk_sel_dat = ['']
                    for tc in best_sota_klee_data:
                        del best_sota_klee_data[tc]['med']
                        del best_sota_klee_data[tc]['min']
                        best_sota_klee_data[tc][''] = best_sota_klee_data[tc]['max']
                        del best_sota_klee_data[tc]['max']
                
                # PLot
                inner_stattest(best_sota_klee_data, outfile_best_sota_klee+'--statest.json')
                median_vals = plotMerge.plot_Box_Grouped(best_sota_klee_data, outfile_best_sota_klee, colors_bw, \
                                        y_repr+"MS"+n_suff+" (%)", yticks_range=get_yticks_range(best_sota_klee_data), \
                                            selectData=bsk_sel_dat)
                dumpJson(median_vals, outfile_best_sota_klee+'.medians.json')

        selected_data = ['min', 'med', 'max']
        if bests_only:
            selected_data = ['max']
            for tc in data:
                del data[tc]['med']
                del data[tc]['min']

        def get_yticks_range(in_data):
            tmp_all_vals = []
            for g in in_data:
                for m in in_data[g]:
                    tmp_all_vals += in_data[g][m]
            min_y = min(tmp_all_vals)
            max_y = max(tmp_all_vals)
            assert min_y >= 0 and min_y <= 100, "invalid min_y: "+str(min_y)
            assert max_y >= 0 and max_y <= 100, "invalid max_y: "+str(max_y)

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
            return yticks_range
        #~ def get_yticks_range()

        # Actual plot with data 
        yticks_range = get_yticks_range(data)

        # stat test
        def inner_stattest(in_data, filename):
            statstest_obj = {}
            for pos1,g1 in enumerate(in_data):
                for pos2, g2 in enumerate(in_data):
                    if pos1 >= pos2:
                        continue
                    tmp_stats = {v:{} for v in list(in_data[g1])}
                    for k in tmp_stats:
                        tmp_stats[k]['p_value'] = wilcoxon(in_data[g1][k], in_data[g2][k], isranksum=False)
                        tmp_stats[k]['A12'] = a12(in_data[g1][k], in_data[g2][k], pairwise=True)
                    statstest_obj[str((g1, g2))] = tmp_stats
            dumpJson(statstest_obj, filename)
        #~ def inner_stattest()

        # plot
        inner_stattest(data, plot_out_file+'--statest.json')
        median_vals = plotMerge.plot_Box_Grouped(data, plot_out_file, colors_bw, \
                                y_repr+"MS"+n_suff+" (%)", yticks_range=yticks_range, \
                                    selectData=selected_data)
        dumpJson(median_vals, plot_out_file+'.medians.json')

        # if case it is having state-of-the art's similar config, plot BEST VS SOTA(zero-propagation) VS KLEE
        
        if emphasis is not None:
            emph1_plot_out_file = os.path.join(outdir, "emph_perconf_apfd_"+pc+"_1."+str(len(emphasis[0][emphasis[0].keys()[0]]['max'])))
            inner_stattest(emphasis[0], emph1_plot_out_file+'--statest.json')
            median_vals = plotMerge.plot_Box_Grouped(emphasis[0], \
                                emph1_plot_out_file, \
                                colors_bw, \
                                y_repr+"MS"+n_suff+" (%)", yticks_range=yticks_range, \
                                    selectData=selected_data)
            dumpJson(median_vals, emph1_plot_out_file+'.medians.json')

            emph2_plot_out_file = os.path.join(outdir, "emph_perconf_apfd_"+pc+"_2."+str(len(emphasis[1][emphasis[1].keys()[0]]['max'])))
            inner_stattest(emphasis[1], emph2_plot_out_file+'--statest.json')
            median_vals = plotMerge.plot_Box_Grouped(emphasis[1], \
                                emph2_plot_out_file, \
                                colors_bw, \
                                y_repr+"MS"+n_suff+" (%)", yticks_range=yticks_range, \
                                    selectData=selected_data)
            dumpJson(median_vals, emph2_plot_out_file+'.medians.json')

    return info_best_sota_klee
#~ def compute_n_plot_param_influence()

def best_worst_conf(merged_df, outdir, SpecialTechs, ms_by_time, n_suff, \
                                techConfbyvalbyconf, ms_apfds, proj_agg_func=None):
    #topN = 1
    topN = 5
    apfd_ordered_techconf_list = sorted(list(set(merged_df[techConfCol])), \
                    reverse=True, \
                    key=lambda x: proj_agg_func(getListAPFDSForTechConf(x, ms_apfds)))
    best_val_tmp = proj_agg_func(getListAPFDSForTechConf(\
                                            apfd_ordered_techconf_list[topN-1], ms_apfds))
    worse_val_tmp = proj_agg_func(getListAPFDSForTechConf(\
                                            apfd_ordered_techconf_list[-topN], ms_apfds))
    best_elems = []
    worse_elems = []
    for i, v in enumerate(apfd_ordered_techconf_list):
        if proj_agg_func(getListAPFDSForTechConf(v, ms_apfds)) >= best_val_tmp:
            best_elems.append(v)
        if proj_agg_func(getListAPFDSForTechConf(v, ms_apfds)) <= worse_val_tmp:
            worse_elems.append(v)
    best_elems.sort(reverse=True, key=lambda x: proj_agg_func(getListAPFDSForTechConf(v, ms_apfds)))
    worse_elems.sort(reverse=False, key=lambda x: proj_agg_func(getListAPFDSForTechConf(v, ms_apfds)))
    # get corresponding param values and save as csv (best and worse)
    best_df_obj = []
    worse_df_obj = []
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
            row['MS'+n_suff+'_INC_APFD'] = proj_agg_func(getListAPFDSForTechConf(v, ms_apfds))
            df_obj_list.append(row)
    best_df = pd.DataFrame(best_df_obj)
    worse_df = pd.DataFrame(worse_df_obj)
    best_df_file = os.path.join(outdir, "best_tech_conf_apfd-top"+str(topN)+".csv")
    worse_df_file = os.path.join(outdir, "worst_tech_conf_apfd-worst"+str(topN)+".csv")
    best_df.to_csv(best_df_file, index=False)
    worse_df.to_csv(worse_df_file, index=False)

    # XXX plot best and worse median over time compared with klee
    b_image = os.path.join(outdir, "best_tech_conf_time-top"+str(topN))
    w_image = os.path.join(outdir, "worst_tech_conf_time-top"+str(topN))
    for bw_elems, bw_image, bw in ((best_elems, b_image, 'best'), (worse_elems, w_image, 'worst')):
        plotobj = {}
        if not KLEE_KEY in bw_elems:
            elems = bw_elems + [KLEE_KEY]
        else:
            elems = bw_elems
        for i, v in enumerate(elems):
            if v == KLEE_KEY:
                name_ = SpecialTechs[v]
            else:
                name_ = bw+str(i+1)
            v_data = ms_by_time[v]
            plotobj[name_] = []
            plotobj[name_].append(np.median([v_data[p][0] for p in v_data]))
            plotobj[name_].append(np.median([v_data[p][1] for p in v_data]))

            if i > topN: #XXX focus on topN
                break

        plotLines(plotobj, sorted(plotobj.keys()), "time(min)", "MS"+n_suff, bw_image, colors, linestyles, linewidths, fontsize)

    return best_elems, worse_elems
#~ def best_worst_conf()

def select_top_or_all(merged_df, customMaxtime, best_elems, worse_elems):
    if customMaxtime is None:
        assert False, "Must specify a customMaxtime"

    # XXX: XXX: Decide whether to continue with all, only topN or only worseN or both topN and worseN
    SEL_use_these_confs = 'topN'
    if SEL_use_these_confs != "ALL":
        if SEL_use_these_confs == "topN":
            considered_c = list(set(best_elems+[KLEE_KEY]))
        elif SEL_use_these_confs == "worstN":
            considered_c = list(set(worse_elems+[KLEE_KEY]))
        elif SEL_use_these_confs == "topN-worstN":
            considered_c = list(set(best_elems+worse_elems+[KLEE_KEY]))
        sel_merged_df = merged_df[merged_df[techConfCol].isin(considered_c)]
    return sel_merged_df
#~ def select_top_or_all()

def plot_extra_data(time_snap_df, time_snap, outdir, ms_apfds, msCol, \
                                    targetCol, covMutsCol, \
                                    numMutsCol, n_suff):
    fixed_y = msCol
    changing_ys = [targetCol, covMutsCol, stateCompTimeCol, numGenTestsCol, \
                        propNoDiffOnForkedMutsStatesCol, \
                        numFailTestsCol] # Fail is used for verification purpose
    # get data and plot
    tmp_tech_confs = set(time_snap_df[techConfCol])
    metric2techconf2values = {}
    # for each metric, get per techConf list on values
    for tech_conf in tmp_tech_confs:
        t_c_tmp_df = time_snap_df[time_snap_df[techConfCol] == tech_conf]
        for metric_col in [fixed_y] + changing_ys + [numMutsCol]:
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
        ms_inc_apfd_y = "Average MS"+n_suff+"-INC"
        ms_inc_apfd_vals = []
        fix_vals = []
        for tech_conf in sorted_techconf_by_ms:
            if tech_conf in SPECIAL_TECHS:
                continue
            fix_vals.append(metric2techconf2values[fixed_y][tech_conf])
            ms_inc_apfd_vals.append([ms_apfds[p][tech_conf] for p in ms_apfds])
        plot_img_out_file = os.path.join(outdir, "msapfdVSms-"+ \
                                                            str(time_snap))
        make_twoside_plot(fix_vals, ms_inc_apfd_vals, img_out_file=plot_img_out_file, \
                        x_label="Configuations", y_left_label=fixed_y, \
                                            y_right_label=ms_inc_apfd_y)

        # Make plots of ms and the others
        for chang_y in changing_ys:
            plot_img_out_file = os.path.join(outdir, "otherVSms-"+ \
                            str(time_snap) + "-"+chang_y.replace('#','n'))
            chang_vals = []
            nMuts_here = []
            for tech_conf in sorted_techconf_by_ms:
                if tech_conf in SPECIAL_TECHS:
                    continue
                chang_vals.append(\
                                metric2techconf2values[chang_y][tech_conf])
                nMuts_here.append(\
                                metric2techconf2values[numMutsCol][tech_conf])

            if chang_y in [targetCol, stateCompTimeCol, covMutsCol]:
                if chang_y in (targetCol, covMutsCol):
                    chang_y = chang_y.replace('#', '%')
                    m_val = nMuts_here
                    # compute percentage
                    if len(m_val) != 0:
                        for c_ind in range(len(chang_vals)):
                            c_tmp = []
                            for i_ind in range(len(chang_vals[c_ind])):
                                if m_val[c_ind][i_ind] != 0:
                                    c_tmp.append(chang_vals[c_ind][i_ind] * 100.0 / m_val[c_ind][i_ind])
                            chang_vals[c_ind] = c_tmp
                elif chang_y == stateCompTimeCol:
                    chang_y = chang_y.replace('(s)', '(%)')
                    m_val = time_snap * 60
                    # compute percentage
                    if m_val != 0:
                        for c_ind in range(len(chang_vals)):
                            chang_vals[c_ind] = \
                                    [v*100.0/m_val for v in chang_vals[c_ind]]
                else:
                    assert False, "BUG, unreachable"
                    
            make_twoside_plot(fix_vals, chang_vals, img_out_file=plot_img_out_file, \
                        x_label="Configuations", y_left_label=fixed_y, \
                                                    y_right_label=chang_y)
    return sorted_techconf_by_ms, metric2techconf2values  
#~ plot_extra_data()

def get_overlap_data(proj2dir, projcommonreldir, time_snap, subsuming, \
                                                        sorted_techconf_by_ms, all_initial):
    overlap_data_dict = {}
    non_overlap_obj = {}
    tech_conf_missed_muts = {}
    for proj in proj2dir:
        # consider the filterings in loadData
        if proj not in all_initial:
            continue
        fulldir = os.path.join(proj2dir[proj], projcommonreldir)
        full_overlap_file = os.path.join(fulldir, \
                    "Techs-relation.json-"+str(int(time_snap))+"min.json")
        assert os.path.isfile(full_overlap_file), "file not existing: "+\
                            full_overlap_file
        with open(full_overlap_file) as f:
            fobj = json.load(f)
            non_overlap_obj[proj] = {}
            overlap_data_dict[proj] = {}
            tech_conf_missed_muts[proj] = {}
            visited = set()
            s_m_key = 'SUBSUMING_CLUSTERS' if subsuming else 'MUTANTS'
            for pair in fobj[s_m_key]["NON_OVERLAP_VENN"]:
                a_tmp = pair.split('&')
                if len(a_tmp) != 2:
                    print("@WARNING: non pair overlap found:", pair)
                    continue
                left_right = tuple(sorted(a_tmp))
                if left_right not in visited:
                    visited.add(left_right)
                    overlap_data_dict[proj][left_right] = fobj[s_m_key]["OVERLAP_VENN"][pair]
                    
                if left_right not in non_overlap_obj[proj]:
                    non_overlap_obj[proj][left_right] = {v:0 for v in left_right}
                
                for tmp_tc in left_right:
                    if tmp_tc not in tech_conf_missed_muts[proj]:
                        tech_conf_missed_muts[proj][tmp_tc] = set()
                for win in fobj[s_m_key]["NON_OVERLAP_VENN"][pair]:
                    non_overlap_obj[proj][left_right][win] += \
                                len(fobj[s_m_key]["NON_OVERLAP_VENN"][pair][win])
                    lose = set(left_right) - {win}
                    assert len(lose) == 1
                    lose = list(lose)[0]
                    tech_conf_missed_muts[proj][lose] |= set(fobj[s_m_key]["NON_OVERLAP_VENN"][pair][win])
    all_tech_confs = {v for p in non_overlap_obj[non_overlap_obj.keys()[0]] for v in p}
    # We only use top 5 and KLEE
    #assert all_tech_confs == set(sorted_techconf_by_ms), \
    #                    "Inconsistemcy between dataframe and overlap json"
    tech_conf2position = {}
    for pos, tech_conf in enumerate(sorted_techconf_by_ms):
        tech_conf2position[tech_conf] = pos
    return tech_conf_missed_muts, non_overlap_obj, overlap_data_dict, tech_conf2position
#~ def get_overlap_data()

def process_minimal_config_set(outdir, tech_conf_missed_muts, techConf2ParamVals, get_all=True):
    # write down the minimal config set to kill all muts
    print("# Computing Minimal config set...")
    semu_only_tech_conf_missed_muts = {}
    for proj in tech_conf_missed_muts:
        semu_only_tech_conf_missed_muts[proj] = {}
        for tc in tech_conf_missed_muts[proj]:
            if tc not in SPECIAL_TECHS:
                semu_only_tech_conf_missed_muts[proj][tc] = tech_conf_missed_muts[proj][tc]

    # TODO: add computing per proj and overal increase of minimal conf and all (including KLEE). Use time_snap_df and pick a conf
    minimal_tech_confs, minimal_missed = get_minimal_conf_set(semu_only_tech_conf_missed_muts, get_all=get_all)
    minimal_df_obj = []
    ordered_minimal_conf = []
    if get_all:
        assert "Get_all enabled is not yet supported"
    else:
        for mtc_clust in minimal_tech_confs[0]:
            mtc = mtc_clust[0]
            ordered_minimal_conf.append(mtc)
            minimal_df_obj.append(dict(list({'_TechConf': mtc}.items())+list(techConf2ParamVals[mtc].items())))
        minimal_df = pd.DataFrame(minimal_df_obj)
        minimal_df.to_csv(os.path.join(outdir, "minimal_tech_confs.csv"), index=False)        
    print("# Done computing Minimal config set...")
    return minimal_missed, ordered_minimal_conf
#~ def process_minimal_config_set()

def compute_and_store_total_increase(outdir, tech_conf_missed_muts, minimal_num_missed_muts, ordered_minimal_set, time_snap_df,\
                                    all_initial, merged_initial_ms_json_obj, initialNumMutsKey, initialKillMutsKey, numMutsCol, killMutsCol, \
                                    n_suff='', use_fixed=False):
    y_repr = "" if use_fixed else "AVERAGE " # Over time Average
    tmp_elem = list(time_snap_df[techConfCol])[0]
    tmp_elem_df = time_snap_df[time_snap_df[techConfCol] == tmp_elem]

    add_total_cand_muts_by_proj = {}
    add_total_killed_muts_by_proj = {}
    for _, row in tmp_elem_df.iterrows():
        proj = row[PROJECT_ID_COL]
        add_total_killed_muts_by_proj[proj] = row[killMutsCol] + len(tech_conf_missed_muts[proj][tmp_elem])
        add_total_cand_muts_by_proj[proj] = row[numMutsCol]

    klee_killed_by_proj = {}
    for proj, tot_k in list(add_total_killed_muts_by_proj.items()):
        klee_killed_by_proj[proj] = tot_k - len(tech_conf_missed_muts[proj][KLEE_KEY])

    def compute_num_killed(tc_list):
        res_per_proj = {}
        for proj, d in list(tech_conf_missed_muts.items()):
            missed = set(d[tc_list[0]])
            for tc in tc_list[1:]:
                missed &= set(d[tc])
            res_per_proj[proj] = add_total_killed_muts_by_proj[proj] - len(missed)
        return res_per_proj
    #~ def compute_num_killed()

    # Plot evolution of minimal
    bp_data = []
    bp_data_final = []
    for i in range(1, len(ordered_minimal_set)+1):
        killed_per_proj = compute_num_killed(ordered_minimal_set[:i])
        bp_data.append([(0 if add_total_cand_muts_by_proj[p] == 0 else killed_per_proj[p] * 100.0 / add_total_cand_muts_by_proj[p]) \
                                                                                                        for p in killed_per_proj])
        bp_data_final.append([(all_initial[p][initialKillMutsKey] + killed_per_proj[p]) * 100.0 / all_initial[p][initialNumMutsKey] \
                                                                                                            for p in killed_per_proj])
    bp_data.append([(0 if add_total_cand_muts_by_proj[p] == 0 else add_total_killed_muts_by_proj[p] * 100.0 / add_total_cand_muts_by_proj[p]) \
                                                                                                for p in add_total_killed_muts_by_proj])
    bp_data_final.append([(all_initial[p][initialKillMutsKey] + add_total_killed_muts_by_proj[p]) * 100.0 / all_initial[p][initialNumMutsKey] \
                                                                                                for p in add_total_killed_muts_by_proj])

    image_file = os.path.join(outdir, "minimal_config_evolution-additional")
    image_file_final = os.path.join(outdir, "minimal_config_evolution-final")

    make_twoside_plot(bp_data, None, img_out_file=image_file, x_vals=list(range(1, len(bp_data)+1))+['all'], \
                x_label="Configuations", y_left_label=y_repr+"MS"+n_suff+" (%)", show_grid=False)
    make_twoside_plot(bp_data_final, None, img_out_file=image_file_final, x_vals=list(range(1, len(bp_data_final)+1))+['all'], \
                x_label="Configuations", y_left_label=y_repr+"MS"+n_suff+" (%)", show_grid=False)

    # Stove Overal data
    json_obj = {}
    json_obj['Inc_Num_Mut_total_killed'] = sum([nk for _, nk in list(add_total_killed_muts_by_proj.items())])
    json_obj['Inc_Num_Mut_total_candidate'] = sum([nk for _, nk in list(add_total_cand_muts_by_proj.items())])
    json_obj['Inc_MS_total'] = 100.0 * json_obj['Inc_Num_Mut_total_killed'] / json_obj['Inc_Num_Mut_total_candidate']
    json_obj['Inc_Num_Mut_Semu_Minimal_killed'] = json_obj['Inc_Num_Mut_total_killed'] - minimal_num_missed_muts
    json_obj['Inc_MS_Semu_minimal'] = 100.0 * json_obj['Inc_Num_Mut_Semu_Minimal_killed'] / json_obj['Inc_Num_Mut_total_candidate']
    json_obj['Final_MS_total'] = 100.0 * \
                            (merged_initial_ms_json_obj[initialKillMutsKey] + json_obj['Inc_Num_Mut_total_killed']) / \
                                merged_initial_ms_json_obj[initialNumMutsKey]
    json_obj['Final_MS_Semu_minimal'] = 100.0 * \
                            (merged_initial_ms_json_obj[initialKillMutsKey] + json_obj['Inc_Num_Mut_Semu_Minimal_killed']) / \
                                merged_initial_ms_json_obj[initialNumMutsKey]
    json_obj['Initial_MS_total'] = 100.0 * merged_initial_ms_json_obj[initialKillMutsKey] / merged_initial_ms_json_obj[initialNumMutsKey]
    dumpJson(json_obj, os.path.join(outdir, "Total_Increase_Data.json"))

    return add_total_cand_muts_by_proj, add_total_killed_muts_by_proj
#~ def compute_and_store_total_increase()

def mutation_scores_best_sota_klee(outdir, add_total_cand_muts_by_proj, add_total_killed_muts_by_proj, \
                                    tech_conf_missed_muts, info_best_sota_klee, \
                                    all_initial, initialNumMutsKey, initialKillMutsKey, numMutsCol, killMutsCol, \
                                    n_suff='', use_fixed=False):
    y_repr = "" if use_fixed else "AVERAGE " # Over time Average

    final_ms = {}
    for key, name in [(KLEE_KEY, 'klee'), \
                        (info_best_sota_klee['max'][infectOnly][techConfCol], infectOnly),\
                        (info_best_sota_klee['max'][semuBEST][techConfCol], semuBEST)]:
        nkilled_by_proj = {}
        nmiss_by_proj = {}
        for proj in tech_conf_missed_muts:
            n_miss = len(tech_conf_missed_muts[proj][key])
            nkilled_by_proj[proj] = add_total_killed_muts_by_proj[proj] - n_miss
            nmiss_by_proj[proj] = n_miss
        techperf_miss = [[], []]
        techperf_miss_final = [[], [], []]
        x_vals = []
        for proj in nkilled_by_proj:
            techperf_miss[0].append(0 if add_total_cand_muts_by_proj[proj] == 0 else nkilled_by_proj[proj] * 100.0 / add_total_cand_muts_by_proj[proj])
            techperf_miss[1].append(0 if add_total_cand_muts_by_proj[proj] == 0 else nmiss_by_proj[proj] * 100.0 / add_total_cand_muts_by_proj[proj])
            techperf_miss_final[1].append(nkilled_by_proj[proj] * 100.0 / all_initial[proj][initialNumMutsKey])
            techperf_miss_final[2].append(nmiss_by_proj[proj] * 100.0 / all_initial[proj][initialNumMutsKey])
            techperf_miss_final[0].append(all_initial[proj][initialKillMutsKey] * 100.0 / all_initial[proj][initialNumMutsKey])
            x_vals.append(proj)

        # sort
        x_vals_final = copy.deepcopy(x_vals)
        techperf_miss[0], techperf_miss[1], x_vals = [list(v) for v in \
                                                zip(*sorted(zip(techperf_miss[0], techperf_miss[1], x_vals)))]
        techperf_miss_final[0], techperf_miss_final[1], techperf_miss_final[2], x_vals_final = [list(v) for v in \
                                            zip(*sorted(zip(techperf_miss_final[0], techperf_miss_final[1], techperf_miss_final[2], x_vals_final)))]


        x_label=None #'Programs'
        image_file = os.path.join(outdir, "selected_techs_MS-additional-"+name)
        image_file_final = os.path.join(outdir, "selected_techs_MS-final-"+name)
        make_twoside_plot(techperf_miss, None, x_vals=x_vals, \
                    img_out_file=image_file, \
                    x_label=x_label, y_left_label=y_repr+"MS"+n_suff+" (%)", \
                                left_stackbar_legends=[name, 'missed'], left_color_list=[goodViewColors[name], goodViewColors['missed']])
        make_twoside_plot(techperf_miss_final, None, x_vals=x_vals_final, \
                    img_out_file=image_file_final, \
                    x_label=x_label, y_left_label=y_repr+"MS"+n_suff+" (%)", \
                                left_stackbar_legends=['initial' ,name, 'missed'], \
                                left_color_list=[goodViewColors['initial'], goodViewColors[name], goodViewColors['missed']])
        final_ms[name] = {}
        final_ms[name]['MED'] = np.median([a+b for a,b in zip(techperf_miss_final[0], techperf_miss_final[1])])
        final_ms[name]['AVG'] = np.average([a+b for a,b in zip(techperf_miss_final[0], techperf_miss_final[1])])
        if 'ALL_TECHS' not in final_ms:
            final_ms['ALL_TECHS'] = {}
            final_ms['ALL_TECHS']['MED'] = np.median([a+b+c for a,b,c in zip(techperf_miss_final[0], techperf_miss_final[1], techperf_miss_final[2])])
            final_ms['ALL_TECHS']['AVG'] = np.average([a+b+c for a,b,c in zip(techperf_miss_final[0], techperf_miss_final[1], techperf_miss_final[2])]) 
    final_ms['INITIAL'] = {}
    final_ms['INITIAL']['MED'] = np.median([all_initial[p][initialKillMutsKey] * 100.0 / all_initial[p][initialNumMutsKey] for p in all_initial])
    final_ms['INITIAL']['AVG'] = np.average([all_initial[p][initialKillMutsKey] * 100.0 / all_initial[p][initialNumMutsKey] for p in all_initial])
    dumpJson(final_ms, image_file_final+'.res.json')
#~def mutation_scores_best_sota_klee()

def plot_gentest_killing(outdir, merged_df, time_snap, best_elems, info_best_sota_klee):
    time_snap_df = merged_df[merged_df[timeCol] == time_snap]

    # get the data
    total_tests = {}
    killing_tests = {}
    killing_over_all = {}
    nmut_per_test = {}
    order = [semuBEST, infectOnly, 'klee']
    for tech in order:
        total_tests[tech] = {}
        killing_tests[tech] = {}
        killing_over_all[tech] = {}
        nmut_per_test[tech] = {}
        raw_name = info_best_sota_klee['max'][tech][techConfCol]
        tech_df = time_snap_df[time_snap_df[techConfCol] == raw_name]
        for _,row in tech_df.iterrows():
            total_tests[tech][row[PROJECT_ID_COL]] = row[numGenTestsCol]
            
    # stat test
    def inner_stattest2(obj_dict, filename):
        statstest_obj = {}
        for pos1,t1 in enumerate(obj_dict):
            for pos2, t2 in enumerate(obj_dict):
                if pos1 >= pos2:
                    continue
                tmp_stats = {}
                tmp_stats['p_value'] = wilcoxon(obj_dict[t1], obj_dict[t2], isranksum=False)
                tmp_stats['A12'] = a12(obj_dict[t1], obj_dict[t2], pairwise=True)
                statstest_obj[str((t1, t2))] = tmp_stats
        dumpJson(statstest_obj, filename)
    #~ def inner_stattest2()

    # plot
    ## total
    imagefile = os.path.join(outdir, 'totalGenTests_best_sota_klee')
    plotobj = {t:[] for t in total_tests}
    for t in plotobj:
        for proj in total_tests[t].keys():
            plotobj[t].append(total_tests[t][proj])
    medians = plotMerge.plotBoxes(plotobj, order, imagefile, colors_bw, ylabel="# Generated Tests" , yticks_range=None)#range(0,101,10))
    inner_stattest2(plotobj, imagefile+'--statest.json')
    dumpJson(medians, imagefile+'.medians.json')
#~ def plot_gentest_killing()

def plot_overlap_1(outdir, time_snap, non_overlap_obj, best_elems, overlap_data_dict, info_best_sota_klee, add_total_cand_muts_by_proj):
    #x_label = "Winning Technique Configuration"
    #y_label = "Other Technique Configuration"
    #hue = "special"
    #num_x_wins = "# Mutants Killed more by X"

    SEL_use_best_apfd = True # decide whether to use best APFD of maxes

    if SEL_use_best_apfd:
        rounds = [[(best_elems[0], semuBEST), (KLEE_KEY, 'klee')],
                    [(best_elems[0], semuBEST), (info_best_sota_klee['max'][infectOnly][techConfCol], infectOnly)],
                    [(info_best_sota_klee['max'][infectOnly][techConfCol], infectOnly), (KLEE_KEY, 'klee')]]
    else:
        rounds = [[(None, 'semu'), (KLEE_KEY, 'klee')]]

    for principal, secondary in rounds:
        princ_key, princ_name = principal
        sec_key, sec_name = secondary

        # Plot klee conf overlap by proj
        klee_n_semu_by_proj = [[], []]
        by_proj_overlap = []
        x_vals = []
        for proj in non_overlap_obj:
            klee_n_semu_by_proj[0].append(None)
            klee_n_semu_by_proj[1].append(None)
            by_proj_overlap.append(0)
            if SEL_use_best_apfd:
                for left_right in non_overlap_obj[proj]:
                    if sec_key in left_right and princ_key in left_right:
                        klee_n_semu_by_proj[0][-1] = non_overlap_obj[proj][left_right][princ_key]
                        klee_n_semu_by_proj[1][-1] = non_overlap_obj[proj][left_right][sec_key]
                        by_proj_overlap[-1] = overlap_data_dict[proj][left_right]
                        x_vals.append(proj)
                        break
            else:
                for left_right in non_overlap_obj[proj]:
                    if sec_key in left_right:
                        s_c_n_o = non_overlap_obj[proj][left_right][list(set(left_right)-{sec_key})[0]]
                        k_n_o = non_overlap_obj[proj][left_right][sec_key]
                        if klee_n_semu_by_proj[0][-1] is None or \
                                s_c_n_o - k_n_o > klee_n_semu_by_proj[0][-1] - klee_n_semu_by_proj[1][-1]:
                            klee_n_semu_by_proj[0][-1] = s_c_n_o
                            klee_n_semu_by_proj[1][-1] = k_n_o
                            by_proj_overlap[-1] = overlap_data_dict[proj][left_right]
                            x_vals.append(proj)
            if klee_n_semu_by_proj[1] > klee_n_semu_by_proj[0]:
                print(">>>> Klee has higher non overlap that all semu for project", proj, "(", klee_n_semu_by_proj[1], "VS", klee_n_semu_by_proj[0], ")")
        

        ## plot
        x_vals_bak = x_vals
        x_label=None #'Programs'
        for suffix, ylabel_name in [('number', "# Killed Mutants"), ('proportion', "Proportion Killed Mutants")]:
            tri_lists = klee_n_semu_by_proj+[by_proj_overlap]
            x_vals = list(x_vals_bak)

            if suffix == 'proportion':
                for ll in tri_lists:
                    for i in range(len(ll)):
                        if add_total_cand_muts_by_proj[x_vals[i]] != 0:
                            ll[i] = ll[i] * 1.0 / add_total_cand_muts_by_proj[x_vals[i]]
                        
            # sort
            tri_lists[0], tri_lists[1], tri_lists[2], x_vals = [list(v) for v in \
                                                zip(*sorted(zip(tri_lists[0], tri_lists[1], tri_lists[2], x_vals)))]
        
            image_file = os.path.join(outdir, "proj_overlap-"+princ_name+'VS'+sec_name+str(time_snap)+"min."+suffix)
            #make_twoside_plot(klee_n_semu_by_proj+[by_proj_overlap], klee_n_semu_by_proj, \
            make_twoside_plot(tri_lists, None, x_vals=x_vals, \
                        img_out_file=image_file, \
                        x_label=x_label, y_left_label=ylabel_name, \
                                        y_right_label="# Non Overlapping Mutants", \
                                    left_stackbar_legends=[princ_name, sec_name, 'overlap'], \
                                    left_color_list=[goodViewColors[princ_name], goodViewColors[sec_name], goodViewColors['overlap']])
            # save data as json
            json_obj = {}
            tmp = tri_lists
            for pos, proj in enumerate(x_vals):
                json_obj[proj] = {princ_name: tmp[0][pos], sec_name: tmp[1][pos],\
                                    'OVERLAP': tmp[2][pos]}
            dumpJson(json_obj, image_file+'.data.json')
#~ def plot_overlap_1()

def plot_overlap_2(outdir, non_overlap_obj, SpecialTechs, tech_conf2position, \
                            proj_agg_func2, proj_agg_func2_name, time_snap):
    x_label = "Winning Technique Configuration"
    y_label = "Other Technique Configuration"
    hue = "special"
    num_x_wins = "# Mutants Killed more by X"
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
            if left in tech_conf2position and right in tech_conf2position:
                df_obj.append({
                        x_label: tech_conf2position[left], 
                        y_label: tech_conf2position[right], 
                        hue: hue_val,
                        num_x_wins: proj_agg_func2([non_overlap_obj[p][left_right][left] for p in non_overlap_obj]), 
                        })
    killed_muts_overlap = pd.DataFrame(df_obj)
    image_out = os.path.join(outdir, "overlap-"+proj_agg_func2_name+"-"+str(time_snap)+"min")
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

    return killed_muts_overlap
#~ def plot_overlap_2()

def plot_overlap_3(outdir, best_elems, msCol, proj_agg_func2_name,time_snap, \
                    killed_muts_overlap, sorted_techconf_by_ms, \
                        metric2techconf2values, tech_conf2position, \
                                        proj_agg_func2, overlap_data_dict):
    x_label = "Winning Technique Configuration"
    y_label = "Other Technique Configuration"
    hue = "special"
    num_x_wins = "# Mutants Killed more by X"
    # plot overlap against pure klee
    image_out2 = os.path.join(outdir, \
                            "semu_klee-nonoverlap-"+proj_agg_func2_name+"-"+str(time_snap)+"min")
    fixed_y = msCol
    chang_y2 = "# Non Overlapping Mutants"
    fix_vals2 = []
    sb_legend = ['semu', 'klee']
    chang_vals2 = [[], []]
    overlap_vals = []
    x_vals = None
    if set(best_elems) == set(sorted_techconf_by_ms):
        x_vals = []
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

    make_twoside_plot(fix_vals2, chang_vals2, img_out_file=image_out2, \
                x_label="Configuations", y_left_label=fixed_y, \
                                            y_right_label=chang_y2, \
                                right_stackbar_legends=sb_legend)
    # overlap and non overlap
    image_out3 = os.path.join(outdir, \
                            "semu_klee-overlap_all-"+proj_agg_func2_name+"-"+str(time_snap)+"min")
    overlap_non_vals = chang_vals2 + [overlap_vals] 
    #make_twoside_plot(overlap_non_vals, chang_vals2, img_out_file=image_out3, \
    make_twoside_plot(overlap_non_vals, None, img_out_file=image_out3, \
                x_label="Configuations", y_left_label="# Killed Mutants", \
                                            y_right_label=chang_y2, \
                            left_stackbar_legends=sb_legend+['overlap'], \
                            right_stackbar_legends=sb_legend)
#~ def plot_overlap_3()

#### CONTROLLER ####
def libMain(outdir, proj2dir, use_func=False, customMaxtime=None, \
                projcommonreldir=None, onlykillable=False, no_concrete=False):
    has_subsuming_data = True

    if projcommonreldir is None:
        projcommonreldir = getProjRelDir()

    # load the data
    loaded_merged_df, all_initial = loadData(proj2dir, use_func, \
                                                projcommonreldir, onlykillable)

    # Compute merged initial
    merged_initial_ms_json_obj = merge_initial(all_initial, outdir, use_func, \
                                        has_subsuming_data=has_subsuming_data)
    
    # COMPUTATIONS ON DF
    merged_df, SpecialTechs = preprocessing_merge_df(loaded_merged_df, \
                                                    customMaxtime, no_concrete)

    ##################################
    ##################################

    # backup outdir
    outdir_bak = outdir
    merged_df_bak = merged_df

    for subsuming in ((True, False) if has_subsuming_data else (False,)):
        merged_df = merged_df_bak.copy(deep=True)
        if subsuming:
            outdir = os.path.join(outdir_bak, "subsumingMS")
            msCol = "MS_SUBSUMING-INC"
            numMutsCol = "#SubsMutantsClusters"
            targetCol = "#SubsTargetedClusters"
            covMutsCol = "#SubsCoveredClusters"
            killMutsCol = "#SubsKilledClusters"
            initialKillMutsKey = "Initial#SubsumingKilledMutants"
            initialNumMutsKey = "Initial#SubsumingMutants"
            n_suff = '*'
        else:
            outdir = os.path.join(outdir_bak, "traditionalMS")
            msCol = "MS-INC"
            numMutsCol = "#Mutants"
            targetCol = "#Targeted"
            covMutsCol = "#Covered"
            killMutsCol = "#Killed"
            initialKillMutsKey = "Initial#KilledMutants"
            initialNumMutsKey = "Initial#Mutants"
            n_suff = ''
        
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        outdir_inner_bak = outdir
        for use_fixed in [True, False]:
            if use_fixed:
                outdir = os.path.join(outdir_inner_bak, "fixed_at_"+str(customMaxtime))
                if not os.path.isdir(outdir):
                    os.mkdir(outdir)
                # fix val Computation
                ms_apfds, ms_by_time = get_ms_apfds(merged_df[merged_df[timeCol] == customMaxtime], msCol)
            else:
                outdir = os.path.join(outdir_inner_bak, "apfd")
                if not os.path.isdir(outdir):
                    os.mkdir(outdir)
                # APFDs Computation
                ms_apfds, ms_by_time = get_ms_apfds(merged_df, msCol)

            # Get only SEMU df
            only_semu_cfg_df = merged_df[~merged_df[techConfCol].isin(SPECIAL_TECHS)]

            # get some utils ...
            techConf2ParamVals, techConfbyvalbyconf = get_techConfUtils(only_semu_cfg_df, SpecialTechs)

            #proj_agg_func = np.median
            proj_agg_func = np.average

            # XXX Check the influence of each parameter. 
            info_best_sota_klee = compute_n_plot_param_influence(techConfbyvalbyconf, outdir, \
                                        SpecialTechs, n_suff, ms_apfds, proj_agg_func=np.median, \
                                        bests_only=False, use_fixed=use_fixed)
            
            # XXX Find best and worse confs
            best_elems, worse_elems = best_worst_conf(merged_df, outdir, \
                                        SpecialTechs, ms_by_time, n_suff, \
                                        techConfbyvalbyconf, ms_apfds, proj_agg_func=np.median)

            # Select what to continue
            sel_merged_df = select_top_or_all(merged_df, \
                                            customMaxtime, best_elems, worse_elems)

            time_snap = customMaxtime
            time_snap_df = sel_merged_df[sel_merged_df[timeCol] == time_snap]
            if time_snap_df.empty:
                continue

            # XXX compare MS with compareState time, %targeted, #testgen, WM%
            sorted_techconf_by_ms, metric2techconf2values = \
                                plot_extra_data(sel_merged_df, time_snap, \
                                        outdir, ms_apfds, msCol, targetCol,\
                                        covMutsCol, numMutsCol, n_suff)

            # XXX Killed mutants overlap
            tech_conf_missed_muts, non_overlap_obj, \
                overlap_data_dict, tech_conf2position = \
                                    get_overlap_data(proj2dir, \
                                    projcommonreldir, time_snap, subsuming, \
                                                        sorted_techconf_by_ms, all_initial)

            minimal_num_missed_muts, ordered_minimal_set = \
                                        process_minimal_config_set(outdir, tech_conf_missed_muts, techConf2ParamVals, get_all=False)

            add_total_cand_muts_by_proj, add_total_killed_muts_by_proj = \
                     compute_and_store_total_increase(outdir, tech_conf_missed_muts, minimal_num_missed_muts, ordered_minimal_set, time_snap_df,
                                                all_initial, merged_initial_ms_json_obj, initialNumMutsKey, initialKillMutsKey, numMutsCol, killMutsCol, n_suff, use_fixed)

            mutation_scores_best_sota_klee(outdir, add_total_cand_muts_by_proj, add_total_killed_muts_by_proj, \
                                    tech_conf_missed_muts, info_best_sota_klee, \
                                    all_initial, initialNumMutsKey, initialKillMutsKey, numMutsCol, killMutsCol, \
                                    n_suff=n_suff, use_fixed=use_fixed)

            plot_gentest_killing(outdir, merged_df, time_snap, best_elems, info_best_sota_klee)

            plot_overlap_1(outdir, time_snap, non_overlap_obj, best_elems, overlap_data_dict, info_best_sota_klee, add_total_cand_muts_by_proj)

                            
            for proj_agg_func2, proj_agg_func2_name in [(np.average, "average"), (np.median, "median")]:
                killed_muts_overlap = plot_overlap_2(outdir, non_overlap_obj, \
                                SpecialTechs, tech_conf2position, \
                                proj_agg_func2, proj_agg_func2_name, time_snap)

                plot_overlap_3(outdir, best_elems, msCol, proj_agg_func2_name,time_snap, \
                            killed_muts_overlap, sorted_techconf_by_ms, \
                                metric2techconf2values, tech_conf2position, \
                                    proj_agg_func2, overlap_data_dict)

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
        #print(" ".join(list(proj2dir)))
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
