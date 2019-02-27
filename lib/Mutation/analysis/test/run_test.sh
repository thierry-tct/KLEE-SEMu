#! /bin/bash

# Example:
# >> ./run_test.sh tritype
#
# Env vars:
# DO_CLEANSTART=on         --> apply cleanstart on MFI to start everything over
# FROM_EXECUTION=SEMU   --> Execute this script from given execution: [PREPARE, SEMU, REPLAY, REPORT]
# MFIRUNSHADOW_VERBOSE=on  --> make klee test generation of MFI verbose
# AUTO_REEXECUTE_FAILED_SEMU=3  --> re-execute Semu when failed up to 3 times (N times specified) 
#                                    (Using EXTRA_ARGS="--semucontinueunfinishedtunings")
# 
#XXX INFO: In case there are some tests that fail with zesti and want to skip them,
#XXX INFO: Just rerun passing the environment variable: SEMU_ZESTI_RUN_SKIP_FAILURE=on

# While debugging semu, if some fail and we do not want to rerun then but only run those that fail,
# pass this Extra arg as env var:  EXTRA_ARGS="--semucontinueunfinishedtunings"
#
#


set -u

error_exit()
{
    echo "Error: $1"
    exit 1
}

[ $# = 1 ] || error_exit "Expects 1 parameter (test project ID), $# given"

projID=$1

topdir=$(dirname $(readlink -f $0))
metadir=$topdir/workspace/metactrl/$projID
reposdir=$topdir/workspace/repos/$projID
semudir=$topdir/SEMU_EXECUTION/$projID

export MART_BINARY_DIR=$(readlink -f ~/mytools/mart/build/tools)

run_semu_config=$(readlink -f ~/mytools/klee-semu/src/lib/Mutation/analysis/example/22/run_cmd.cfg)

from_exec=0
if [ "${FROM_EXECUTION:-}" != "" ]
then
    if [ "$FROM_EXECUTION" = "PREPARE" ]; then
        from_exec=1
    elif [ "$FROM_EXECUTION" = "GETSEEDS" ]; then
        from_exec=2
    elif [ "$FROM_EXECUTION" = "SEMU" ]; then
        from_exec=3
    elif [ "$FROM_EXECUTION" = "RESETSTATE" ]; then
        from_exec=4
    elif [ "$FROM_EXECUTION" = "REPLAY" ]; then
        from_exec=5
    elif [ "$FROM_EXECUTION" = "REPORT" ]; then
        from_exec=6
    else
        error_exit "Invalid FROM_EXECUTION value: '$FROM_EXECUTION'"
    fi
fi

# run MFI
if [ $from_exec -le 0 ] #true 
then
    cleanstart=""
    [ "${DO_CLEANSTART:-}" = "on" ] && cleanstart=cleanstart
    echo "# RUNNING MFI 1..."
    cd $metadir || error_exit "cd $metadir"
    ~/mytools/MFI-V2.0/MFI.sh "$projID"_conf-script.conf $cleanstart || error_exit "MFI Failed!"
    cd - > /dev/null
fi

# Prepare for SEMU
if [ $from_exec -le 1 ] #true 
then
    echo "# RUNNING prepareData..."
    cd $(dirname $semudir) || error_exit "failed entering semudir parent!"
    if test -d $projID
    then
        echo "## Removing existing dir..."
        rm -rf $projID
    fi
    bash ~/mytools/klee-semu/src/lib/Mutation/analysis/example/22/prepareData.sh $metadir/"$projID"_conf-script.conf . $run_semu_config || error_exit "Prepare for semu failed!"
    cd - > /dev/null
fi

# GETSEEDS
if [ $from_exec -le 2 ] #true 
then
    echo "# RUNNING GET SEEDS"
    cd $semudir || error_exit "failed to enter semudir!"
    #SKIP_TASKS="SEMU_EXECUTION COMPUTE_TASK ANALYSE_TASK" GIVEN_CONF_SCRIPT=$metadir/"$projID"_conf-script.conf \
    SKIP_TASKS="" GIVEN_CONF_SCRIPT=$metadir/"$projID"_conf-script.conf \
                                            bash ~/mytools/klee-semu/src/lib/Mutation/analysis/example/22/run_cmd . $run_semu_config || error_exit "Semu Failed"
    cd - > /dev/null
fi

# Run SEMU
if [ $from_exec -le 3 ] #true 
then
    echo "# RUNNING SEMU..."
    cd $semudir || error_exit "failed to enter semudir!"

    contunfinished="--semucontinueunfinishedtunings"
    if [ "${EXTRA_ARGS:-}" != "" ]
    then
        echo $EXTRA_ARGS | grep "\--semucontinueunfinishedtunings" > /dev/null && contunfinished=""
    fi
    
    tot_n_runs=2
    [ "${AUTO_REEXECUTE_FAILED_SEMU:-}" != "" ] && tot_n_runs=$AUTO_REEXECUTE_FAILED_SEMU
    [ $tot_n_runs -gt 0 ] || error_exit "Invalid tot_n_runs: $tot_n_runs"

    n_runs=$tot_n_runs
    fail=1
    while [ $fail -ne 0 ]
    do
        extra_args=""
        [ $n_runs -lt $tot_n_runs ] && extra_args="$contunfinished"
        SKIP_TASKS="ZESTI_DEV_TASK TEST_GEN_TASK" GIVEN_CONF_SCRIPT=$metadir/"$projID"_conf-script.conf EXTRA_ARGS="${EXTRA_ARGS:-} $extra_args" \
                                                                bash ~/mytools/klee-semu/src/lib/Mutation/analysis/example/22/run_cmd . $run_semu_config && fail=0
        n_runs=$(($n_runs - 1))
        [ $n_runs -le 0 ] && break
    done
    [ $fail -ne 0 ] && error_exit "Semu Failed. repeated $tot_n_runs times"

    cd - > /dev/null
fi

# ---------- RUN additional generated tests and analyse
# SET STATE
if [ $from_exec -le 4 ] #true 
then
    echo "# Setting back the State for Replay..."
    cd $metadir || error_exit "cd $metadir 2"
    python ~/mytools/MFI-V2.0/utilities/navigator.py --setexecstate 5 . || error_exit "failed to set exec state to 5"
    cd - > /dev/null
fi

# run additional
if [ $from_exec -le 5 ] #true 
then
    echo "# RUNNING MFI additional..."
    cd $metadir || error_exit "cd $metadir 3"
    sampl_mode=""
    for tmp in 'PASS' 'KLEE' 'DEV' 'NUM'
    do
        if test -f $semudir/OUTPUT/TestGenFinalAggregated"$tmp"_100.0/mfirun_mutants_list.txt
        then
            sampl_mode=$tmp
            break
        fi
    done
    [ "$sampl_mode" = "" ] && error_exit "maybe problem with gentest replaying: sampl_mode neither of PASS KLEE DEV NUM"

    MFI_OVERRIDE_OUTPUT=$semudir/OUTPUT/TestGenFinalAggregated"$sampl_mode"_100.0/mfirun_output MFI_OVERRIDE_MUTANTSLIST=$semudir/OUTPUT/TestGenFinalAggregated"$sampl_mode"_100.0/mfirun_mutants_list.txt MFI_OVERRIDE_GENTESTSDIR=$semudir/OUTPUT/TestGenFinalAggregated"$sampl_mode"_100.0/mfirun_ktests_dir ~/mytools/MFI-V2.0/MFI.sh "$projID"_conf-script.conf || error_exit "MFI Failed 2!"
    cd - > /dev/null
fi

# Analyse
if [ $from_exec -le 6 ] #true 
then
    echo "# RUNNING Semu analyse..."
    cd $semudir || error_exit "failed to enter semudir 2!"
    SKIP_TASKS="ZESTI_DEV_TASK TEST_GEN_TASK SEMU_EXECUTION COMPUTE_TASK" GIVEN_CONF_SCRIPT=$metadir/"$projID"_conf-script.conf bash ~/mytools/klee-semu/src/lib/Mutation/analysis/example/22/run_cmd . $run_semu_config || error_exit "Semu Failed analyse"
    cd - > /dev/null
fi

echo "DONE!"

