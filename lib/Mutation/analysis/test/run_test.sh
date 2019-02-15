#! /bin/bash

# Example:
# >> ./run_test.sh tritype
#
# Env vars:
# DO_CLEANSTART=on         --> apply cleanstart on MFI to start everything over
# FROM_SEMU_EXECUTION=on   --> Execute this script from SEMU execution
# MFIRUNSHADOW_VERBOSE=on  --> make klee test generation of MFI verbose
# 
#XXX INFO: In case there are some tests that fail with zesti and want to skip them,
#XXX INFO: Just rerun passing the environment variable: SEMU_ZESTI_RUN_SKIP_FAILURE=on

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

# run MFI
if [ "${FROM_SEMU_EXECUTION:-}" != "on" ] #true 
then
    cleanstart=""
    [ "${DO_CLEANSTART:-}" = "on" ] && cleanstart=cleanstart
    echo "# RUNNING MFI 1..."
    cd $metadir || error_exit "cd $metadir"
    ~/mytools/MFI-V2.0/MFI.sh "$projID"_conf-script.conf $cleanstart || error_exit "MFI Failed!"
    cd - > /dev/null
fi

# Prepare for SEMU
if [ "${FROM_SEMU_EXECUTION:-}" != "on" ] #true 
#if false
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

# Run SEMU
if true
then
    echo "# RUNNING SEMU..."
    cd $semudir || error_exit "failed to enter semudir!"
    SKIP_TASKS="${SKIP_TASKS:-}" GIVEN_CONF_SCRIPT=$metadir/"$projID"_conf-script.conf bash ~/mytools/klee-semu/src/lib/Mutation/analysis/example/22/run_cmd . $run_semu_config || error_exit "Semu Failed"
    cd - > /dev/null
fi

# ---------- RUN additional generated tests and analyse
# run additional
if true
then
    echo "# RUNNING MFI additional..."
    cd $metadir || error_exit "cd $metadir 2"
    python ~/mytools/MFI-V2.0/utilities/navigator.py --setexecstate 5 . || error_exit "failed to set exec state to 5"

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
if true
then
    echo "# RUNNING Semu analyse..."
    cd $semudir || error_exit "failed to enter semudir 2!"
    SKIP_TASKS="ZESTI_DEV_TASK TEST_GEN_TASK SEMU_EXECUTION COMPUTE_TASK" GIVEN_CONF_SCRIPT=$metadir/"$projID"_conf-script.conf bash ~/mytools/klee-semu/src/lib/Mutation/analysis/example/22/run_cmd . $run_semu_config || error_exit "Semu Failed analyse"
    cd - > /dev/null
fi

echo "DONE!"

