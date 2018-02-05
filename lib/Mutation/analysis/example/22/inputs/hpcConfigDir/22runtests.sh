#!/bin/bash

#Note: Make sure that here, all used tool are refered with absolute path (to avoid a control command to be considered as a test)

set -u
error_exit()
{
    /bin/echo $1
    exit 3
}

[ $# = 1 ] || error_exit "Error: runtest script expected 1 parameter, $# provided: <$0> <test name>"

TC=$1

/usr/bin/test -f $MFI_EXEDIR/$MFI_PROGRAM || error_exit "Error: Could not find executable $MFI_PROGRAM"
rootDir=$MFI_ROOTDIR 

testcaseDir="tests/expr"       #change here

#optimization of running time
# exit with code:
# 255 , when the optimization is enabled here, and the optimized run succeed
# 254 , when the optimization is enabled here, but the optimized run failed
if /bin/echo $TC | /bin/grep '\*\*MFI\*\*OPTIMIZE\*\*' > /dev/null
then
#error_exit "wrong test"
    cd $rootDir/$testcaseDir
    tmptc="mfi_ktest-replay-optimizer"
    /bin/echo "#! /bin/bash" > $tmptc
    /bin/echo $TC | /bin/sed 's/\*\*MFI\*\*OPTIMIZE\*\*//g' >> $tmptc
    /bin/chmod +x "$tmptc"
    /bin/echo "Replaying ktests..."
    returnCode=255   #This show that optimize is supported and was run sucessfully
     /usr/bin/make -i check-TESTS TESTS="$tmptc" VERBOSE=no > $tmptc.makeout || returnCode=254 
    /bin/grep "^# PASS:  1" $tmptc.makeout || returnCode=254 
    /bin/rm -f $tmptc ${tmptc}.log ${tmptc}.trs $tmptc.makeout
    cd - > /dev/null
    exit $returnCode
    #when 
fi        
 

cd $rootDir/$testcaseDir

export RUN_VERY_EXPENSIVE_TESTS=yes
export RUN_EXPENSIVE_TESTS=yes

fail=0

#KLEE Generated tests: scrip placed in exedir
testFile=$(/usr/bin/basename $TC)
[ "$testFile" = "MFI_KLEE_TOPDIR_TEST_TEMPLATE.sh" ] && TC=$MFI_EXEDIR/$testFile


#/bin/rm -rf gt-* 
#/bin/grep `/usr/bin/basename $TC` "$testcaseDir/root/testcaselist.txt" > /dev/null    #change here

#if [ $? = 0 ]                   
#then                                        #change here
#    if [ "${MFIOPT_KTESTSREPLAY_OPTIMIZER:-off}" = "on" ]
#    then
#        sudo  $TC > /dev/null
#    else
#        sudo  /usr/bin/make -i check-TESTS TESTS="$TC" VERBOSE=no | /bin/grep "PASS: $TC" >/dev/null || fail=1
#    fi
#else
    versionning=""
    [ "$TC" = "basic_s1" ] && versionning="yes"
    
    if [ "${MFIOPT_KTESTSREPLAY_OPTIMIZER:-off}" = "on" ]
    then
        VERBOSE=$versionning /bin/bash $TC > /dev/null
    else
         /usr/bin/make -i check-TESTS TESTS="$TC" VERBOSE=$versionning | /bin/grep "^# PASS:  1" > /dev/null || fail=1 
    fi
#fi

# /bin/rm -rf gt-*  

#  (let the report MFI execution whenever the OLD version timeout)
[ "${EKLEEPSE_REPLAY_LOG:-}" != "" ] && if [ "${EKLEEPSE_REPLAY_LOG: -11}" = ".old.newout" ]
then
    if /bin/grep "klee-replay: EXIT STATUS: TIMED OUT\|RETURN CODE: 124" $EKLEEPSE_REPLAY_LOG > \
                  /dev/null
    then
        # store the list of these tests into a file : filakyKtests.txt
        # build folder must be in the output dor since replay old new is not parallel
        /bin/echo $EKLEEPSE_REPLAY_KTEST >> $(/bin/readlink -f $MFI_BUILD_FOLDER/..)/data/genktests/flakyKtests.txt || \
                        error_exit "FLAKY-TESTS: original timeout(KLEE & Dev) --- Failed to save as FLAKY tests"
    fi
fi


cd - > /dev/null

exit $fail
