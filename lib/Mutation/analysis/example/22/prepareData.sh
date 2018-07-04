#! /bin/bash
# USAGE example: 
# bash ~/mytools/klee-semu/src/lib/Mutation/analysis/example/22/prepareData.sh ../workspace/metactrl/chroot/chroot_conf-script.conf .

set -u

error_exit()
{
    echo "Error: $1"
    exit 1
}

[ $# = 2 ] || error_exit "Expected two params $# given: <confFile> <destTopDir>"
confscript=$(readlink -f $1)
destTopDir=$(readlink -f $2)
zest=1

# load conf
cd $(dirname $confscript)
. $confscript
cd -

inmutsodldir=$(dirname $confscript)/$MFI_ID-output/mutants/$MFI_PROGRAM/mutation_SODL.txt
indatadir=$(dirname $confscript)/$MFI_ID-output/data

if [ "$zest" = "1" ]
then
    # COMPILE Project with llvm-2.7 and save with Zesti
    echo "Building fo Zesti BC ..."
    export LLVM_COMPILER='llvm-gcc'
    export LLVM_COMPILER_PATH='/home/shadowvm/shadow/kleeDeploy/llvm-2.9/Release+Asserts/bin' # CHANGE This XXX
    export PATH=$PATH:'/home/shadowvm/shadow/kleeDeploy/llvm-gcc4.2-2.9-x86_64-linux/bin' # CHANGE This XXX
    $MFI_BUILDSCRIPT "wllvm" "" build || error_exit "Build failed with wllvl llvm2.7"
    cd $MFI_EXEDIR && extract-bc $MFI_PROGRAM || error_exit "Failed extract bc"
    cp $MFI_PROGRAM.bc $inmutsodldir/$MFI_PROGRAM.Zesti.bc || error_exit "copy bc failed"
fi

# Copy stuffs
projOut=$destTopDir/$MFI_ID
test -d $projOut && error_exit "already processed $projOut. Delete manually for redo"
mkdir $projOut || error_exit "Failed to create $projOut"

mkdir -p $projOut/inputs/hpcConfigDir || error_exit "Failed to mkdir"

cp -rf $indatadir/matrices $projOut/inputs/ || error_exit "Failed to copy matrices"
cp -rf $indatadir/genktests $projOut/inputs/ || error_exit "Failed to copy genktests"
cp -rf $inmutsodldir $projOut/inputs/mutantsdata || error_exit "Failed to copy mutantsdata"

for file in $MFI_ID"_build.sh"  $MFI_ID"_conf-script.conf" $MFI_ID"_klee-args-template.args"  $MFI_ID"_runtests.sh"  $MFI_ID"_srclist.txt"  $MFI_ID"_testscases.txt"
do
    cp $(dirname $confscript)/$file $projOut/inputs/hpcConfigDir/ || error_exit "Failed to copy $file"
done

# Specific to Coreutils here
mkdir -p $projOut/inputs/hpcConfigDir/repos/$MFI_ID/src || error_exit "failed to make repos src"
touch $projOut/inputs/hpcConfigDir/repos/$MFI_ID/src/$MFI_PROGRAM || error_exit "failed to touch $MFI_PROGRAM"

echo "Done with $MFI_ID!"
