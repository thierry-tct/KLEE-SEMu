#! /bin/bash
#
# 1. Create a folder <root of repo>/lib/Mutation and copy this file into the folder
# 2. Execute the file from the copied location as following:
#     cd <root of repo>/lib/Mutation && ./UpdateFromKlee.sh
#

error_exit()
{
    echo "Error: $1"
    exit 1
}

here=$(dirname $(readlink -f $0))

rm -rf $here/MyCore

cp -r $here/../Core $here/MyCore || error_exit "Failed to copy Core into MyCore"

mkdir $here/MyCore/klee || error_exit "Failed to create directory MyCore/klee"

echo '#include "../../ExecutionState_KS.h"' > $here/MyCore/klee/ExecutionState.h || error_exit "echo failed for ExecutionState_KS.h"
echo '#include "../Executor_KS.h"' > $here/MyCore/Executor.h || error_exit "echo failed for Executor_KS.h"

# Update lib/Mutation/CMakeLists.txt 
sed 's|\(\S\+\).cpp|myCore/\1.cpp|g;s|myCore/Executor.cpp|Executor_KS.cpp|g;s|myCore/ExecutionState.cpp|ExecutionState_KS.cpp|g' $here/../Core/CMakeLists.txt > $here/CMakeLists.txt 

# Patch lib/CMakeLists.txt
echo "
# @KLEE-SEMu
add_subdirectory(Mutation)
#~KS" >> $here/../CMakeLists.txt


# Patch tools/CMakeLists.txt
echo "
# @KLEE-SEMu
add_subdirectory(klee-semu)
#~KS" >> $here/../../tools/CMakeLists.txt

# Patch tools/klee-semu/CMakeLists.txt file
test -d $here/../../tools/klee-semu || mkdir $here/../../tools/klee-semu || error_exit "failed to create klee-semu dir"
test -f $here/../../tools/klee/CMakeLists.txt || error_exit "CMakeLists.txt missing for tools/klee"
sed 's|klee|klee-semu|g;s|main.cpp|klee-semu.cpp|g;s|kleeCore|kleeSemuCore|g' $here/../../tools/klee/CMakeLists.txt > $here/../../tools/klee-semu/CMakeLists.txt

# Ask to manually update $here/Makefile, $here, $here/../../tools/klee-semu
echo "# DONE!"
echo
echo "# 1. PLEASE Manually update the following files 'Dockerfile', 'ExecutionState_KS.cpp', 'ExecutionState_KS.h', 'Executor_KS.cpp', 'Executor_KS.h', 'README.md', 'configureCMDS' from $here !"
echo
echo "# 2. PLEASE Manually update the file 'klee-semu.cpp' from the folder $(readlink -f $here/../../tools/klee-semu), based on $(readlink -f $here/../../tools/klee/main.cpp) !"
echo

# Patch the bug in ExternalDispatcher.cpp (when 2 SEGV appear, the second is not handled)
#cd $here 
#patch -p3 -i patchesforupdate.patch || error_exit "Failed to apply patch patchesforupdate.patch"

