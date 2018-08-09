#! /bin/bash

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

# Patch the bug in ExternalDispatcher.cpp (when 2 SEGV appear, the second is not handled)
cd $here 
patch -p3 -i patchesforupdate.patch || error_exit "Failed to apply patch patchesforupdate.patch"

