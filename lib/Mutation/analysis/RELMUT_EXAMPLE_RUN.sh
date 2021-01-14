#! /bin/bash
# Ensure that semu is compile for relevan mutants by setting -DSEMU_RELMUT_PRED_ENABLED
# run from within SEMu's Docker container

set -u

error_exit()
{
        echo "error: $1"
        exit 1
}

topdir=$(dirname $(readlink -f $0))

rm -rf $topdir/klee-out-*

cd $topdir

code_file_name=RELMUT_EXAMPLE_SRC

clang -c -g -emit-llvm $code_file_name.c -o $code_file_name.bc || error_exit "build failed"

postmut=''
#postmut='--semu-disable-post-mutation-check'

critical=''
#critical='--semu-testsgen-only-for-critical-diffs'

klee-semu $postmut $critical \
  --allow-external-sym-calls \
  --posix-runtime \
  --semu-no-error-on-memory-limit \
  --solver-backend z3 \
  --max-memory 8048 \
  --max-time 300 \
  --libc uclibc \
  --search bfs \
  --semu-precondition-length 0 \
  --semu-checkpoint-window 100000 \
  --semu-propagation-proportion 0 \
  --semu-minimum-propagation-depth 0 \
  --semu-number-of-tests-per-mutant 10 \
  $code_file_name.bc --sym-arg 1
