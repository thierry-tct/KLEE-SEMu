
Run using 'run.py'

[Update this] A more easy to run script (that make call to run.py) is found in example/22/run_cmd

# Issues and Solution running experiments:
1. `LLVM ERROR: not enough shared memory to process testcase`.  This is caused by the use of [_stp_](https://github.com/stp/stp) constraint solver for which klee limit the shared memory size to 2^20. Examples are: `sum`, `csplit` from gnu coreutils.
   
   The solution is to use [_z3_](https://github.com/Z3Prover/z3) as constraing solver instead of [_stp_](https://github.com/stp/stp).
2. `Error while writing ktest and mutant test info`. This is caused by limited number of open files on linux (see `ulimit -n`). Too many files are open when experimenting on some programs such as `cut`, `groups`.
   
   The solution could be to increase the `ulimit` of number of open files using `ulimit n <new limit>` before running the experiment script.
