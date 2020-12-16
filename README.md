# KLEE-SEMu Dynamic Symbolic Execution-Assisted Mutation Framework

KLEE-SEMu is a Dynamic Symbolic Execution-Assisted Mutation framework build on top of [KLEE](https://github.com/klee/klee) Symbolic Virtual Machine.

Easily used with the [Muteria](https://github.com/muteria/muteria) Framework and generate test to kill mutants generated by the [Mart](https://github.com/thierry-tct/mart) mutation tool. 

__Docker__: This tool has a Docker image available here [https://hub.docker.com/r/thierrytct/klee-semu](https://hub.docker.com/r/thierrytct/klee-semu).

__Maintenance__: This tool need to be integrated into newer versions of KLEE. The master branch correspond to the master branch of klee. For each use version of klee, the branch `semu-klee-<version>` correspong to SEMu ported to the version `<version>` of klee.
Update the master branch from the [KLEE](https://github.com/klee/klee) repository and update the tags as following (assuming that the git remote `klee` was added as following `git remote add klee https://github.com/klee/klee.git`):
```
git checkout master
git pull klee master
git push

git fetch --tags klee master
git push origin --tags

```

After SEMu is ported to a newer version of klee, set the default branch of the remote SEMu repository to the newest `semu-klee-<version>` branch

=============================
=============================

KLEE Symbolic Virtual Machine
=============================

[![Build Status](https://travis-ci.org/klee/klee.svg?branch=master)](https://travis-ci.org/klee/klee)

`KLEE` is a symbolic virtual machine built on top of the LLVM compiler
infrastructure. Currently, there are two primary components:

  1. The core symbolic virtual machine engine; this is responsible for
     executing LLVM bitcode modules with support for symbolic
     values. This is comprised of the code in lib/.

  2. A POSIX/Linux emulation layer oriented towards supporting uClibc,
     with additional support for making parts of the operating system
     environment symbolic.

Additionally, there is a simple library for replaying computed inputs
on native code (for closed programs). There is also a more complicated
infrastructure for replaying the inputs generated for the POSIX/Linux
emulation layer, which handles running native programs in an
environment that matches a computed test input, including setting up
files, pipes, environment variables, and passing command line
arguments.

Coverage information can be found [here](http://vm-klee.doc.ic.ac.uk:55555/index.html).

For further information, see the [webpage](http://klee.github.io/).
