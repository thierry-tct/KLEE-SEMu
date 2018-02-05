#!/bin/bash
# Build script for (1) rm
#Note: Make sure that here, all used tool are refered with absolute path (to avoid a control command to be considered as a test)
set -u
error_exit()
{
    /bin/echo $1
    exit 1
}

[ $# = 3 ] || error_exit "Error: build script expected 3 parameters, $# provided: <$0> <compiler> <CFLAGS> <mode: [ build(make distclean,bootstrap, config, make); clean-make(make clean, make); make]>"
COMPILER=$1
CFLAGS=$2
MODE=$3
rootDir=$MFI_ROOTDIR

calldir=`/bin/pwd`
cd $rootDir

/bin/rm -f $MFI_EXEDIR/$MFI_PROGRAM


if [ "$MODE" = "build" ]
then
    make clean && make distclean #|| error_exit "Error: make distclean failed. (in $0)"
    ./configure --disable-nls CC=$COMPILER CFLAGS="-Wno-error -std=c99 $CFLAGS" || error_exit "Error: configure failed. (in $0)"
    #repair Makefile...
    echo "all: ;" > doc/Makefile
    echo "all: ;" > po/Makefile
    # Add optimiser script's test  log entry into tests' makefile
    testcaseDir="tests/expr"
    cat $(dirname $0)/mfi_ktest-replay-optimizer.testlog >> $rootDir/$testcaseDir/Makefile || error_exit "Failed to add optimizer test log. (in $0)"
    make CC=$COMPILER CFLAGS="-Wno-error -std=c99 $CFLAGS" #|| error_exit "Error: make failed. (in $0)"
elif [ "$MODE" = "clean-make" ]
then
    cd src && make clean; cd - #|| error_exit "Error: make clean failed. (clean-make in $0)"
    make CC=$COMPILER CFLAGS="-std=c99 $CFLAGS" #|| error_exit "Error: make failed. (in $0)"
elif [ "$MODE" = "make" ]
then
    cd src
    make CC=$COMPILER CFLAGS="-std=c99 $CFLAGS" expr #-std=gnu99 || error_exit "Error: make failed. (in $0)"
    cd -
else
    error_exit "Error: Wrong build mode: $MODE. (in $0)"
fi

cd $calldir

# *** wllvm can't generate BC when the extension is not .o, .so,...
### $COMPILER $CFLAGS -c `dirname $0`/src/main.c -o `dirname $0`/src/main.o	#Compile	

### $COMPILER $CFLAGS -o `dirname $0`/src/mainT `dirname $0`/src/main.o	#link

