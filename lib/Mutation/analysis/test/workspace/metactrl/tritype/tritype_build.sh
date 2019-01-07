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

/bin/rm -f `/usr/bin/dirname $0`/$MFI_PROGRAM

[ "$COMPILER" = "wllvm" ] && CFLAGS+=" -std=c89"

if [ "$MODE" = "build" ]
then
    make clean || error_exit "Error: make clean failed. (in $0)"
    make CC=$COMPILER CFLAGS="-Wno-error $CFLAGS" LDFLAGS="$CFLAGS" #|| error_exit "Error: make failed. (in $0)"
elif [ "$MODE" = "clean-make" ]
then
    make clean #|| error_exit "Error: make clean failed. (clean-make in $0)"
    make CC=$COMPILER CFLAGS="$CFLAGS" LDFLAGS="$CFLAGS" #|| error_exit "Error: make failed. (in $0)"
elif [ "$MODE" = "make" ]
then
    make CC=$COMPILER CFLAGS="$CFLAGS" LDFLAGS="$CFLAGS" #-std=gnu99 || error_exit "Error: make failed. (in $0)"
else
    error_exit "Error: Wrong build mode: $MODE. (in $0)"
fi

cd $calldir

# *** wllvm can't generate BC when the extension is not .o, .so,...
### $COMPILER $CFLAGS -c `dirname $0`/src/main.c -o `dirname $0`/src/main.o	#Compile	

### $COMPILER $CFLAGS -o `dirname $0`/src/mainT `dirname $0`/src/main.o	#link

