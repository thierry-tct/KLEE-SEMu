#! /bin/bash

##
# This themplate should be copied into MFI_EXEDIR and the exe file template replaced by MFI_PROGRAM (the executable name)
# Use this with wrappers only, that will call klee replay...
##

set -u

topdir=$(/bin/readlink -f $(/usr/bin/dirname $0))

$topdir/$MFI_PROGRAM

