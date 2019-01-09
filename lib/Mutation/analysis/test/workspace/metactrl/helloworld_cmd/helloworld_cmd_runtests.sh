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

#optimization of running time
if /bin/echo $TC | /bin/grep '\*\*MFI\*\*OPTIMIZE\*\*' > /dev/null
then
    exit 1
fi        
 
cd $rootDir

fail=0

#./$MFI_PROGRAM < $TC 
./$MFI_PROGRAM $(cat $TC) 

cd - > /dev/null

exit $fail
