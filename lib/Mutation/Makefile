#===-- lib/Core/Makefile -----------------------------------*- Makefile -*--===#
#
#                     The KLEE Symbolic Virtual Machine
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

LEVEL=../..

all:: MyCore                                                                   
MyCore::
	test -d $(BuildMode)/MyCore || mkdir -p $(BuildMode)/MyCore && echo "$(BuildMode)/MyCore CREATED"

SOURCES=MyCore/UserSearcher.cpp \
MyCore/ExecutorUtil.cpp \
MyCore/ExternalDispatcher.cpp \
MyCore/ExecutorTimers.cpp \
MyCore/StatsTracker.cpp \
MyCore/CoreStats.cpp \
MyCore/TimingSolver.cpp \
MyCore/AddressSpace.cpp \
MyCore/Searcher.cpp \
MyCore/CallPathManager.cpp \
MyCore/Context.cpp \
MyCore/SpecialFunctionHandler.cpp \
MyCore/MemoryManager.cpp \
MyCore/SeedInfo.cpp \
MyCore/PTree.cpp \
MyCore/ImpliedValue.cpp \
MyCore/Memory.cpp \
Executor_KS.cpp \
ExecutionState_KS.cpp

LIBRARYNAME=kleeSemuCore
DONT_BUILD_RELINKED=1
BUILD_ARCHIVE=1
NO_INSTALL=1

include $(LEVEL)/Makefile.common
