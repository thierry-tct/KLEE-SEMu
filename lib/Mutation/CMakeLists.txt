#===------------------------------------------------------------------------===#
#
#                     The KLEE Symbolic Virtual Machine
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
klee_add_component(kleeCore
  myCore/AddressSpace.cpp
  myCore/MergeHandler.cpp
  myCore/CallPathManager.cpp
  myCore/Context.cpp
  myCore/CoreStats.cpp
  ExecutionState_KS.cpp
  Executor_KS.cpp
  myCore/ExecutorUtil.cpp
  myCore/ExternalDispatcher.cpp
  myCore/ImpliedValue.cpp
  myCore/Memory.cpp
  myCore/MemoryManager.cpp
  myCore/PTree.cpp
  myCore/Searcher.cpp
  myCore/SeedInfo.cpp
  myCore/SpecialFunctionHandler.cpp
  myCore/StatsTracker.cpp
  myCore/TimingSolver.cpp
  myCore/UserSearcher.cpp
)

# TODO: Work out what the correct LLVM components are for
# kleeCore.
set(LLVM_COMPONENTS
  core
  executionengine
  mcjit
  native
  support
)

klee_get_llvm_libs(LLVM_LIBS ${LLVM_COMPONENTS})
target_link_libraries(kleeCore PUBLIC ${LLVM_LIBS} ${SQLITE3_LIBRARIES})
target_link_libraries(kleeCore PRIVATE
  kleeBasic
  kleeModule
  kleaverSolver
  kleaverExpr
  kleeSupport
)
