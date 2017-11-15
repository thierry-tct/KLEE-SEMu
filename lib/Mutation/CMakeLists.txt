#===------------------------------------------------------------------------===#
#
#                     The KLEE Symbolic Virtual Machine
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
klee_add_component(kleeSemuCore
  MyCore/AddressSpace.cpp
  MyCore/CallPathManager.cpp
  MyCore/Context.cpp
  MyCore/CoreStats.cpp
  ExecutionState_KS.cpp
  Executor_KS.cpp
  MyCore/ExecutorTimers.cpp
  MyCore/ExecutorUtil.cpp
  MyCore/ExternalDispatcher.cpp
  MyCore/ImpliedValue.cpp
  MyCore/Memory.cpp
  MyCore/MemoryManager.cpp
  MyCore/PTree.cpp
  MyCore/Searcher.cpp
  MyCore/SeedInfo.cpp
  MyCore/SpecialFunctionHandler.cpp
  MyCore/StatsTracker.cpp
  MyCore/TimingSolver.cpp
  MyCore/UserSearcher.cpp
)


# TODO: Work out what the correct LLVM components are for
# kleeSemuCore.
set(LLVM_COMPONENTS
  core
  support
)

if ("${LLVM_PACKAGE_VERSION}" VERSION_EQUAL "3.6" OR
    "${LLVM_PACKAGE_VERSION}" VERSION_GREATER "3.6")
  list(APPEND LLVM_COMPONENTS mcjit executionengine native)
else()
  list(APPEND LLVM_COMPONENTS jit engine)
endif()


klee_get_llvm_libs(LLVM_LIBS ${LLVM_COMPONENTS})
target_link_libraries(kleeSemuCore PUBLIC ${LLVM_LIBS})
target_link_libraries(kleeSemuCore PRIVATE
  kleeBasic
  kleeModule
  kleaverSolver
  kleaverExpr
  kleeSupport
)
