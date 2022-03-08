//===-- Executor.cpp ------------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// @KLEE-SEMu
#include "Executor_KS.h"
#include "../Core/Context.h"
#include "../Core/CoreStats.h"
#include "../Core/ExternalDispatcher.h"
#include "../Core/ImpliedValue.h"
#include "../Core/Memory.h"
#include "../Core/MemoryManager.h"
#include "../Core/PTree.h"
#include "../Core/Searcher.h"
#include "../Core/SeedInfo.h"
#include "../Core/SpecialFunctionHandler.h"
#include "../Core/StatsTracker.h"
#include "../Core/TimingSolver.h"
#include "../Core/UserSearcher.h"
#include "../Core/ExecutorTimerInfo.h"


#include "ExecutionState_KS.h"

#include "expr/Lexer.h"
#include "expr/Parser.h"
#include "klee/ExprBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CFG.h"
//#include "llvm/IR/IRBuilder.h"
//#include "llvm/Support/FileSystem.h"
#include <unistd.h>
#include <sys/wait.h>
//~KS

#include "klee/Expr.h"
#include "klee/Interpreter.h"
#include "klee/TimerStatIncrementer.h"
#include "klee/CommandLine.h"
#include "klee/Common.h"
#include "klee/util/Assignment.h"
#include "klee/util/ExprPPrinter.h"
#include "klee/util/ExprSMTLIBPrinter.h"
#include "klee/util/ExprUtil.h"
#include "klee/util/GetElementPtrTypeIterator.h"
#include "klee/Config/Version.h"
#include "klee/Internal/ADT/KTest.h"
#include "klee/Internal/ADT/RNG.h"
#include "klee/Internal/Module/Cell.h"
#include "klee/Internal/Module/InstructionInfoTable.h"
#include "klee/Internal/Module/KInstruction.h"
#include "klee/Internal/Module/KModule.h"
#include "klee/Internal/Support/ErrorHandling.h"
#include "klee/Internal/Support/FloatEvaluation.h"
#include "klee/Internal/System/Time.h"
#include "klee/Internal/System/MemoryUsage.h"
#include "klee/SolverStats.h"

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
#include "llvm/IR/Function.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/TypeBuilder.h"
#else
#include "llvm/Attributes.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#if LLVM_VERSION_CODE <= LLVM_VERSION(3, 1)
#include "llvm/Target/TargetData.h"
#else
#include "llvm/DataLayout.h"
#include "llvm/TypeBuilder.h"
#endif
#endif
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#if LLVM_VERSION_CODE < LLVM_VERSION(3, 5)
#include "llvm/Support/CallSite.h"
#else
#include "llvm/IR/CallSite.h"
#endif

#ifdef HAVE_ZLIB_H
#include "klee/Internal/Support/CompressionStream.h"
#endif

#include <cassert>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

#include <sys/mman.h>

#include <errno.h>
#include <cxxabi.h>

using namespace llvm;
using namespace klee;



// @KLEE-SEMu
namespace {

/**** SEMu Main Parameters ****/

// precondition Length value is fixed if >= 0; stop when any state finds a mutants and set to that depth if == -1; stop when each state reach a mutant if < -1
cl::opt<int> semuPreconditionLength("semu-precondition-length", 
                                 cl::init(6), 
                                 cl::desc("(PL) default=6. Sets number of pre-conditions that will be taken from the seeds' (initial tests) path conditions and used as precondition. Use the two negative values: -1 for Global Minimum Distance to Mutated Statement (GMD2MS), and -2 for Specific Minimum Distance to Mutated Statement (SMD2MS)."));

// optional watch point max depth to leverage mutant state explosion
// When the value of semuMaxDepthWP is 0, check right after the mutation point (similar to waek mutation)
// When not semuMaxDepthWP value v != 0, we check all mutants when the depth is k*v and destroy all mutant states seen so far
// TODO: Add this to ks_reachedCheckMaxDepth
cl::opt<unsigned> semuMaxDepthWP("semu-checkpoint-window", 
                                 cl::init(0), 
                                 cl::desc("(CW) default=0. Maximum length of mutant path condition from mutation point to watch point (number of fork locations since mutation point)"));

// TODO implement different selection strategies on which to continue
cl::opt<double> semuPostCheckpointMutantStateContinueProba("semu-propagation-proportion", 
                                 cl::init(0.0), 
                                 cl::desc("(PP) default=0. Set the proportion of mutant states of a particular mutant that will continue after checkpoint(checkpoint postponing). For a given mutant ID, will see the states that reach checkpoint and remove the specified proportion to continue past the checkpoint."));

cl::opt<bool> semuApplyMinDistToOutputForMutContinue(
                                "semu-MDO-propagation-selection-strategy",
                                 cl::init(false),
                                 cl::desc("(PSS-MDO) Enable using the distance to the output to select which states to continue post mutant checkpoint"));

cl::opt<unsigned> semuGenTestForDiscardedFromCheckNum(
                                 "semu-minimum-propagation-depth",
                                 cl::init(0),
                                 cl::desc("(MPD) default=0. Number of checkpoints where mutants states that reach are discarded without test generated for them, before a test is generated. Help to generate test only for the states, for a mutant, that goes deep"));

cl::opt<bool> semuDisableStateDiffInTestGen("semu-no-state-difference",
                                 cl::init(false),
                                 cl::desc("(NSD) Disable the state comparison with original when generating test for mutant (only consider mutant PC)"));

cl::opt<unsigned> semuMaxNumTestGenPerMutant("semu-number-of-tests-per-mutant", 
                                 cl::init(0), 
                                 cl::desc("(NTPM) default=0. Set the maximum number of test generated to kill each mutant. (== 0) and semu-max-total-tests-gen=0 means do not generate test, only get state difference. Either (> 0) means generate test, do not get state differences. default is 0"));

/***** SEMu Auxiliary parameters *****/

cl::opt<bool> semuNoConsiderOutEnvForDiffs("semu-no-environment-output-diff", 
                                          cl::init(false), 
                                          cl::desc("Disable checking environment output calls parameters when checking for state differences."));
	
cl::opt<bool> semuUseBBForDistance("semu-use-basicblock-for-distance",
                                 cl::init(false),
                                 cl::desc("Enable using number of Basic blocks instead of number of instructions for distance (for Minimum Distance to Output - MDO strategy)"));

cl::opt<unsigned> semuMaxTotalNumTestGen("semu-max-total-tests-gen", 
                                 cl::init(0), 
                                 cl::desc("default=0. Set the maximum number of test generated to kill the mutants. (== 0) and semu-max-tests-gen-per-mutant=0 means do not generate test, only get state difference. Either (> 0) means generate test, do not get state differences. default is 0"));

// Optionally set the list of mutants to consider, the remaining will be removed from meta module
cl::opt<std::string> semuCandidateMutantsFile("semu-candidate-mutants-list-file", 
                                  cl::init(""),
                                  cl::desc("File containing the subset  list of mutants to consider in the analysis"));

cl::opt<bool> semuUseOnlyBranchForDepth("semu-use-only-multi-branching-for-depth",
                                          cl::init(false),
                                          cl::desc("Enable to use only symbolic state branching (when both sides are feasible) to measure the depth. Otherwise, one side fork are also used. This reduces the accuracy of the depth measure"));

cl::opt<bool> semuForkProcessForSegvExternalCalls("semu-forkprocessfor-segv-externalcalls",
                                          cl::init(false),
                                          cl::desc("Enable forking a new process for external call to 'printf' which can lead to Segmentation Fault."));

// TODO, actually detect loop and limit the number of iterations. For now just limit the time. set this time large enough to capture loop, not simple instructions
cl::opt<double> semuLoopBreakDelay("semu-loop-break-delay", 
                                 cl::init(120.0), 
                                 cl::desc("default=120. Set the maximum time in seconds that the same state is executed without stop"));

cl::opt<bool> semuDisableCheckAtPostMutationPoint("semu-disable-post-mutation-check",
                                          cl::init(false),
                                          cl::desc("Setting this will disable the check post mutation point for state diff. This is particularly needed for higher order mutants, where the diff can happend in a different component."));

cl::opt<bool> semuTestgenOnlyForCriticalDiffs("semu-testsgen-only-for-critical-diffs", 
                                          cl::init(false), 
                                          cl::desc("Enable Outputting tests only for critial diffs (involving environment (this excludes local/global vars))"));

cl::opt<bool> semuEnableNoErrorOnMemoryLimit("semu-no-error-on-memory-limit",
                                 cl::init(false),
                                 cl::desc("Enable no error  on memory limit. This mean that states can be remove uncontrollabl and m worsen the effectiveness of SEMu"));
	
cl::opt<bool> semuQuiet("semu-quiet",
                                 cl::init(false),
                                 cl::desc("Enable quiet log"));
	
cl::list<std::string> semuCustomOutEnvFunction("semu-custom-output-function",
					       cl::desc("Specify the functions to consider as output functions, besides the standard ones."));
                                      
cl::opt<bool> semuNoCompareMemoryLimitDiscarded("semu-no-compare-memory-limit-discarded",
                                 cl::init(false),
                                 cl::desc("(TODO) Disable comparison of states that were stopped due to memory limit (useful to ensure the the memory limit is kept when large checkpoint window)"));

/**** SEMu Under development ****/
// Use shadow test case generation for mutants ()
cl::opt<bool> semuShadowTestGeneration("semu-shadow-test-gen", 
                                          cl::init(false), 
                                          cl::desc("(FIXME: DO NOT USE THIS) Enable Test generation using the shadow SE based approach"));

// Automatically set the arguments of the entry function symbolic
cl::opt<bool> semuSetEntryFuncArgsSymbolic("semu-set-entyfunction-args-symbolic", 
                                          cl::init(false), 
                                          cl::desc("(FIXME: DO NOT USE THIS - ONGOING) Enable automatically set the parameters of the entry point symbolic"));

// Optional file containing the precondition of symbolic execution, maybe extracted from existing test using Zesti
cl::list<std::string> semuPrecondFiles("semu-precondition-file", 
                                  cl::desc("(FIXME: DO NOT USE THIS) precondition for bounded semu (use this many times for multiple files)"));

}
//~ KS


namespace {
  cl::opt<bool>
  DumpStatesOnHalt("dump-states-on-halt",
                   cl::init(true),
		   cl::desc("Dump test cases for all active states on exit (default=on)"));
  
  cl::opt<bool>
  AllowExternalSymCalls("allow-external-sym-calls",
                        cl::init(false),
			cl::desc("Allow calls with symbolic arguments to external functions.  This concretizes the symbolic arguments.  (default=off)"));

  /// The different query logging solvers that can switched on/off
  enum PrintDebugInstructionsType {
    STDERR_ALL, ///
    STDERR_SRC,
    STDERR_COMPACT,
    FILE_ALL,    ///
    FILE_SRC,    ///
    FILE_COMPACT ///
  };

  llvm::cl::list<PrintDebugInstructionsType> DebugPrintInstructions(
      "debug-print-instructions",
      llvm::cl::desc("Log instructions during execution."),
      llvm::cl::values(
          clEnumValN(STDERR_ALL, "all:stderr", "Log all instructions to stderr "
                                               "in format [src, inst_id, "
                                               "llvm_inst]"),
          clEnumValN(STDERR_SRC, "src:stderr",
                     "Log all instructions to stderr in format [src, inst_id]"),
          clEnumValN(STDERR_COMPACT, "compact:stderr",
                     "Log all instructions to stderr in format [inst_id]"),
          clEnumValN(FILE_ALL, "all:file", "Log all instructions to file "
                                           "instructions.txt in format [src, "
                                           "inst_id, llvm_inst]"),
          clEnumValN(FILE_SRC, "src:file", "Log all instructions to file "
                                           "instructions.txt in format [src, "
                                           "inst_id]"),
          clEnumValN(FILE_COMPACT, "compact:file",
                     "Log all instructions to file instructions.txt in format "
                     "[inst_id]"),
          clEnumValEnd),
      llvm::cl::CommaSeparated);
#ifdef HAVE_ZLIB_H
  cl::opt<bool> DebugCompressInstructions(
      "debug-compress-instructions", cl::init(false),
      cl::desc("Compress the logged instructions in gzip format."));
#endif

  cl::opt<bool>
  DebugCheckForImpliedValues("debug-check-for-implied-values");


  cl::opt<bool>
  SimplifySymIndices("simplify-sym-indices",
                     cl::init(false),
		     cl::desc("Simplify symbolic accesses using equalities from other constraints (default=off)"));

  cl::opt<bool>
  EqualitySubstitution("equality-substitution",
		       cl::init(true),
		       cl::desc("Simplify equality expressions before querying the solver (default=on)."));

  cl::opt<unsigned>
  MaxSymArraySize("max-sym-array-size",
                  cl::init(0));

  cl::opt<bool>
  SuppressExternalWarnings("suppress-external-warnings",
			   cl::init(false),
			   cl::desc("Supress warnings about calling external functions."));

  cl::opt<bool>
  AllExternalWarnings("all-external-warnings",
		      cl::init(false),
		      cl::desc("Issue an warning everytime an external call is made," 
			       "as opposed to once per function (default=off)"));

  cl::opt<bool>
  OnlyOutputStatesCoveringNew("only-output-states-covering-new",
                              cl::init(false),
			      cl::desc("Only output test cases covering new code (default=off)."));

  cl::opt<bool>
  EmitAllErrors("emit-all-errors",
                cl::init(false),
                cl::desc("Generate tests cases for all errors "
                         "(default=off, i.e. one per (error,instruction) pair)"));
  
  cl::opt<bool>
  NoExternals("no-externals", 
           cl::desc("Do not allow external function calls (default=off)"));

  cl::opt<bool>
  AlwaysOutputSeeds("always-output-seeds",
		    cl::init(true));

  cl::opt<bool>
  OnlyReplaySeeds("only-replay-seeds",
		  cl::init(false),
                  cl::desc("Discard states that do not have a seed (default=off)."));
 
  cl::opt<bool>
  OnlySeed("only-seed",
	   cl::init(false),
           cl::desc("Stop execution after seeding is done without doing regular search (default=off)."));
 
  cl::opt<bool>
  AllowSeedExtension("allow-seed-extension",
		     cl::init(false),
                     cl::desc("Allow extra (unbound) values to become symbolic during seeding (default=false)."));
 
  cl::opt<bool>
  ZeroSeedExtension("zero-seed-extension",
		    cl::init(false),
		    cl::desc("(default=off)"));
 
  cl::opt<bool>
  AllowSeedTruncation("allow-seed-truncation",
		      cl::init(false),
                      cl::desc("Allow smaller buffers than in seeds (default=off)."));
 
  cl::opt<bool>
  NamedSeedMatching("named-seed-matching",
		    cl::init(false),
                    cl::desc("Use names to match symbolic objects to inputs (default=off)."));

  cl::opt<double>
  MaxStaticForkPct("max-static-fork-pct", 
		   cl::init(1.),
		   cl::desc("(default=1.0)"));

  cl::opt<double>
  MaxStaticSolvePct("max-static-solve-pct",
		    cl::init(1.),
		    cl::desc("(default=1.0)"));

  cl::opt<double>
  MaxStaticCPForkPct("max-static-cpfork-pct", 
		     cl::init(1.),
		     cl::desc("(default=1.0)"));

  cl::opt<double>
  MaxStaticCPSolvePct("max-static-cpsolve-pct",
		      cl::init(1.),
		      cl::desc("(default=1.0)"));

  cl::opt<double>
  MaxInstructionTime("max-instruction-time",
                     cl::desc("Only allow a single instruction to take this much time (default=0s (off)). Enables --use-forked-solver"),
                     cl::init(0));
  
  cl::opt<double>
  SeedTime("seed-time",
           cl::desc("Amount of time to dedicate to seeds, before normal search (default=0 (off))"),
           cl::init(0));
  
  cl::list<Executor::TerminateReason>
  ExitOnErrorType("exit-on-error-type",
		  cl::desc("Stop execution after reaching a specified condition.  (default=off)"),
		  cl::values(
		    clEnumValN(Executor::Abort, "Abort", "The program crashed"),
		    clEnumValN(Executor::Assert, "Assert", "An assertion was hit"),
		    clEnumValN(Executor::Exec, "Exec", "Trying to execute an unexpected instruction"),
		    clEnumValN(Executor::External, "External", "External objects referenced"),
		    clEnumValN(Executor::Free, "Free", "Freeing invalid memory"),
		    clEnumValN(Executor::Model, "Model", "Memory model limit hit"),
		    clEnumValN(Executor::Overflow, "Overflow", "An overflow occurred"),
		    clEnumValN(Executor::Ptr, "Ptr", "Pointer error"),
		    clEnumValN(Executor::ReadOnly, "ReadOnly", "Write to read-only memory"),
		    clEnumValN(Executor::ReportError, "ReportError", "klee_report_error called"),
		    clEnumValN(Executor::User, "User", "Wrong klee_* functions invocation"),
		    clEnumValN(Executor::Unhandled, "Unhandled", "Unhandled instruction hit"),
		    clEnumValEnd),
		  cl::ZeroOrMore);

  cl::opt<unsigned int>
  StopAfterNInstructions("stop-after-n-instructions",
                         cl::desc("Stop execution after specified number of instructions (default=0 (off))"),
                         cl::init(0));
  
  cl::opt<unsigned>
  MaxForks("max-forks",
           cl::desc("Only fork this many times (default=-1 (off))"),
           cl::init(~0u));
  
  cl::opt<unsigned>
  MaxDepth("max-depth",
           cl::desc("Only allow this many symbolic branches (default=0 (off))"),
           cl::init(0));
  
  cl::opt<unsigned>
  MaxMemory("max-memory",
            cl::desc("Refuse to fork when above this amount of memory (in MB, default=2000)"),
            cl::init(2000));

  cl::opt<bool>
  MaxMemoryInhibit("max-memory-inhibit",
            cl::desc("Inhibit forking at memory cap (vs. random terminate) (default=on)"),
            cl::init(true));
}


namespace klee {
  RNG theRNG;
}

const char *Executor::TerminateReasonNames[] = {
  [ Abort ] = "abort",
  [ Assert ] = "assert",
  [ Exec ] = "exec",
  [ External ] = "external",
  [ Free ] = "free",
  [ Model ] = "model",
  [ Overflow ] = "overflow",
  [ Ptr ] = "ptr",
  [ ReadOnly ] = "readonly",
  [ ReportError ] = "reporterror",
  [ User ] = "user",
  [ Unhandled ] = "xxx",
};

Executor::Executor(const InterpreterOptions &opts, InterpreterHandler *ih)
    : Interpreter(opts), kmodule(0), interpreterHandler(ih), searcher(0),
      externalDispatcher(new ExternalDispatcher()), statsTracker(0),
      pathWriter(0), symPathWriter(0), specialFunctionHandler(0),
      processTree(0), replayKTest(0), replayPath(0), usingSeeds(0),
      atMemoryLimit(false), inhibitForking(false), haltExecution(false),
      ivcEnabled(false),
      coreSolverTimeout(MaxCoreSolverTime != 0 && MaxInstructionTime != 0
                            ? std::min(MaxCoreSolverTime, MaxInstructionTime)
                            : std::max(MaxCoreSolverTime, MaxInstructionTime)),
      debugInstFile(0), debugLogBuffer(debugBufferString) {

  if (coreSolverTimeout) UseForkedCoreSolver = true;
  Solver *coreSolver = klee::createCoreSolver(CoreSolverToUse);
  if (!coreSolver) {
    klee_error("Failed to create core solver\n");
  }

  Solver *solver = constructSolverChain(
      coreSolver,
      interpreterHandler->getOutputFilename(ALL_QUERIES_SMT2_FILE_NAME),
      interpreterHandler->getOutputFilename(SOLVER_QUERIES_SMT2_FILE_NAME),
      interpreterHandler->getOutputFilename(ALL_QUERIES_KQUERY_FILE_NAME),
      interpreterHandler->getOutputFilename(SOLVER_QUERIES_KQUERY_FILE_NAME));

  this->solver = new TimingSolver(solver, EqualitySubstitution);
  memory = new MemoryManager(&arrayCache);

  if (optionIsSet(DebugPrintInstructions, FILE_ALL) ||
      optionIsSet(DebugPrintInstructions, FILE_COMPACT) ||
      optionIsSet(DebugPrintInstructions, FILE_SRC)) {
    std::string debug_file_name =
        interpreterHandler->getOutputFilename("instructions.txt");
    std::string ErrorInfo;
#ifdef HAVE_ZLIB_H
    if (!DebugCompressInstructions) {
#endif

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 5)
    debugInstFile = new llvm::raw_fd_ostream(debug_file_name.c_str(), ErrorInfo,
                                             llvm::sys::fs::OpenFlags::F_Text),
#else
    debugInstFile =
        new llvm::raw_fd_ostream(debug_file_name.c_str(), ErrorInfo);
#endif
#ifdef HAVE_ZLIB_H
    } else {
      debugInstFile = new compressed_fd_ostream(
          (debug_file_name + ".gz").c_str(), ErrorInfo);
    }
#endif
    if (ErrorInfo != "") {
      klee_error("Could not open file %s : %s", debug_file_name.c_str(),
                 ErrorInfo.c_str());
    }
  }
}


const Module *Executor::setModule(llvm::Module *module, 
                                  const ModuleOptions &opts) {
  assert(!kmodule && module && "can only register one module"); // XXX gross
  
  // @KLEE-SEMu
  ExecutionState::ks_setMode(semuShadowTestGeneration ? ExecutionState::KS_Mode::TESTGEN_MODE: ExecutionState::KS_Mode::SEMU_MODE);
  ks_outputTestsCases = (semuMaxNumTestGenPerMutant > 0 || semuMaxTotalNumTestGen > 0);
  ks_outputStateDiffs = !ks_outputTestsCases;

  // Handle case where only one of semuMaxTotalNumTestGen and semuMaxNumTestGenPerMutant is > 0
  if (ks_outputTestsCases) {
    if (semuMaxTotalNumTestGen == 0)
      semuMaxTotalNumTestGen = (unsigned) -1;  // Very high value
    else // semuMaxNumTestGenPerMutant == 0
      semuMaxNumTestGenPerMutant = (unsigned) -1; // Very high value
  }

  if (semuSetEntryFuncArgsSymbolic) {
    ks_entryFunction = module->getFunction(opts.EntryPoint);
    ks_setInitialSymbolics (*module, *ks_entryFunction);   
  }
  ks_mutantIDSelectorGlobal = module->getNamedGlobal(ks_mutantIDSelectorName);
  if (!ks_mutantIDSelectorGlobal) {
    assert (false && "@KLEE-SEMu - ERROR: The module is unmutated(no mutant ID selector global var)");
    klee_error ("@KLEE-SEMu - ERROR: The module is unmutated(no mutant ID selector global var)");
  }
  // Not needed, it has value the number of mutants + 1 Make sure that the value of the mutIDSelector global variable is 0 (original)
  if (! ks_mutantIDSelectorGlobal->hasInitializer() ) {
          //&& ks_mutantIDSelectorGlobal->getInitializer()->isNullValue()
    assert (false && "@KLEE-SEMu - ERROR: mutant ID selector Must be initialized to 0!");
    klee_error ("@KLEE-SEMu - ERROR: mutant ID selector Must be initialized to 0!");
  }
          
  ks_mutantIDSelectorGlobal_Func = module->getFunction(ks_mutantIDSelectorName_Func);
  if (!ks_mutantIDSelectorGlobal_Func || ks_mutantIDSelectorGlobal_Func->arg_size() != 2) {
    assert (false && "@KLEE-SEMu - ERROR: The module is missing mutant selector Function");
    klee_error ("@KLEE-SEMu - ERROR: The module is missing mutant selector Function");
  }
  ks_postMutationPoint_Func = module->getFunction(ks_postMutationPointFuncName);
  if (!semuDisableCheckAtPostMutationPoint && (!ks_postMutationPoint_Func || ks_postMutationPoint_Func->arg_size() != 2)) {
    assert (false && "@KLEE-SEMu - ERROR: The module is missing post mutation point function");
    klee_error ("@KLEE-SEMu - ERROR: The module is missing post mutation point function");
  }

  // Add custom outenv
  for (auto it=semuCustomOutEnvFunction.begin(), ie=semuCustomOutEnvFunction.end(); it != ie; ++it)
    ks_customOutEnvFuncNames.insert(*it);
  
  ks_FilterMutants(module);
  ks_initialize_ks_instruction2closestout_distance(module);
#ifdef SEMU_RELMUT_PRED_ENABLED
  ks_odlNewPrepareModule(module);
#endif
  //~KS
  
  kmodule = new KModule(module);

  // Initialize the context.
#if LLVM_VERSION_CODE <= LLVM_VERSION(3, 1)
  TargetData *TD = kmodule->targetData;
#else
  DataLayout *TD = kmodule->targetData;
#endif
  Context::initialize(TD->isLittleEndian(),
                      (Expr::Width) TD->getPointerSizeInBits());

  specialFunctionHandler = new SpecialFunctionHandler(*this);

  specialFunctionHandler->prepare();
  kmodule->prepare(opts, interpreterHandler);
  specialFunctionHandler->bind();

  if (StatsTracker::useStatistics() || userSearcherRequiresMD2U()) {
    statsTracker = 
      new StatsTracker(*this,
                       interpreterHandler->getOutputFilename("assembly.ll"),
                       userSearcherRequiresMD2U());
  }
  
  // @KLEE-SEMu
  // Reverify since module was changed
  ks_mutantIDSelectorGlobal = module->getNamedGlobal(ks_mutantIDSelectorName);
  if (!ks_mutantIDSelectorGlobal) {
    assert (false && "@KLEE-SEMu - ERROR: The module is unmutated(no mutant ID selector global var)");
    klee_error ("@KLEE-SEMu - ERROR: The module is unmutated(no mutant ID selector global var)");
  }
  //Make sure that the value of the mutIDSelector global variable is 0 (original)
  if (!ks_mutantIDSelectorGlobal->hasInitializer()) {
          //&& ks_mutantIDSelectorGlobal->getInitializer()->isNullValue()
    assert (false && "@KLEE-SEMu - ERROR: mutant ID selector Must be initialized to 0!");
    klee_error ("@KLEE-SEMu - ERROR: mutant ID selector Must be initialized to 0!");
  }
          
  ks_mutantIDSelectorGlobal_Func = module->getFunction(ks_mutantIDSelectorName_Func);
  if (!ks_mutantIDSelectorGlobal_Func || ks_mutantIDSelectorGlobal_Func->arg_size() != 2) {
    assert (false && "@KLEE-SEMu - ERROR: The module is missing mutant selector Function");
    klee_error ("@KLEE-SEMu - ERROR: The module is missing mutant selector Function");
  }
  ks_postMutationPoint_Func = module->getFunction(ks_postMutationPointFuncName);
  if (!semuDisableCheckAtPostMutationPoint && (!ks_postMutationPoint_Func || ks_postMutationPoint_Func->arg_size() != 2)) {
    assert (false && "@KLEE-SEMu - ERROR: The module is missing post mutation point function");
    klee_error ("@KLEE-SEMu - ERROR: The module is missing post mutation point function");
  }
  //~KS
  
  return module;
}

Executor::~Executor() {
  delete memory;
  delete externalDispatcher;
  if (processTree)
    delete processTree;
  if (specialFunctionHandler)
    delete specialFunctionHandler;
  if (statsTracker)
    delete statsTracker;
  delete solver;
  delete kmodule;
  while(!timers.empty()) {
    delete timers.back();
    timers.pop_back();
  }
  if (debugInstFile) {
    delete debugInstFile;
  }
}

/***/

void Executor::initializeGlobalObject(ExecutionState &state, ObjectState *os,
                                      const Constant *c, 
                                      unsigned offset) {
#if LLVM_VERSION_CODE <= LLVM_VERSION(3, 1)
  TargetData *targetData = kmodule->targetData;
#else
  DataLayout *targetData = kmodule->targetData;
#endif
  if (const ConstantVector *cp = dyn_cast<ConstantVector>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(cp->getType()->getElementType());
    for (unsigned i=0, e=cp->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, cp->getOperand(i), 
			     offset + i*elementSize);
  } else if (isa<ConstantAggregateZero>(c)) {
    unsigned i, size = targetData->getTypeStoreSize(c->getType());
    for (i=0; i<size; i++)
      os->write8(offset+i, (uint8_t) 0);
  } else if (const ConstantArray *ca = dyn_cast<ConstantArray>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(ca->getType()->getElementType());
    for (unsigned i=0, e=ca->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, ca->getOperand(i), 
			     offset + i*elementSize);
  } else if (const ConstantStruct *cs = dyn_cast<ConstantStruct>(c)) {
    const StructLayout *sl =
      targetData->getStructLayout(cast<StructType>(cs->getType()));
    for (unsigned i=0, e=cs->getNumOperands(); i != e; ++i)
      initializeGlobalObject(state, os, cs->getOperand(i), 
			     offset + sl->getElementOffset(i));
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
  } else if (const ConstantDataSequential *cds =
               dyn_cast<ConstantDataSequential>(c)) {
    unsigned elementSize =
      targetData->getTypeStoreSize(cds->getElementType());
    for (unsigned i=0, e=cds->getNumElements(); i != e; ++i)
      initializeGlobalObject(state, os, cds->getElementAsConstant(i),
                             offset + i*elementSize);
#endif
  } else if (!isa<UndefValue>(c)) {
    unsigned StoreBits = targetData->getTypeStoreSizeInBits(c->getType());
    ref<ConstantExpr> C = evalConstant(c);

    // Extend the constant if necessary;
    assert(StoreBits >= C->getWidth() && "Invalid store size!");
    if (StoreBits > C->getWidth())
      C = C->ZExt(StoreBits);

    os->write(offset, C);
  }
}

MemoryObject * Executor::addExternalObject(ExecutionState &state, 
                                           void *addr, unsigned size, 
                                           bool isReadOnly) {
  MemoryObject *mo = memory->allocateFixed((uint64_t) (unsigned long) addr, 
                                           size, 0);
  ObjectState *os = bindObjectInState(state, mo, false);
  for(unsigned i = 0; i < size; i++)
    os->write8(i, ((uint8_t*)addr)[i]);
  if(isReadOnly)
    os->setReadOnly(true);  
  return mo;
}


extern void *__dso_handle __attribute__ ((__weak__));

void Executor::initializeGlobals(ExecutionState &state) {
  Module *m = kmodule->module;

  if (m->getModuleInlineAsm() != "")
    klee_warning("executable has module level assembly (ignoring)");
#if LLVM_VERSION_CODE < LLVM_VERSION(3, 3)
  assert(m->lib_begin() == m->lib_end() &&
         "XXX do not support dependent libraries");
#endif
  // represent function globals using the address of the actual llvm function
  // object. given that we use malloc to allocate memory in states this also
  // ensures that we won't conflict. we don't need to allocate a memory object
  // since reading/writing via a function pointer is unsupported anyway.
  for (Module::iterator i = m->begin(), ie = m->end(); i != ie; ++i) {
    Function *f = i;
    ref<ConstantExpr> addr(0);

    // If the symbol has external weak linkage then it is implicitly
    // not defined in this module; if it isn't resolvable then it
    // should be null.
    if (f->hasExternalWeakLinkage() && 
        !externalDispatcher->resolveSymbol(f->getName())) {
      addr = Expr::createPointer(0);
    } else {
      addr = Expr::createPointer((unsigned long) (void*) f);
      legalFunctions.insert((uint64_t) (unsigned long) (void*) f);
    }
    
    globalAddresses.insert(std::make_pair(f, addr));
  }

  // Disabled, we don't want to promote use of live externals.
#ifdef HAVE_CTYPE_EXTERNALS
#ifndef WINDOWS
#ifndef DARWIN
  /* From /usr/include/errno.h: it [errno] is a per-thread variable. */
  int *errno_addr = __errno_location();
  addExternalObject(state, (void *)errno_addr, sizeof *errno_addr, false);

  /* from /usr/include/ctype.h:
       These point into arrays of 384, so they can be indexed by any `unsigned
       char' value [0,255]; by EOF (-1); or by any `signed char' value
       [-128,-1).  ISO C requires that the ctype functions work for `unsigned */
  const uint16_t **addr = __ctype_b_loc();
  addExternalObject(state, const_cast<uint16_t*>(*addr-128),
                    384 * sizeof **addr, true);
  addExternalObject(state, addr, sizeof(*addr), true);
    
  const int32_t **lower_addr = __ctype_tolower_loc();
  addExternalObject(state, const_cast<int32_t*>(*lower_addr-128),
                    384 * sizeof **lower_addr, true);
  addExternalObject(state, lower_addr, sizeof(*lower_addr), true);
  
  const int32_t **upper_addr = __ctype_toupper_loc();
  addExternalObject(state, const_cast<int32_t*>(*upper_addr-128),
                    384 * sizeof **upper_addr, true);
  addExternalObject(state, upper_addr, sizeof(*upper_addr), true);
#endif
#endif
#endif

  // allocate and initialize globals, done in two passes since we may
  // need address of a global in order to initialize some other one.

  // allocate memory objects for all globals
  for (Module::const_global_iterator i = m->global_begin(),
         e = m->global_end();
       i != e; ++i) {
    if (i->isDeclaration()) {
      // FIXME: We have no general way of handling unknown external
      // symbols. If we really cared about making external stuff work
      // better we could support user definition, or use the EXE style
      // hack where we check the object file information.

      LLVM_TYPE_Q Type *ty = i->getType()->getElementType();
      uint64_t size = 0;
      if (ty->isSized()) {
	size = kmodule->targetData->getTypeStoreSize(ty);
      } else {
        klee_warning("Type for %.*s is not sized", (int)i->getName().size(),
			i->getName().data());
      }

      // XXX - DWD - hardcode some things until we decide how to fix.
#ifndef WINDOWS
      if (i->getName() == "_ZTVN10__cxxabiv117__class_type_infoE") {
        size = 0x2C;
      } else if (i->getName() == "_ZTVN10__cxxabiv120__si_class_type_infoE") {
        size = 0x2C;
      } else if (i->getName() == "_ZTVN10__cxxabiv121__vmi_class_type_infoE") {
        size = 0x2C;
      }
#endif

      if (size == 0) {
        klee_warning("Unable to find size for global variable: %.*s (use will result in out of bounds access)",
			(int)i->getName().size(), i->getName().data());
      }

      MemoryObject *mo = memory->allocate(size, false, true, i);
      ObjectState *os = bindObjectInState(state, mo, false);
      globalObjects.insert(std::make_pair(i, mo));
      globalAddresses.insert(std::make_pair(i, mo->getBaseExpr()));

      // Program already running = object already initialized.  Read
      // concrete value and write it to our copy.
      if (size) {
        void *addr;
        if (i->getName() == "__dso_handle") {
          addr = &__dso_handle; // wtf ?
        } else {
          addr = externalDispatcher->resolveSymbol(i->getName());
        }
        if (!addr)
          klee_error("unable to load symbol(%s) while initializing globals.", 
                     i->getName().data());

        for (unsigned offset=0; offset<mo->size; offset++)
          os->write8(offset, ((unsigned char*)addr)[offset]);
      }
    } else {
      LLVM_TYPE_Q Type *ty = i->getType()->getElementType();
      uint64_t size = kmodule->targetData->getTypeStoreSize(ty);
      MemoryObject *mo = memory->allocate(size, false, true, &*i);
      if (!mo)
        llvm::report_fatal_error("out of memory");
      ObjectState *os = bindObjectInState(state, mo, false);
      globalObjects.insert(std::make_pair(i, mo));
      globalAddresses.insert(std::make_pair(i, mo->getBaseExpr()));

      if (!i->hasInitializer())
          os->initializeToRandom();
    }
  }
  
  // link aliases to their definitions (if bound)
  for (Module::alias_iterator i = m->alias_begin(), ie = m->alias_end(); 
       i != ie; ++i) {
    // Map the alias to its aliasee's address. This works because we have
    // addresses for everything, even undefined functions. 
    globalAddresses.insert(std::make_pair(i, evalConstant(i->getAliasee())));
  }

  // once all objects are allocated, do the actual initialization
  for (Module::const_global_iterator i = m->global_begin(),
         e = m->global_end();
       i != e; ++i) {
    if (i->hasInitializer()) {
      MemoryObject *mo = globalObjects.find(i)->second;
      const ObjectState *os = state.addressSpace.findObject(mo);
      assert(os);
      ObjectState *wos = state.addressSpace.getWriteable(mo, os);
      
      initializeGlobalObject(state, wos, i->getInitializer(), 0);
      // if(i->isConstant()) os->setReadOnly(true);
    }
  }
}

void Executor::branch(ExecutionState &state, 
                      const std::vector< ref<Expr> > &conditions,
                      std::vector<ExecutionState*> &result) {
  TimerStatIncrementer timer(stats::forkTime);
  unsigned N = conditions.size();
  assert(N);

  if (MaxForks!=~0u && stats::forks >= MaxForks) {
    unsigned next = theRNG.getInt32() % N;
    for (unsigned i=0; i<N; ++i) {
      if (i == next) {
        result.push_back(&state);
      } else {
        result.push_back(NULL);
      }
    }
  } else {
    stats::forks += N-1;

    // XXX do proper balance or keep random?
    result.push_back(&state);
    for (unsigned i=1; i<N; ++i) {
      ExecutionState *es = result[theRNG.getInt32() % i];
      ExecutionState *ns = es->branch();
      addedStates.push_back(ns);
      result.push_back(ns);
      es->ptreeNode->data = 0;
      std::pair<PTree::Node*,PTree::Node*> res = 
        processTree->split(es->ptreeNode, ns, es);
      ns->ptreeNode = res.first;
      es->ptreeNode = res.second;
    }
  }

  // If necessary redistribute seeds to match conditions, killing
  // states if necessary due to OnlyReplaySeeds (inefficient but
  // simple).
  
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it != seedMap.end()) {
    std::vector<SeedInfo> seeds = it->second;
    seedMap.erase(it);

    // Assume each seed only satisfies one condition (necessarily true
    // when conditions are mutually exclusive and their conjunction is
    // a tautology).
    for (std::vector<SeedInfo>::iterator siit = seeds.begin(), 
           siie = seeds.end(); siit != siie; ++siit) {
      unsigned i;
      for (i=0; i<N; ++i) {
        ref<ConstantExpr> res;
        bool success = 
          solver->getValue(state, siit->assignment.evaluate(conditions[i]), 
                           res);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (res->isTrue())
          break;
      }
      
      // If we didn't find a satisfying condition randomly pick one
      // (the seed will be patched).
      if (i==N)
        i = theRNG.getInt32() % N;

      // Extra check in case we're replaying seeds with a max-fork
      if (result[i])
        seedMap[result[i]].push_back(*siit);
    }

    if (OnlyReplaySeeds) {
      for (unsigned i=0; i<N; ++i) {
        if (result[i] && !seedMap.count(result[i])) {
          // @KLEE-SEMu
          if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
            terminateStateEarly(*result[i], "Do not follow Seed");
          } else {
          //~KS
            terminateState(*result[i]);
            result[i] = NULL;
          // @KLEE-SEMu
          }
          //~KS
        }
      } 
    }
  }

  for (unsigned i=0; i<N; ++i)
    if (result[i])
      addConstraint(*result[i], conditions[i]);
}

Executor::StatePair 
Executor::fork(ExecutionState &current, ref<Expr> condition, bool isInternal) {
  Solver::Validity res;
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&current);
  bool isSeeding = it != seedMap.end();

  if (!isSeeding && !isa<ConstantExpr>(condition) && 
      (MaxStaticForkPct!=1. || MaxStaticSolvePct != 1. ||
       MaxStaticCPForkPct!=1. || MaxStaticCPSolvePct != 1.) &&
      statsTracker->elapsed() > 60.) {
    StatisticManager &sm = *theStatisticManager;
    CallPathNode *cpn = current.stack.back().callPathNode;
    if ((MaxStaticForkPct<1. &&
         sm.getIndexedValue(stats::forks, sm.getIndex()) > 
         stats::forks*MaxStaticForkPct) ||
        (MaxStaticCPForkPct<1. &&
         cpn && (cpn->statistics.getValue(stats::forks) > 
                 stats::forks*MaxStaticCPForkPct)) ||
        (MaxStaticSolvePct<1 &&
         sm.getIndexedValue(stats::solverTime, sm.getIndex()) > 
         stats::solverTime*MaxStaticSolvePct) ||
        (MaxStaticCPForkPct<1. &&
         cpn && (cpn->statistics.getValue(stats::solverTime) > 
                 stats::solverTime*MaxStaticCPSolvePct))) {
      ref<ConstantExpr> value; 
      bool success = solver->getValue(current, condition, value);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      addConstraint(current, EqExpr::create(value, condition));
      condition = value;
    }
  }

  double timeout = coreSolverTimeout;
  if (isSeeding)
    timeout *= it->second.size();
  solver->setTimeout(timeout);
  bool success = solver->evaluate(current, condition, res);
  solver->setTimeout(0);
  if (!success) {
    current.pc = current.prevPC;
    terminateStateEarly(current, "Query timed out (fork).");
    return StatePair(0, 0);
  }

  if (!isSeeding) {
    if (replayPath && !isInternal) {
      assert(replayPosition<replayPath->size() &&
             "ran out of branches in replay path mode");
      bool branch = (*replayPath)[replayPosition++];
      
      if (res==Solver::True) {
        assert(branch && "hit invalid branch in replay path mode");
      } else if (res==Solver::False) {
        assert(!branch && "hit invalid branch in replay path mode");
      } else {
        // add constraints
        if(branch) {
          res = Solver::True;
          addConstraint(current, condition);
        } else  {
          res = Solver::False;
          addConstraint(current, Expr::createIsZero(condition));
        }
      }
    } else if (res==Solver::Unknown) {
      assert(!replayKTest && "in replay mode, only one branch can be true.");
      
      if ((MaxMemoryInhibit && atMemoryLimit) || 
          current.forkDisabled ||
          inhibitForking || 
          (MaxForks!=~0u && stats::forks >= MaxForks)) {

        if (MaxMemoryInhibit && atMemoryLimit)
          klee_warning_once(0, "skipping fork (memory cap exceeded)");
        else if (current.forkDisabled)
          klee_warning_once(0, "skipping fork (fork disabled on current path)");
        else if (inhibitForking)
          klee_warning_once(0, "skipping fork (fork disabled globally)");
        else 
          klee_warning_once(0, "skipping fork (max-forks reached)");

        TimerStatIncrementer timer(stats::forkTime);
        if (theRNG.getBool()) {
          addConstraint(current, condition);
          res = Solver::True;        
        } else {
          addConstraint(current, Expr::createIsZero(condition));
          res = Solver::False;
        }
      }
    }
  }

  // Fix branch in only-replay-seed mode, if we don't have both true
  // and false seeds.
  if (isSeeding && 
      (current.forkDisabled || OnlyReplaySeeds) && 
      res == Solver::Unknown) {
    bool trueSeed=false, falseSeed=false;
    // Is seed extension still ok here?
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      ref<ConstantExpr> res;
      bool success = 
        solver->getValue(current, siit->assignment.evaluate(condition), res);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res->isTrue()) {
        trueSeed = true;
      } else {
        falseSeed = true;
      }
      if (trueSeed && falseSeed)
        break;
    }
    if (!(trueSeed && falseSeed)) {
      assert(trueSeed || falseSeed);
      
      res = trueSeed ? Solver::True : Solver::False;
      addConstraint(current, trueSeed ? condition : Expr::createIsZero(condition));
    }
  }


  // XXX - even if the constraint is provable one way or the other we
  // can probably benefit by adding this constraint and allowing it to
  // reduce the other constraints. For example, if we do a binary
  // search on a particular value, and then see a comparison against
  // the value it has been fixed at, we should take this as a nice
  // hint to just use the single constraint instead of all the binary
  // search ones. If that makes sense.
  if (res==Solver::True) {
    if (!isInternal) {
      if (pathWriter) {
        current.pathOS << "1";
      }
    }

    // @KLEE-SEMu 
    if (!semuUseOnlyBranchForDepth)
      current.depth++;
    //~KS
    return StatePair(&current, 0);
  } else if (res==Solver::False) {
    if (!isInternal) {
      if (pathWriter) {
        current.pathOS << "0";
      }
    }

    // @KLEE-SEMu 
    if (!semuUseOnlyBranchForDepth)
      current.depth++;
    //~KS
    return StatePair(0, &current);
  } else {
    TimerStatIncrementer timer(stats::forkTime);
    ExecutionState *falseState, *trueState = &current;

    ++stats::forks;

    falseState = trueState->branch();
    addedStates.push_back(falseState);

    if (it != seedMap.end()) {
      std::vector<SeedInfo> seeds = it->second;
      it->second.clear();
      std::vector<SeedInfo> &trueSeeds = seedMap[trueState];
      std::vector<SeedInfo> &falseSeeds = seedMap[falseState];
      for (std::vector<SeedInfo>::iterator siit = seeds.begin(), 
             siie = seeds.end(); siit != siie; ++siit) {
        ref<ConstantExpr> res;
        bool success = 
          solver->getValue(current, siit->assignment.evaluate(condition), res);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (res->isTrue()) {
          trueSeeds.push_back(*siit);
        } else {
          falseSeeds.push_back(*siit);
        }
      }
      
      bool swapInfo = false;
      if (trueSeeds.empty()) {
        if (&current == trueState) swapInfo = true;
        seedMap.erase(trueState);
      }
      if (falseSeeds.empty()) {
        if (&current == falseState) swapInfo = true;
        seedMap.erase(falseState);
      }
      if (swapInfo) {
        std::swap(trueState->coveredNew, falseState->coveredNew);
        std::swap(trueState->coveredLines, falseState->coveredLines);
      }
    }

    current.ptreeNode->data = 0;
    std::pair<PTree::Node*, PTree::Node*> res =
      processTree->split(current.ptreeNode, falseState, trueState);
    falseState->ptreeNode = res.first;
    trueState->ptreeNode = res.second;

    if (!isInternal) {
      if (pathWriter) {
        falseState->pathOS = pathWriter->open(current.pathOS);
        trueState->pathOS << "1";
        falseState->pathOS << "0";
      }      
      if (symPathWriter) {
        falseState->symPathOS = symPathWriter->open(current.symPathOS);
        trueState->symPathOS << "1";
        falseState->symPathOS << "0";
      }
    }

    addConstraint(*trueState, condition);
    addConstraint(*falseState, Expr::createIsZero(condition));

    // Kinda gross, do we even really still want this option?
    if (MaxDepth && MaxDepth<=trueState->depth) {
      terminateStateEarly(*trueState, "max-depth exceeded.");
      terminateStateEarly(*falseState, "max-depth exceeded.");
      return StatePair(0, 0);
    }

    return StatePair(trueState, falseState);
  }
}

void Executor::addConstraint(ExecutionState &state, ref<Expr> condition) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(condition)) {
    if (!CE->isTrue())
      llvm::report_fatal_error("attempt to add invalid constraint");
    return;
  }

  // Check to see if this constraint violates seeds.
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it != seedMap.end()) {
    bool warn = false;
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      bool res;
      bool success = 
        solver->mustBeFalse(state, siit->assignment.evaluate(condition), res);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res) {
        siit->patchSeed(state, condition, solver);
        warn = true;
      }
    }
    if (warn)
      klee_warning("seeds patched for violating constraint"); 
  }

  state.addConstraint(condition);
  if (ivcEnabled)
    doImpliedValueConcretization(state, condition, 
                                 ConstantExpr::alloc(1, Expr::Bool));
}

ref<klee::ConstantExpr> Executor::evalConstant(const Constant *c) {
  if (const llvm::ConstantExpr *ce = dyn_cast<llvm::ConstantExpr>(c)) {
    return evalConstantExpr(ce);
  } else {
    if (const ConstantInt *ci = dyn_cast<ConstantInt>(c)) {
      return ConstantExpr::alloc(ci->getValue());
    } else if (const ConstantFP *cf = dyn_cast<ConstantFP>(c)) {      
      return ConstantExpr::alloc(cf->getValueAPF().bitcastToAPInt());
    } else if (const GlobalValue *gv = dyn_cast<GlobalValue>(c)) {
      return globalAddresses.find(gv)->second;
    } else if (isa<ConstantPointerNull>(c)) {
      return Expr::createPointer(0);
    } else if (isa<UndefValue>(c) || isa<ConstantAggregateZero>(c)) {
      return ConstantExpr::create(0, getWidthForLLVMType(c->getType()));
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
    } else if (const ConstantDataSequential *cds =
                 dyn_cast<ConstantDataSequential>(c)) {
      std::vector<ref<Expr> > kids;
      for (unsigned i = 0, e = cds->getNumElements(); i != e; ++i) {
        ref<Expr> kid = evalConstant(cds->getElementAsConstant(i));
        kids.push_back(kid);
      }
      ref<Expr> res = ConcatExpr::createN(kids.size(), kids.data());
      return cast<ConstantExpr>(res);
#endif
    } else if (const ConstantStruct *cs = dyn_cast<ConstantStruct>(c)) {
      const StructLayout *sl = kmodule->targetData->getStructLayout(cs->getType());
      llvm::SmallVector<ref<Expr>, 4> kids;
      for (unsigned i = cs->getNumOperands(); i != 0; --i) {
        unsigned op = i-1;
        ref<Expr> kid = evalConstant(cs->getOperand(op));

        uint64_t thisOffset = sl->getElementOffsetInBits(op),
                 nextOffset = (op == cs->getNumOperands() - 1)
                              ? sl->getSizeInBits()
                              : sl->getElementOffsetInBits(op+1);
        if (nextOffset-thisOffset > kid->getWidth()) {
          uint64_t paddingWidth = nextOffset-thisOffset-kid->getWidth();
          kids.push_back(ConstantExpr::create(0, paddingWidth));
        }

        kids.push_back(kid);
      }
      ref<Expr> res = ConcatExpr::createN(kids.size(), kids.data());
      return cast<ConstantExpr>(res);
    } else if (const ConstantArray *ca = dyn_cast<ConstantArray>(c)){
      llvm::SmallVector<ref<Expr>, 4> kids;
      for (unsigned i = ca->getNumOperands(); i != 0; --i) {
        unsigned op = i-1;
        ref<Expr> kid = evalConstant(ca->getOperand(op));
        kids.push_back(kid);
      }
      ref<Expr> res = ConcatExpr::createN(kids.size(), kids.data());
      return cast<ConstantExpr>(res);
    } else {
      // Constant{Vector}
      llvm::report_fatal_error("invalid argument to evalConstant()");
    }
  }
}

const Cell& Executor::eval(KInstruction *ki, unsigned index, 
                           ExecutionState &state) const {
  assert(index < ki->inst->getNumOperands());
  int vnumber = ki->operands[index];

  assert(vnumber != -1 &&
         "Invalid operand to eval(), not a value or constant!");

  // Determine if this is a constant or not.
  if (vnumber < 0) {
    unsigned index = -vnumber - 2;
    return kmodule->constantTable[index];
  } else {
    unsigned index = vnumber;
    StackFrame &sf = state.stack.back();
    return sf.locals[index];
  }
}

void Executor::bindLocal(KInstruction *target, ExecutionState &state, 
                         ref<Expr> value) {
  getDestCell(state, target).value = value;
}

void Executor::bindArgument(KFunction *kf, unsigned index, 
                            ExecutionState &state, ref<Expr> value) {
  getArgumentCell(state, kf, index).value = value;
}

ref<Expr> Executor::toUnique(const ExecutionState &state, 
                             ref<Expr> &e) {
  ref<Expr> result = e;

  if (!isa<ConstantExpr>(e)) {
    ref<ConstantExpr> value;
    bool isTrue = false;

    solver->setTimeout(coreSolverTimeout);      
    if (solver->getValue(state, e, value) &&
        solver->mustBeTrue(state, EqExpr::create(e, value), isTrue) &&
        isTrue)
      result = value;
    solver->setTimeout(0);
  }
  
  return result;
}


/* Concretize the given expression, and return a possible constant value. 
   'reason' is just a documentation string stating the reason for concretization. */
ref<klee::ConstantExpr> 
Executor::toConstant(ExecutionState &state, 
                     ref<Expr> e,
                     const char *reason) {
  e = state.constraints.simplifyExpr(e);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(e))
    return CE;

  ref<ConstantExpr> value;
  bool success = solver->getValue(state, e, value);
  assert(success && "FIXME: Unhandled solver failure");
  (void) success;

  std::string str;
  llvm::raw_string_ostream os(str);
  os << "silently concretizing (reason: " << reason << ") expression " << e
     << " to value " << value << " (" << (*(state.pc)).info->file << ":"
     << (*(state.pc)).info->line << ")";

  if (AllExternalWarnings)
    klee_warning(reason, os.str().c_str());
  else
    klee_warning_once(reason, "%s", os.str().c_str());

  addConstraint(state, EqExpr::create(e, value));
    
  return value;
}

void Executor::executeGetValue(ExecutionState &state,
                               ref<Expr> e,
                               KInstruction *target) {
  e = state.constraints.simplifyExpr(e);
  std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
    seedMap.find(&state);
  if (it==seedMap.end() || isa<ConstantExpr>(e)) {
    ref<ConstantExpr> value;
    bool success = solver->getValue(state, e, value);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    bindLocal(target, state, value);
  } else {
    std::set< ref<Expr> > values;
    for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
           siie = it->second.end(); siit != siie; ++siit) {
      ref<ConstantExpr> value;
      bool success = 
        solver->getValue(state, siit->assignment.evaluate(e), value);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      values.insert(value);
    }
    
    std::vector< ref<Expr> > conditions;
    for (std::set< ref<Expr> >::iterator vit = values.begin(), 
           vie = values.end(); vit != vie; ++vit)
      conditions.push_back(EqExpr::create(e, *vit));

    std::vector<ExecutionState*> branches;
    branch(state, conditions, branches);
    
    std::vector<ExecutionState*>::iterator bit = branches.begin();
    for (std::set< ref<Expr> >::iterator vit = values.begin(), 
           vie = values.end(); vit != vie; ++vit) {
      ExecutionState *es = *bit;
      if (es)
        bindLocal(target, *es, *vit);
      ++bit;
    }
  }
}

void Executor::printDebugInstructions(ExecutionState &state) {
  // check do not print
  if (DebugPrintInstructions.size() == 0)
	  return;

  llvm::raw_ostream *stream = 0;
  if (optionIsSet(DebugPrintInstructions, STDERR_ALL) ||
      optionIsSet(DebugPrintInstructions, STDERR_SRC) ||
      optionIsSet(DebugPrintInstructions, STDERR_COMPACT))
    stream = &llvm::errs();
  else
    stream = &debugLogBuffer;

  if (!optionIsSet(DebugPrintInstructions, STDERR_COMPACT) &&
      !optionIsSet(DebugPrintInstructions, FILE_COMPACT))
    printFileLine(state, state.pc, *stream);

  (*stream) << state.pc->info->id;

  if (optionIsSet(DebugPrintInstructions, STDERR_ALL) ||
      optionIsSet(DebugPrintInstructions, FILE_ALL))
    (*stream) << ":" << *(state.pc->inst);
  (*stream) << "\n";

  if (optionIsSet(DebugPrintInstructions, FILE_ALL) ||
      optionIsSet(DebugPrintInstructions, FILE_COMPACT) ||
      optionIsSet(DebugPrintInstructions, FILE_SRC)) {
    debugLogBuffer.flush();
    (*debugInstFile) << debugLogBuffer.str();
    debugBufferString = "";
  }
}

void Executor::stepInstruction(ExecutionState &state) {
  printDebugInstructions(state);
  if (statsTracker)
    statsTracker->stepInstruction(state);

  ++stats::instructions;
  state.prevPC = state.pc;
  ++state.pc;

  if (stats::instructions==StopAfterNInstructions)
    haltExecution = true;
}

void Executor::executeCall(ExecutionState &state, 
                           KInstruction *ki,
                           Function *f,
                           std::vector< ref<Expr> > &arguments) {
  Instruction *i = ki->inst;
  if (f && f->isDeclaration()) {
    switch(f->getIntrinsicID()) {
    case Intrinsic::not_intrinsic:
      // state may be destroyed by this call, cannot touch
      callExternalFunction(state, ki, f, arguments);
      break;
        
      // va_arg is handled by caller and intrinsic lowering, see comment for
      // ExecutionState::varargs
    case Intrinsic::vastart:  {
      StackFrame &sf = state.stack.back();

      // varargs can be zero if no varargs were provided
      if (!sf.varargs)
        return;

      // FIXME: This is really specific to the architecture, not the pointer
      // size. This happens to work fir x86-32 and x86-64, however.
      Expr::Width WordSize = Context::get().getPointerWidth();
      if (WordSize == Expr::Int32) {
        executeMemoryOperation(state, true, arguments[0], 
                               sf.varargs->getBaseExpr(), 0);
      } else {
        assert(WordSize == Expr::Int64 && "Unknown word size!");

        // X86-64 has quite complicated calling convention. However,
        // instead of implementing it, we can do a simple hack: just
        // make a function believe that all varargs are on stack.
        executeMemoryOperation(state, true, arguments[0],
                               ConstantExpr::create(48, 32), 0); // gp_offset
        executeMemoryOperation(state, true,
                               AddExpr::create(arguments[0], 
                                               ConstantExpr::create(4, 64)),
                               ConstantExpr::create(304, 32), 0); // fp_offset
        executeMemoryOperation(state, true,
                               AddExpr::create(arguments[0], 
                                               ConstantExpr::create(8, 64)),
                               sf.varargs->getBaseExpr(), 0); // overflow_arg_area
        executeMemoryOperation(state, true,
                               AddExpr::create(arguments[0], 
                                               ConstantExpr::create(16, 64)),
                               ConstantExpr::create(0, 64), 0); // reg_save_area
      }
      break;
    }
    case Intrinsic::vaend:
      // va_end is a noop for the interpreter.
      //
      // FIXME: We should validate that the target didn't do something bad
      // with vaeend, however (like call it twice).
      break;
        
  // @KLEE-SEMu
    case Intrinsic::trap:
      terminateStateEarly (state, "Trap instruction");
      break;
  //~KS
    case Intrinsic::vacopy:
      // va_copy should have been lowered.
      //
      // FIXME: It would be nice to check for errors in the usage of this as
      // well.
    default:
      klee_error("unknown intrinsic: %s", f->getName().data());
    }

    if (InvokeInst *ii = dyn_cast<InvokeInst>(i))
      transferToBasicBlock(ii->getNormalDest(), i->getParent(), state);
  } else {
  // @KLEE-SEMu
    // Make sure to unset the ks_hasToReachPostMutationPoint so to avoid
    // post mutation point checking when the execution of the mutated stmt
    // calls a function. the reason for this are:
    // - Other mutants may be spawn in the called function, confusing
    // therefore the original.
    // - A mutant may mutate the value returned by the function call
    // and considered same as original because no diff found 
    // within the called funcion.
    if (state.ks_hasToReachPostMutationPoint)
      state.ks_hasToReachPostMutationPoint = false;
  //~KS
    // FIXME: I'm not really happy about this reliance on prevPC but it is ok, I
    // guess. This just done to avoid having to pass KInstIterator everywhere
    // instead of the actual instruction, since we can't make a KInstIterator
    // from just an instruction (unlike LLVM).
    KFunction *kf = kmodule->functionMap[f];
    state.pushFrame(state.prevPC, kf);
    state.pc = kf->instructions;

    if (statsTracker)
      statsTracker->framePushed(state, &state.stack[state.stack.size()-2]);

     // TODO: support "byval" parameter attribute
     // TODO: support zeroext, signext, sret attributes

    unsigned callingArgs = arguments.size();
    unsigned funcArgs = f->arg_size();
    if (!f->isVarArg()) {
      if (callingArgs > funcArgs) {
        klee_warning_once(f, "calling %s with extra arguments.", 
                          f->getName().data());
      } else if (callingArgs < funcArgs) {
        terminateStateOnError(state, "calling function with too few arguments",
                              User);
        return;
      }
    } else {
      Expr::Width WordSize = Context::get().getPointerWidth();

      if (callingArgs < funcArgs) {
        terminateStateOnError(state, "calling function with too few arguments",
                              User);
        return;
      }

      StackFrame &sf = state.stack.back();
      unsigned size = 0;
      bool requires16ByteAlignment = false;
      for (unsigned i = funcArgs; i < callingArgs; i++) {
        // FIXME: This is really specific to the architecture, not the pointer
        // size. This happens to work for x86-32 and x86-64, however.
        if (WordSize == Expr::Int32) {
          size += Expr::getMinBytesForWidth(arguments[i]->getWidth());
        } else {
          Expr::Width argWidth = arguments[i]->getWidth();
          // AMD64-ABI 3.5.7p5: Step 7. Align l->overflow_arg_area upwards to a
          // 16 byte boundary if alignment needed by type exceeds 8 byte
          // boundary.
          //
          // Alignment requirements for scalar types is the same as their size
          if (argWidth > Expr::Int64) {
             size = llvm::RoundUpToAlignment(size, 16);
             requires16ByteAlignment = true;
          }
          size += llvm::RoundUpToAlignment(argWidth, WordSize) / 8;
        }
      }

      MemoryObject *mo = sf.varargs =
          memory->allocate(size, true, false, state.prevPC->inst,
                           (requires16ByteAlignment ? 16 : 8));
      if (!mo && size) {
        terminateStateOnExecError(state, "out of memory (varargs)");
        return;
      }

      if (mo) {
        if ((WordSize == Expr::Int64) && (mo->address & 15) &&
            requires16ByteAlignment) {
          // Both 64bit Linux/Glibc and 64bit MacOSX should align to 16 bytes.
          klee_warning_once(
              0, "While allocating varargs: malloc did not align to 16 bytes.");
        }

        ObjectState *os = bindObjectInState(state, mo, true);
        unsigned offset = 0;
        for (unsigned i = funcArgs; i < callingArgs; i++) {
          // FIXME: This is really specific to the architecture, not the pointer
          // size. This happens to work for x86-32 and x86-64, however.
          if (WordSize == Expr::Int32) {
            os->write(offset, arguments[i]);
            offset += Expr::getMinBytesForWidth(arguments[i]->getWidth());
          } else {
            assert(WordSize == Expr::Int64 && "Unknown word size!");

            Expr::Width argWidth = arguments[i]->getWidth();
            if (argWidth > Expr::Int64) {
              offset = llvm::RoundUpToAlignment(offset, 16);
            }
            os->write(offset, arguments[i]);
            offset += llvm::RoundUpToAlignment(argWidth, WordSize) / 8;
          }
        }
      }
    }

    unsigned numFormals = f->arg_size();
    for (unsigned i=0; i<numFormals; ++i) 
      bindArgument(kf, i, state, arguments[i]);
  }
}

void Executor::transferToBasicBlock(BasicBlock *dst, BasicBlock *src, 
                                    ExecutionState &state) {
  // Note that in general phi nodes can reuse phi values from the same
  // block but the incoming value is the eval() result *before* the
  // execution of any phi nodes. this is pathological and doesn't
  // really seem to occur, but just in case we run the PhiCleanerPass
  // which makes sure this cannot happen and so it is safe to just
  // eval things in order. The PhiCleanerPass also makes sure that all
  // incoming blocks have the same order for each PHINode so we only
  // have to compute the index once.
  //
  // With that done we simply set an index in the state so that PHI
  // instructions know which argument to eval, set the pc, and continue.
  
  // XXX this lookup has to go ?
  KFunction *kf = state.stack.back().kf;
  unsigned entry = kf->basicBlockEntry[dst];
  state.pc = &kf->instructions[entry];
  if (state.pc->inst->getOpcode() == Instruction::PHI) {
    PHINode *first = static_cast<PHINode*>(state.pc->inst);
    state.incomingBBIndex = first->getBasicBlockIndex(src);
  }
}

void Executor::printFileLine(ExecutionState &state, KInstruction *ki,
                             llvm::raw_ostream &debugFile) {
  const InstructionInfo &ii = *ki->info;
  if (ii.file != "")
    debugFile << "     " << ii.file << ":" << ii.line << ":";
  else
    debugFile << "     [no debug info]:";
}

/// Compute the true target of a function call, resolving LLVM and KLEE aliases
/// and bitcasts.
Function* Executor::getTargetFunction(Value *calledVal, ExecutionState &state) {
  SmallPtrSet<const GlobalValue*, 3> Visited;

  Constant *c = dyn_cast<Constant>(calledVal);
  if (!c)
    return 0;

  while (true) {
    if (GlobalValue *gv = dyn_cast<GlobalValue>(c)) {
      if (!Visited.insert(gv))
        return 0;

      std::string alias = state.getFnAlias(gv->getName());
      if (alias != "") {
        llvm::Module* currModule = kmodule->module;
        GlobalValue *old_gv = gv;
        gv = currModule->getNamedValue(alias);
        if (!gv) {
          klee_error("Function %s(), alias for %s not found!\n", alias.c_str(),
                     old_gv->getName().str().c_str());
        }
      }
     
      if (Function *f = dyn_cast<Function>(gv))
        return f;
      else if (GlobalAlias *ga = dyn_cast<GlobalAlias>(gv))
        c = ga->getAliasee();
      else
        return 0;
    } else if (llvm::ConstantExpr *ce = dyn_cast<llvm::ConstantExpr>(c)) {
      if (ce->getOpcode()==Instruction::BitCast)
        c = ce->getOperand(0);
      else
        return 0;
    } else
      return 0;
  }
}

/// TODO remove?
static bool isDebugIntrinsic(const Function *f, KModule *KM) {
  return false;
}

static inline const llvm::fltSemantics * fpWidthToSemantics(unsigned width) {
  switch(width) {
  case Expr::Int32:
    return &llvm::APFloat::IEEEsingle;
  case Expr::Int64:
    return &llvm::APFloat::IEEEdouble;
  case Expr::Fl80:
    return &llvm::APFloat::x87DoubleExtended;
  default:
    return 0;
  }
}

void Executor::executeInstruction(ExecutionState &state, KInstruction *ki) {
  Instruction *i = ki->inst;
  switch (i->getOpcode()) {
    // Control flow
  case Instruction::Ret: {
    ReturnInst *ri = cast<ReturnInst>(i);
    KInstIterator kcaller = state.stack.back().caller;
    Instruction *caller = kcaller ? kcaller->inst : 0;
    bool isVoidReturn = (ri->getNumOperands() == 0);
    /*// @KLEE-SEMU - COMMENT
    ref<Expr> result = ConstantExpr::alloc(0, Expr::Bool);
    */ //~KS
    
    // @KLEE-SEMu
    ref<Expr> &result = state.ks_lastReturnedVal;
    result = ConstantExpr::alloc(0, Expr::Bool);
    //~KS
    
    if (!isVoidReturn) {
      result = eval(ki, 0, state).value;
    }
    
    if (state.stack.size() <= 1) {
      assert(!caller && "caller set on initial stack frame");
      terminateStateOnExit(state);
    } else {
      state.popFrame();

      if (statsTracker)
        statsTracker->framePopped(state);

      if (InvokeInst *ii = dyn_cast<InvokeInst>(caller)) {
        transferToBasicBlock(ii->getNormalDest(), caller->getParent(), state);
      } else {
        state.pc = kcaller;
        ++state.pc;
      }

      if (!isVoidReturn) {
        LLVM_TYPE_Q Type *t = caller->getType();
        if (t != Type::getVoidTy(getGlobalContext())) {
          // may need to do coercion due to bitcasts
          Expr::Width from = result->getWidth();
          Expr::Width to = getWidthForLLVMType(t);
            
          if (from != to) {
            CallSite cs = (isa<InvokeInst>(caller) ? CallSite(cast<InvokeInst>(caller)) : 
                           CallSite(cast<CallInst>(caller)));

            // XXX need to check other param attrs ?
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
            bool isSExt = cs.paramHasAttr(0, llvm::Attribute::SExt);
#elif LLVM_VERSION_CODE >= LLVM_VERSION(3, 2)
	    bool isSExt = cs.paramHasAttr(0, llvm::Attributes::SExt);
#else
	    bool isSExt = cs.paramHasAttr(0, llvm::Attribute::SExt);
#endif
            if (isSExt) {
              result = SExtExpr::create(result, to);
            } else {
              result = ZExtExpr::create(result, to);
            }
          }

          bindLocal(kcaller, state, result);
        }
      } else {
        // We check that the return value has no users instead of
        // checking the type, since C defaults to returning int for
        // undeclared functions.
        if (!caller->use_empty()) {
          terminateStateOnExecError(state, "return void when caller expected a result");
        }
      }
    }      
    break;
  }
#if LLVM_VERSION_CODE < LLVM_VERSION(3, 1)
  case Instruction::Unwind: {
    for (;;) {
      KInstruction *kcaller = state.stack.back().caller;
      state.popFrame();

      if (statsTracker)
        statsTracker->framePopped(state);

      if (state.stack.empty()) {
        terminateStateOnExecError(state, "unwind from initial stack frame");
        break;
      } else {
        Instruction *caller = kcaller->inst;
        if (InvokeInst *ii = dyn_cast<InvokeInst>(caller)) {
          transferToBasicBlock(ii->getUnwindDest(), caller->getParent(), state);
          break;
        }
      }
    }
    break;
  }
#endif
  case Instruction::Br: {
    BranchInst *bi = cast<BranchInst>(i);
    if (bi->isUnconditional()) {
      transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), state);
    } else {
      // FIXME: Find a way that we don't have this hidden dependency.
      assert(bi->getCondition() == bi->getOperand(0) &&
             "Wrong operand index!");
      ref<Expr> cond = eval(ki, 0, state).value;
      Executor::StatePair branches = fork(state, cond, false);

      // NOTE: There is a hidden dependency here, markBranchVisited
      // requires that we still be in the context of the branch
      // instruction (it reuses its statistic id). Should be cleaned
      // up with convenient instruction specific data.
      if (statsTracker && state.stack.back().kf->trackCoverage)
        statsTracker->markBranchVisited(branches.first, branches.second);

      if (branches.first)
        transferToBasicBlock(bi->getSuccessor(0), bi->getParent(), *branches.first);
      if (branches.second)
        transferToBasicBlock(bi->getSuccessor(1), bi->getParent(), *branches.second);
    }
    break;
  }
  case Instruction::Switch: {
    SwitchInst *si = cast<SwitchInst>(i);
    
    ref<Expr> cond = eval(ki, 0, state).value;
    BasicBlock *bb = si->getParent();

    cond = toUnique(state, cond);
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(cond)) {
      // Somewhat gross to create these all the time, but fine till we
      // switch to an internal rep.
      LLVM_TYPE_Q llvm::IntegerType *Ty = 
        cast<IntegerType>(si->getCondition()->getType());
      ConstantInt *ci = ConstantInt::get(Ty, CE->getZExtValue());
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
      unsigned index = si->findCaseValue(ci).getSuccessorIndex();
#else
      unsigned index = si->findCaseValue(ci);
#endif
      transferToBasicBlock(si->getSuccessor(index), si->getParent(), state);
    } else {
      // Handle possible different branch targets

      // We have the following assumptions:
      // - each case value is mutual exclusive to all other values including the
      //   default value
      // - order of case branches is based on the order of the expressions of
      //   the scase values, still default is handled last
      std::vector<BasicBlock *> bbOrder;
      std::map<BasicBlock *, ref<Expr> > branchTargets;

      std::map<ref<Expr>, BasicBlock *> expressionOrder;

      // Iterate through all non-default cases and order them by expressions
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
      for (SwitchInst::CaseIt i = si->case_begin(), e = si->case_end(); i != e;
           ++i) {
        ref<Expr> value = evalConstant(i.getCaseValue());
#else
      for (unsigned i = 1, cases = si->getNumCases(); i < cases; ++i) {
        ref<Expr> value = evalConstant(si->getCaseValue(i));
#endif

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 1)
        BasicBlock *caseSuccessor = i.getCaseSuccessor();
#else
        BasicBlock *caseSuccessor = si->getSuccessor(i);
#endif
        expressionOrder.insert(std::make_pair(value, caseSuccessor));
      }

      // Track default branch values
      ref<Expr> defaultValue = ConstantExpr::alloc(1, Expr::Bool);

      // iterate through all non-default cases but in order of the expressions
      for (std::map<ref<Expr>, BasicBlock *>::iterator
               it = expressionOrder.begin(),
               itE = expressionOrder.end();
           it != itE; ++it) {
        ref<Expr> match = EqExpr::create(cond, it->first);

        // Make sure that the default value does not contain this target's value
        defaultValue = AndExpr::create(defaultValue, Expr::createIsZero(match));

        // Check if control flow could take this case
        bool result;
        bool success = solver->mayBeTrue(state, match, result);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        if (result) {
          BasicBlock *caseSuccessor = it->second;

          // Handle the case that a basic block might be the target of multiple
          // switch cases.
          // Currently we generate an expression containing all switch-case
          // values for the same target basic block. We spare us forking too
          // many times but we generate more complex condition expressions
          // TODO Add option to allow to choose between those behaviors
          std::pair<std::map<BasicBlock *, ref<Expr> >::iterator, bool> res =
              branchTargets.insert(std::make_pair(
                  caseSuccessor, ConstantExpr::alloc(0, Expr::Bool)));

          res.first->second = OrExpr::create(match, res.first->second);

          // Only add basic blocks which have not been target of a branch yet
          if (res.second) {
            bbOrder.push_back(caseSuccessor);
          }
        }
      }

      // Check if control could take the default case
      bool res;
      bool success = solver->mayBeTrue(state, defaultValue, res);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      if (res) {
        std::pair<std::map<BasicBlock *, ref<Expr> >::iterator, bool> ret =
            branchTargets.insert(
                std::make_pair(si->getDefaultDest(), defaultValue));
        if (ret.second) {
          bbOrder.push_back(si->getDefaultDest());
        }
      }

      // Fork the current state with each state having one of the possible
      // successors of this switch
      std::vector< ref<Expr> > conditions;
      for (std::vector<BasicBlock *>::iterator it = bbOrder.begin(),
                                               ie = bbOrder.end();
           it != ie; ++it) {
        conditions.push_back(branchTargets[*it]);
      }
      std::vector<ExecutionState*> branches;
      branch(state, conditions, branches);

      std::vector<ExecutionState*>::iterator bit = branches.begin();
      for (std::vector<BasicBlock *>::iterator it = bbOrder.begin(),
                                               ie = bbOrder.end();
           it != ie; ++it) {
        ExecutionState *es = *bit;
        if (es)
          transferToBasicBlock(*it, bb, *es);
        ++bit;
      }
    }
    break;
 }
  case Instruction::Unreachable:
    // Note that this is not necessarily an internal bug, llvm will
    // generate unreachable instructions in cases where it knows the
    // program will crash. So it is effectively a SEGV or internal
    // error.
    terminateStateOnExecError(state, "reached \"unreachable\" instruction");
    break;

  case Instruction::Invoke:
  case Instruction::Call: {
    CallSite cs(i);

    unsigned numArgs = cs.arg_size();
    Value *fp = cs.getCalledValue();
    Function *f = getTargetFunction(fp, state);

    // @KLEE-SEMu
    // if the state executing is original (ks_mutantID==0) and first time this is executed
    // by this state (This instruction ki is not in ks_covMutPointInst of this state), 
    // and the condition of switch is the value of the global variable 
    // @klee_semu_GenMu_Mutant_ID_Selector, do mutations branchings.
    if (f == ks_mutantIDSelectorGlobal_Func) {
        if (state.ks_mutantID == 0 && state.ks_VisitedMutPointsSet.count(i) == 0) {
        
            uint64_t fromMID = dyn_cast<ConstantInt>(dyn_cast<CallInst>(i)->getArgOperand(0))->getZExtValue();
            uint64_t toMID = dyn_cast<ConstantInt>(dyn_cast<CallInst>(i)->getArgOperand(1))->getZExtValue();
            assert (fromMID <= toMID && "Invalid mutant range");
            
            std::vector<uint64_t> mIdsHere;
            for (unsigned mIds = fromMID; mIds <= toMID; mIds++) {
                auto ins_ret = state.ks_VisitedMutantsSet.insert(mIds);
                // Consider only the mutants that were not seen before (Higher order mutant case)
                if (ins_ret.second)
                    mIdsHere.push_back(mIds);
            }
            // produce mutants states from the current original's state with each state having 
            // one of the possible Mutants of this mutation point (successors of this switch)
            if (!mIdsHere.empty())
                ks_mutationPointBranching(state, mIdsHere);
            //llvm::errs() << fromMID << " " << toMID << " --\n";    
            //i->dump();
            // Set to visited, so that subsequent pass do not mutate anymore
            state.ks_VisitedMutPointsSet.insert(i);
                
            break;  //case
        }
        else { //Simply skip that call when mutant or already seen
            break;  //case
        }
    } else if (f == ks_postMutationPoint_Func) {
      // Do nothing. Will be checked later, if !semuDisableCheckAtPostMutationPoint
      break;
    }
    //~KS

    // Skip debug intrinsics, we can't evaluate their metadata arguments.
    if (f && isDebugIntrinsic(f, kmodule))
      break;

    if (isa<InlineAsm>(fp)) {
      terminateStateOnExecError(state, "inline assembly is unsupported");
      break;
    }
    // evaluate arguments
    std::vector< ref<Expr> > arguments;
    arguments.reserve(numArgs);

    for (unsigned j=0; j<numArgs; ++j)
      arguments.push_back(eval(ki, j+1, state).value);

    if (f) {
      const FunctionType *fType = 
        dyn_cast<FunctionType>(cast<PointerType>(f->getType())->getElementType());
      const FunctionType *fpType =
        dyn_cast<FunctionType>(cast<PointerType>(fp->getType())->getElementType());

      // special case the call with a bitcast case
      if (fType != fpType) {
        assert(fType && fpType && "unable to get function type");

        // XXX check result coercion

        // XXX this really needs thought and validation
        unsigned i=0;
        for (std::vector< ref<Expr> >::iterator
               ai = arguments.begin(), ie = arguments.end();
             ai != ie; ++ai) {
          Expr::Width to, from = (*ai)->getWidth();
            
          if (i<fType->getNumParams()) {
            to = getWidthForLLVMType(fType->getParamType(i));

            if (from != to) {
              // XXX need to check other param attrs ?
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
              bool isSExt = cs.paramHasAttr(i+1, llvm::Attribute::SExt);
#elif LLVM_VERSION_CODE >= LLVM_VERSION(3, 2)
	      bool isSExt = cs.paramHasAttr(i+1, llvm::Attributes::SExt);
#else
	      bool isSExt = cs.paramHasAttr(i+1, llvm::Attribute::SExt);
#endif
              if (isSExt) {
                arguments[i] = SExtExpr::create(arguments[i], to);
              } else {
                arguments[i] = ZExtExpr::create(arguments[i], to);
              }
            }
          }
            
          i++;
        }
      }

      executeCall(state, ki, f, arguments);
    } else {
      ref<Expr> v = eval(ki, 0, state).value;

      ExecutionState *free = &state;
      bool hasInvalid = false, first = true;

      /* XXX This is wasteful, no need to do a full evaluate since we
         have already got a value. But in the end the caches should
         handle it for us, albeit with some overhead. */
      do {
        ref<ConstantExpr> value;
        bool success = solver->getValue(*free, v, value);
        assert(success && "FIXME: Unhandled solver failure");
        (void) success;
        StatePair res = fork(*free, EqExpr::create(v, value), true);
        if (res.first) {
          uint64_t addr = value->getZExtValue();
          if (legalFunctions.count(addr)) {
            f = (Function*) addr;

            // Don't give warning on unique resolution
            if (res.second || !first)
              klee_warning_once((void*) (unsigned long) addr, 
                                "resolved symbolic function pointer to: %s",
                                f->getName().data());

            executeCall(*res.first, ki, f, arguments);
          } else {
            if (!hasInvalid) {
              terminateStateOnExecError(state, "invalid function pointer");
              hasInvalid = true;
            }
          }
        }

        first = false;
        free = res.second;
      } while (free);
    }
    // @KLEE-SEMu
#ifdef SEMU_RELMUT_PRED_ENABLED
    // branch old new
    if (f == ks_klee_change_function && state.ks_old_new == 0)
      ks_oldNewBranching(state);
#endif
    //~KS
    break;
  }
  case Instruction::PHI: {
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 0)
    ref<Expr> result = eval(ki, state.incomingBBIndex, state).value;
#else
    ref<Expr> result = eval(ki, state.incomingBBIndex * 2, state).value;
#endif
    bindLocal(ki, state, result);
    break;
  }

    // Special instructions
  case Instruction::Select: {
    ref<Expr> cond = eval(ki, 0, state).value;
    ref<Expr> tExpr = eval(ki, 1, state).value;
    ref<Expr> fExpr = eval(ki, 2, state).value;
    ref<Expr> result = SelectExpr::create(cond, tExpr, fExpr);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::VAArg:
    terminateStateOnExecError(state, "unexpected VAArg instruction");
    break;

    // Arithmetic / logical

  case Instruction::Add: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    bindLocal(ki, state, AddExpr::create(left, right));
    break;
  }

  case Instruction::Sub: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    bindLocal(ki, state, SubExpr::create(left, right));
    break;
  }
 
  case Instruction::Mul: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    bindLocal(ki, state, MulExpr::create(left, right));
    break;
  }

  case Instruction::UDiv: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = UDivExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::SDiv: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = SDivExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::URem: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = URemExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }
 
  case Instruction::SRem: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = SRemExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::And: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = AndExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Or: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = OrExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Xor: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = XorExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::Shl: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = ShlExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::LShr: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = LShrExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::AShr: {
    ref<Expr> left = eval(ki, 0, state).value;
    ref<Expr> right = eval(ki, 1, state).value;
    ref<Expr> result = AShrExpr::create(left, right);
    bindLocal(ki, state, result);
    break;
  }

    // Compare

  case Instruction::ICmp: {
    CmpInst *ci = cast<CmpInst>(i);
    ICmpInst *ii = cast<ICmpInst>(ci);
 
    switch(ii->getPredicate()) {
    case ICmpInst::ICMP_EQ: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = EqExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_NE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = NeExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_UGT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = UgtExpr::create(left, right);
      bindLocal(ki, state,result);
      break;
    }

    case ICmpInst::ICMP_UGE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = UgeExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_ULT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = UltExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_ULE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = UleExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SGT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = SgtExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SGE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = SgeExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SLT: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = SltExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    case ICmpInst::ICMP_SLE: {
      ref<Expr> left = eval(ki, 0, state).value;
      ref<Expr> right = eval(ki, 1, state).value;
      ref<Expr> result = SleExpr::create(left, right);
      bindLocal(ki, state, result);
      break;
    }

    default:
      terminateStateOnExecError(state, "invalid ICmp predicate");
    }
    break;
  }
 
    // Memory instructions...
  case Instruction::Alloca: {
    AllocaInst *ai = cast<AllocaInst>(i);
    unsigned elementSize = 
      kmodule->targetData->getTypeStoreSize(ai->getAllocatedType());
    ref<Expr> size = Expr::createPointer(elementSize);
    if (ai->isArrayAllocation()) {
      ref<Expr> count = eval(ki, 0, state).value;
      count = Expr::createZExtToPointerWidth(count);
      size = MulExpr::create(size, count);
    }
    executeAlloc(state, size, true, ki);
    break;
  }

  case Instruction::Load: {
    ref<Expr> base = eval(ki, 0, state).value;
    executeMemoryOperation(state, false, base, 0, ki);
    break;
  }
  case Instruction::Store: {
    ref<Expr> base = eval(ki, 1, state).value;
    ref<Expr> value = eval(ki, 0, state).value;
    executeMemoryOperation(state, true, base, value, 0);
    break;
  }

  case Instruction::GetElementPtr: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);
    ref<Expr> base = eval(ki, 0, state).value;

    for (std::vector< std::pair<unsigned, uint64_t> >::iterator 
           it = kgepi->indices.begin(), ie = kgepi->indices.end(); 
         it != ie; ++it) {
      uint64_t elementSize = it->second;
      ref<Expr> index = eval(ki, it->first, state).value;
      base = AddExpr::create(base,
                             MulExpr::create(Expr::createSExtToPointerWidth(index),
                                             Expr::createPointer(elementSize)));
    }
    if (kgepi->offset)
      base = AddExpr::create(base,
                             Expr::createPointer(kgepi->offset));
    bindLocal(ki, state, base);
    break;
  }

    // Conversion
  case Instruction::Trunc: {
    CastInst *ci = cast<CastInst>(i);
    ref<Expr> result = ExtractExpr::create(eval(ki, 0, state).value,
                                           0,
                                           getWidthForLLVMType(ci->getType()));
    bindLocal(ki, state, result);
    break;
  }
  case Instruction::ZExt: {
    CastInst *ci = cast<CastInst>(i);
    ref<Expr> result = ZExtExpr::create(eval(ki, 0, state).value,
                                        getWidthForLLVMType(ci->getType()));
    bindLocal(ki, state, result);
    break;
  }
  case Instruction::SExt: {
    CastInst *ci = cast<CastInst>(i);
    ref<Expr> result = SExtExpr::create(eval(ki, 0, state).value,
                                        getWidthForLLVMType(ci->getType()));
    bindLocal(ki, state, result);
    break;
  }

  case Instruction::IntToPtr: {
    CastInst *ci = cast<CastInst>(i);
    Expr::Width pType = getWidthForLLVMType(ci->getType());
    ref<Expr> arg = eval(ki, 0, state).value;
    bindLocal(ki, state, ZExtExpr::create(arg, pType));
    break;
  } 
  case Instruction::PtrToInt: {
    CastInst *ci = cast<CastInst>(i);
    Expr::Width iType = getWidthForLLVMType(ci->getType());
    ref<Expr> arg = eval(ki, 0, state).value;
    bindLocal(ki, state, ZExtExpr::create(arg, iType));
    break;
  }

  case Instruction::BitCast: {
    ref<Expr> result = eval(ki, 0, state).value;
    bindLocal(ki, state, result);
    break;
  }

    // Floating point instructions

  case Instruction::FAdd: {
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FAdd operation");

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
    llvm::APFloat Res(*fpWidthToSemantics(left->getWidth()), left->getAPValue());
    Res.add(APFloat(*fpWidthToSemantics(right->getWidth()),right->getAPValue()), APFloat::rmNearestTiesToEven);
#else
    llvm::APFloat Res(left->getAPValue());
    Res.add(APFloat(right->getAPValue()), APFloat::rmNearestTiesToEven);
#endif
    bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
    break;
  }

  case Instruction::FSub: {
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FSub operation");
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
    llvm::APFloat Res(*fpWidthToSemantics(left->getWidth()), left->getAPValue());
    Res.subtract(APFloat(*fpWidthToSemantics(right->getWidth()), right->getAPValue()), APFloat::rmNearestTiesToEven);
#else
    llvm::APFloat Res(left->getAPValue());
    Res.subtract(APFloat(right->getAPValue()), APFloat::rmNearestTiesToEven);
#endif
    bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
    break;
  }
 
  case Instruction::FMul: {
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FMul operation");

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
    llvm::APFloat Res(*fpWidthToSemantics(left->getWidth()), left->getAPValue());
    Res.multiply(APFloat(*fpWidthToSemantics(right->getWidth()), right->getAPValue()), APFloat::rmNearestTiesToEven);
#else
    llvm::APFloat Res(left->getAPValue());
    Res.multiply(APFloat(right->getAPValue()), APFloat::rmNearestTiesToEven);
#endif
    bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
    break;
  }

  case Instruction::FDiv: {
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FDiv operation");

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
    llvm::APFloat Res(*fpWidthToSemantics(left->getWidth()), left->getAPValue());
    Res.divide(APFloat(*fpWidthToSemantics(right->getWidth()), right->getAPValue()), APFloat::rmNearestTiesToEven);
#else
    llvm::APFloat Res(left->getAPValue());
    Res.divide(APFloat(right->getAPValue()), APFloat::rmNearestTiesToEven);
#endif
    bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
    break;
  }

  case Instruction::FRem: {
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FRem operation");
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
    llvm::APFloat Res(*fpWidthToSemantics(left->getWidth()), left->getAPValue());
    Res.mod(APFloat(*fpWidthToSemantics(right->getWidth()),right->getAPValue()),
            APFloat::rmNearestTiesToEven);
#else
    llvm::APFloat Res(left->getAPValue());
    Res.mod(APFloat(right->getAPValue()), APFloat::rmNearestTiesToEven);
#endif
    bindLocal(ki, state, ConstantExpr::alloc(Res.bitcastToAPInt()));
    break;
  }

  case Instruction::FPTrunc: {
    FPTruncInst *fi = cast<FPTruncInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > arg->getWidth())
      return terminateStateOnExecError(state, "Unsupported FPTrunc operation");

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
    llvm::APFloat Res(*fpWidthToSemantics(arg->getWidth()), arg->getAPValue());
#else
    llvm::APFloat Res(arg->getAPValue());
#endif
    bool losesInfo = false;
    Res.convert(*fpWidthToSemantics(resultType),
                llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
    bindLocal(ki, state, ConstantExpr::alloc(Res));
    break;
  }

  case Instruction::FPExt: {
    FPExtInst *fi = cast<FPExtInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || arg->getWidth() > resultType)
      return terminateStateOnExecError(state, "Unsupported FPExt operation");
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
    llvm::APFloat Res(*fpWidthToSemantics(arg->getWidth()), arg->getAPValue());
#else
    llvm::APFloat Res(arg->getAPValue());
#endif
    bool losesInfo = false;
    Res.convert(*fpWidthToSemantics(resultType),
                llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
    bindLocal(ki, state, ConstantExpr::alloc(Res));
    break;
  }

  case Instruction::FPToUI: {
    FPToUIInst *fi = cast<FPToUIInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > 64)
      return terminateStateOnExecError(state, "Unsupported FPToUI operation");

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
    llvm::APFloat Arg(*fpWidthToSemantics(arg->getWidth()), arg->getAPValue());
#else
    llvm::APFloat Arg(arg->getAPValue());
#endif
    uint64_t value = 0;
    bool isExact = true;
    Arg.convertToInteger(&value, resultType, false,
                         llvm::APFloat::rmTowardZero, &isExact);
    bindLocal(ki, state, ConstantExpr::alloc(value, resultType));
    break;
  }

  case Instruction::FPToSI: {
    FPToSIInst *fi = cast<FPToSIInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    if (!fpWidthToSemantics(arg->getWidth()) || resultType > 64)
      return terminateStateOnExecError(state, "Unsupported FPToSI operation");
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
    llvm::APFloat Arg(*fpWidthToSemantics(arg->getWidth()), arg->getAPValue());
#else
    llvm::APFloat Arg(arg->getAPValue());

#endif
    uint64_t value = 0;
    bool isExact = true;
    Arg.convertToInteger(&value, resultType, true,
                         llvm::APFloat::rmTowardZero, &isExact);
    bindLocal(ki, state, ConstantExpr::alloc(value, resultType));
    break;
  }

  case Instruction::UIToFP: {
    UIToFPInst *fi = cast<UIToFPInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
    if (!semantics)
      return terminateStateOnExecError(state, "Unsupported UIToFP operation");
    llvm::APFloat f(*semantics, 0);
    f.convertFromAPInt(arg->getAPValue(), false,
                       llvm::APFloat::rmNearestTiesToEven);

    bindLocal(ki, state, ConstantExpr::alloc(f));
    break;
  }

  case Instruction::SIToFP: {
    SIToFPInst *fi = cast<SIToFPInst>(i);
    Expr::Width resultType = getWidthForLLVMType(fi->getType());
    ref<ConstantExpr> arg = toConstant(state, eval(ki, 0, state).value,
                                       "floating point");
    const llvm::fltSemantics *semantics = fpWidthToSemantics(resultType);
    if (!semantics)
      return terminateStateOnExecError(state, "Unsupported SIToFP operation");
    llvm::APFloat f(*semantics, 0);
    f.convertFromAPInt(arg->getAPValue(), true,
                       llvm::APFloat::rmNearestTiesToEven);

    bindLocal(ki, state, ConstantExpr::alloc(f));
    break;
  }

  case Instruction::FCmp: {
    FCmpInst *fi = cast<FCmpInst>(i);
    ref<ConstantExpr> left = toConstant(state, eval(ki, 0, state).value,
                                        "floating point");
    ref<ConstantExpr> right = toConstant(state, eval(ki, 1, state).value,
                                         "floating point");
    if (!fpWidthToSemantics(left->getWidth()) ||
        !fpWidthToSemantics(right->getWidth()))
      return terminateStateOnExecError(state, "Unsupported FCmp operation");

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
    APFloat LHS(*fpWidthToSemantics(left->getWidth()),left->getAPValue());
    APFloat RHS(*fpWidthToSemantics(right->getWidth()),right->getAPValue());
#else
    APFloat LHS(left->getAPValue());
    APFloat RHS(right->getAPValue());
#endif
    APFloat::cmpResult CmpRes = LHS.compare(RHS);

    bool Result = false;
    switch( fi->getPredicate() ) {
      // Predicates which only care about whether or not the operands are NaNs.
    case FCmpInst::FCMP_ORD:
      Result = CmpRes != APFloat::cmpUnordered;
      break;

    case FCmpInst::FCMP_UNO:
      Result = CmpRes == APFloat::cmpUnordered;
      break;

      // Ordered comparisons return false if either operand is NaN.  Unordered
      // comparisons return true if either operand is NaN.
    case FCmpInst::FCMP_UEQ:
      if (CmpRes == APFloat::cmpUnordered) {
        Result = true;
        break;
      }
    case FCmpInst::FCMP_OEQ:
      Result = CmpRes == APFloat::cmpEqual;
      break;

    case FCmpInst::FCMP_UGT:
      if (CmpRes == APFloat::cmpUnordered) {
        Result = true;
        break;
      }
    case FCmpInst::FCMP_OGT:
      Result = CmpRes == APFloat::cmpGreaterThan;
      break;

    case FCmpInst::FCMP_UGE:
      if (CmpRes == APFloat::cmpUnordered) {
        Result = true;
        break;
      }
    case FCmpInst::FCMP_OGE:
      Result = CmpRes == APFloat::cmpGreaterThan || CmpRes == APFloat::cmpEqual;
      break;

    case FCmpInst::FCMP_ULT:
      if (CmpRes == APFloat::cmpUnordered) {
        Result = true;
        break;
      }
    case FCmpInst::FCMP_OLT:
      Result = CmpRes == APFloat::cmpLessThan;
      break;

    case FCmpInst::FCMP_ULE:
      if (CmpRes == APFloat::cmpUnordered) {
        Result = true;
        break;
      }
    case FCmpInst::FCMP_OLE:
      Result = CmpRes == APFloat::cmpLessThan || CmpRes == APFloat::cmpEqual;
      break;

    case FCmpInst::FCMP_UNE:
      Result = CmpRes == APFloat::cmpUnordered || CmpRes != APFloat::cmpEqual;
      break;
    case FCmpInst::FCMP_ONE:
      Result = CmpRes != APFloat::cmpUnordered && CmpRes != APFloat::cmpEqual;
      break;

    default:
      assert(0 && "Invalid FCMP predicate!");
    case FCmpInst::FCMP_FALSE:
      Result = false;
      break;
    case FCmpInst::FCMP_TRUE:
      Result = true;
      break;
    }

    bindLocal(ki, state, ConstantExpr::alloc(Result, Expr::Bool));
    break;
  }
  case Instruction::InsertValue: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    ref<Expr> agg = eval(ki, 0, state).value;
    ref<Expr> val = eval(ki, 1, state).value;

    ref<Expr> l = NULL, r = NULL;
    unsigned lOffset = kgepi->offset*8, rOffset = kgepi->offset*8 + val->getWidth();

    if (lOffset > 0)
      l = ExtractExpr::create(agg, 0, lOffset);
    if (rOffset < agg->getWidth())
      r = ExtractExpr::create(agg, rOffset, agg->getWidth() - rOffset);

    ref<Expr> result;
    if (!l.isNull() && !r.isNull())
      result = ConcatExpr::create(r, ConcatExpr::create(val, l));
    else if (!l.isNull())
      result = ConcatExpr::create(val, l);
    else if (!r.isNull())
      result = ConcatExpr::create(r, val);
    else
      result = val;

    bindLocal(ki, state, result);
    break;
  }
  case Instruction::ExtractValue: {
    KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(ki);

    ref<Expr> agg = eval(ki, 0, state).value;

    ref<Expr> result = ExtractExpr::create(agg, kgepi->offset*8, getWidthForLLVMType(i->getType()));

    bindLocal(ki, state, result);
    break;
  }
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
  case Instruction::Fence: {
    // Ignore for now
    break;
  }
#endif

  // Other instructions...
  // Unhandled
  case Instruction::ExtractElement:
  case Instruction::InsertElement:
  case Instruction::ShuffleVector:
    terminateStateOnError(state, "XXX vector instructions unhandled",
                          Unhandled);
    break;
 
  default:
    terminateStateOnExecError(state, "illegal instruction");
    break;
  }
}

void Executor::updateStates(ExecutionState *current) {
  if (searcher) {
    searcher->update(current, addedStates, removedStates);
  }
  
  states.insert(addedStates.begin(), addedStates.end());
  addedStates.clear();

  for (std::vector<ExecutionState *>::iterator it = removedStates.begin(),
                                               ie = removedStates.end();
       it != ie; ++it) {
    ExecutionState *es = *it;
    std::set<ExecutionState*>::iterator it2 = states.find(es);
    assert(it2!=states.end());
    states.erase(it2);
    std::map<ExecutionState*, std::vector<SeedInfo> >::iterator it3 = 
      seedMap.find(es);
    if (it3 != seedMap.end())
      seedMap.erase(it3);
    processTree->remove(es->ptreeNode);
    delete es;
  }
  removedStates.clear();
}

template <typename TypeIt>
void Executor::computeOffsets(KGEPInstruction *kgepi, TypeIt ib, TypeIt ie) {
  ref<ConstantExpr> constantOffset =
    ConstantExpr::alloc(0, Context::get().getPointerWidth());
  uint64_t index = 1;
  for (TypeIt ii = ib; ii != ie; ++ii) {
    if (LLVM_TYPE_Q StructType *st = dyn_cast<StructType>(*ii)) {
      const StructLayout *sl = kmodule->targetData->getStructLayout(st);
      const ConstantInt *ci = cast<ConstantInt>(ii.getOperand());
      uint64_t addend = sl->getElementOffset((unsigned) ci->getZExtValue());
      constantOffset = constantOffset->Add(ConstantExpr::alloc(addend,
                                                               Context::get().getPointerWidth()));
    } else {
      const SequentialType *set = cast<SequentialType>(*ii);
      uint64_t elementSize = 
        kmodule->targetData->getTypeStoreSize(set->getElementType());
      Value *operand = ii.getOperand();
      if (Constant *c = dyn_cast<Constant>(operand)) {
        ref<ConstantExpr> index = 
          evalConstant(c)->SExt(Context::get().getPointerWidth());
        ref<ConstantExpr> addend = 
          index->Mul(ConstantExpr::alloc(elementSize,
                                         Context::get().getPointerWidth()));
        constantOffset = constantOffset->Add(addend);
      } else {
        kgepi->indices.push_back(std::make_pair(index, elementSize));
      }
    }
    index++;
  }
  kgepi->offset = constantOffset->getZExtValue();
}

void Executor::bindInstructionConstants(KInstruction *KI) {
  KGEPInstruction *kgepi = static_cast<KGEPInstruction*>(KI);

  if (GetElementPtrInst *gepi = dyn_cast<GetElementPtrInst>(KI->inst)) {
    computeOffsets(kgepi, gep_type_begin(gepi), gep_type_end(gepi));
  } else if (InsertValueInst *ivi = dyn_cast<InsertValueInst>(KI->inst)) {
    computeOffsets(kgepi, iv_type_begin(ivi), iv_type_end(ivi));
    assert(kgepi->indices.empty() && "InsertValue constant offset expected");
  } else if (ExtractValueInst *evi = dyn_cast<ExtractValueInst>(KI->inst)) {
    computeOffsets(kgepi, ev_type_begin(evi), ev_type_end(evi));
    assert(kgepi->indices.empty() && "ExtractValue constant offset expected");
  }
}

void Executor::bindModuleConstants() {
  for (std::vector<KFunction*>::iterator it = kmodule->functions.begin(), 
         ie = kmodule->functions.end(); it != ie; ++it) {
    KFunction *kf = *it;
    for (unsigned i=0; i<kf->numInstructions; ++i)
      bindInstructionConstants(kf->instructions[i]);
  }

  kmodule->constantTable = new Cell[kmodule->constants.size()];
  for (unsigned i=0; i<kmodule->constants.size(); ++i) {
    Cell &c = kmodule->constantTable[i];
    c.value = evalConstant(kmodule->constants[i]);
  }
}

void Executor::checkMemoryUsage() {
  if (!MaxMemory)
    return;
  if ((stats::instructions & 0xFFFF) == 0) {
    // We need to avoid calling GetTotalMallocUsage() often because it
    // is O(elts on freelist). This is really bad since we start
    // to pummel the freelist once we hit the memory cap.
    unsigned mbs = (util::GetTotalMallocUsage() >> 20) +
                   (memory->getUsedDeterministicSize() >> 20);

    if (mbs > MaxMemory) {
      if (mbs > MaxMemory + 100) {
        // just guess at how many to kill
        unsigned numStates = states.size();
        unsigned toKill = std::max(1U, numStates - numStates * MaxMemory / mbs);
        klee_warning("killing %d states (over memory cap)", toKill);
        std::vector<ExecutionState *> arr(states.begin(), states.end());
        for (unsigned i = 0, N = arr.size(); N && i < toKill; ++i, --N) {
          unsigned idx = rand() % N;
          // Make two pulls to try and not hit a state that
          // covered new code.
          if (arr[idx]->coveredNew)
            idx = rand() % N;

          std::swap(arr[idx], arr[N - 1]);
#ifdef SEMU_RELMUT_PRED_ENABLED
          terminateState(*arr[N - 1]);
#else
          terminateStateEarly(*arr[N - 1], "Memory limit exceeded.");
#endif
        }
      }
      atMemoryLimit = true;
    } else {
      atMemoryLimit = false;
    }
  }
}

void Executor::doDumpStates() {
  // @KLEE-SEMu
  if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
    // In case there was timeout, compare the states that already reached watchpoint
    std::vector<ExecutionState *> remainWPStates;
    ks_checkID++;
    ks_compareStates(remainWPStates); // TODO TODO: modify comparestates to take account of the fact that some states may not have reached the watchpoint
    for (auto *s: remainWPStates) {
      s->pc = s->prevPC;
      terminateState (*s);
    }
    return; // do not do the real dumpStates
  }
  //~KS

  if (!DumpStatesOnHalt || states.empty())
    return;
  klee_message("halting execution, dumping remaining states");
  for (std::set<ExecutionState *>::iterator it = states.begin(),
                                            ie = states.end();
       it != ie; ++it) {
    ExecutionState &state = **it;
    stepInstruction(state); // keep stats rolling
    terminateStateEarly(state, "Execution halting.");
  }
  updateStates(0);
}

void Executor::run(ExecutionState &initialState) {
  bindModuleConstants();

  // Delay init till now so that ticks don't accrue during
  // optimization and such.
  initTimers();

  states.insert(&initialState);

  // @KLEE-SEMu         //Temporary TODO
  if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
    // Must not be only seed mode XXX: need to check this, but should be okay
    //if (OnlySeed)
    //  klee_error("SEMU@ERROR: OnlySeed must not be enabled in SEMU mode");

    // OnlyReplaySeeds must be set if precondition is enabled by giving seed dir or so
    // XXX Maybe need to automatically set OnlyReplaySeeds here instead of displaying error?
    if (usingSeeds) { //!seedMap.empty()) {
      if (!OnlyReplaySeeds) {
        klee_error("SEMU@ERROR: OnlyReplaySeeds is not set in Semu mode whith seed precondition. Must be set!");
        exit(1);
      }
    }

    ks_checkID = 0;
    ks_nextDepthID = 1;

    ks_runStartTime = util::getWallTime();
    
    // In SEMU mode set optional user specified precondition, before any seeding
    // Precondition (bounded) symb ex for scalability, existing TC precond
    // XXX: in case want to specify precondition instead of using seeds
    std::vector<ConstraintManager> symbexPreconditionsList;
    ks_loadKQueryConstraints(symbexPreconditionsList);
    // Make precondition Expression
    if (symbexPreconditionsList.size() > 0)
        assert (semuPreconditionLength >= 0 && "Must use symbexPreconditionsList only with fixed preconditionLength");
    unsigned nSelectedConds = semuPreconditionLength;
    ref<Expr> mergedPrecond = ConstantExpr::alloc(0, Expr::Bool);  //False
    for (auto &testcasePC: symbexPreconditionsList) {
      ref<Expr> tcPCCond = ConstantExpr::alloc(1, Expr::Bool); //True
      unsigned condcount = 0;
      for (auto cit=testcasePC.begin(); cit != testcasePC.end();++cit) {
        if (condcount++ >= nSelectedConds)
          break;
        tcPCCond = AndExpr::create(tcPCCond, *cit);
      }
      mergedPrecond = OrExpr::create(mergedPrecond, tcPCCond);
    }
    // add precondition  
    if (!mergedPrecond->isFalse())
      for (auto *as: states) {
        as->constraints.addConstraint(mergedPrecond);
          //(mergedPrecond)->dump();
      }
  }
  //~KS

  if (usingSeeds) {
    std::vector<SeedInfo> &v = seedMap[&initialState];
    
    for (std::vector<KTest*>::const_iterator it = usingSeeds->begin(), 
           ie = usingSeeds->end(); it != ie; ++it)
      v.push_back(SeedInfo(*it));

    int lastNumSeeds = usingSeeds->size()+10;
    double lastTime, startTime = lastTime = util::getWallTime();
    ExecutionState *lastState = 0;

    // @KLEE-SEMu         //Temporary TODO
    //ks_watchpoint = true;
    double ks_initTime = util::getWallTime();
    ExecutionState *ks_prevState = nullptr;  

    // will have the number of states that already reached precondition(for compare to be sound)
    uint64_t precond_offset = 0;
    uint64_t prevSeedMapSize = seedMap.size();
    //~KS

    while (!seedMap.empty()) {
      if (haltExecution) {
        doDumpStates();
        return;
      }

      // @KLEE-SEMu
      std::map<ExecutionState*, std::vector<SeedInfo> >::iterator it;
      if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
        if (seedMap.size() != prevSeedMapSize) {
          it = seedMap.upper_bound(lastState); // take another if seedMap changed
          prevSeedMapSize = seedMap.size();
        } else { 
          it = seedMap.lower_bound(lastState); //take same if no change (for loop break)
        }
      } else {
        it = seedMap.upper_bound(lastState);
      }

      /*
      //~KS
      std::map<ExecutionState*, std::vector<SeedInfo> >::iterator it = 
        seedMap.upper_bound(lastState);
      // @KLEE-SEMu
      */
      //~KS
      if (it == seedMap.end())
        it = seedMap.begin();
      lastState = it->first;
      unsigned numSeeds = it->second.size();
      ExecutionState &state = *lastState;
      KInstruction *ki = state.pc;

      // @KLEE-SEMu
      if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
        // remove the last state from seedMap if depth exceeded the precondition length
        if (semuPreconditionLength < 0) {
          // We are in non fixed mode: precondition until we reach a point:
          // == -1: until any state reach a mutant and the depth is fixed for others
          // < -1: until each state reach a mutant
          if (ks_reachedAMutant(ki)
#ifdef SEMU_RELMUT_PRED_ENABLED
              || ks_reachedAnOldNew(ki)
#endif
                                  ) {
            if (semuPreconditionLength == -1)
              semuPreconditionLength = state.depth; 
            // else ;
            seedMap.erase(it);
            precond_offset++;
            continue;
          }
        } else if (state.depth >= semuPreconditionLength) {
          seedMap.erase(it);
          precond_offset++;
          continue;
        }

      //Handle infinite loop, TODO: Check whether it fits the search strategy of seedMap (next elem) 
  //#ifdef SEMU_INFINITE_LOOP_BREAK
        ks_CheckAndBreakInfinitLoop(state, ks_prevState, ks_initTime);
  //#endif
      }
      //~KS

      stepInstruction(state);

      executeInstruction(state, ki);
      processTimers(&state, MaxInstructionTime * numSeeds);

      // @KLEE-SEMu
      if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
	// No check when precondlength is -2 because seeding stop when a mutant reached
        if (semuPreconditionLength >= -1) {
          if (ks_CheckpointingMainCheck(state, ki, true/*is Seeidng*/, precond_offset)) {
            processTimers(nullptr, MaxInstructionTime * numSeeds); // Make sure inst timer is reset
            continue;  // avoid putting back the state in the searcher bellow "updateStates(&state);"
          }
        } else {
          // No check, thus we need to terminate terminated originals that did not reach a mutant
          for (auto *s: ks_justTerminatedStates)
                  terminateState(*s);
          ks_justTerminatedStates.clear();
        }
      }
      //~KS

      updateStates(&state);

      if ((stats::instructions % 1000) == 0) {
        int numSeeds = 0, numStates = 0;
        for (std::map<ExecutionState*, std::vector<SeedInfo> >::iterator
               it = seedMap.begin(), ie = seedMap.end();
             it != ie; ++it) {
          numSeeds += it->second.size();
          numStates++;
        }
        double time = util::getWallTime();
        if (SeedTime>0. && time > startTime + SeedTime) {
          klee_warning("seed time expired, %d seeds remain over %d states",
                       numSeeds, numStates);
          break;
        } else if (numSeeds<=lastNumSeeds-10 ||
                   time >= lastTime+10) {
          lastTime = time;
          lastNumSeeds = numSeeds;          
          klee_message("%d seeds remaining over: %d states", 
                       numSeeds, numStates);
        }
      }
    }

    klee_message("seeding done (%d states remain)", (int) states.size());

    // XXX total hack, just because I like non uniform better but want
    // seed results to be equally weighted.
    for (std::set<ExecutionState*>::iterator
           it = states.begin(), ie = states.end();
         it != ie; ++it) {
      (*it)->weight = 1.;
    }

    if (OnlySeed) {
      doDumpStates();
      return;
    }
  }

  searcher = constructUserSearcher(*this);

  // @KLEE-SEMu
  // XXX require bfs as searcher (for SEMU execution) for now
  std::string searcherName;
  llvm::raw_string_ostream rsos(searcherName);
  searcher->printName(rsos);
  if (rsos.str().compare("BFSSearcher\n")) {
    llvm::errs() << "SEMU@ERROR: SEMU require BFS as search!";
    assert (false && "SEMU require BFS as search");
  }

  std::vector<ExecutionState *> newStates;
  for (auto *s: states) {
    if (ks_terminatedBeforeWP.count(s) > 0 
        || ks_reachedWatchPoint.count(s) > 0 
        || ks_reachedOutEnv.count(s) > 0
        || ks_atPointPostMutation.count(s) > 0
        || ks_ongoingExecutionAtWP.count(s) > 0)
      continue;
    newStates.push_back(s);
  }
  /*
  //~KS
  std::vector<ExecutionState *> newStates(states.begin(), states.end());
  // @KLEE-SEMu
  */
  //~KS
  searcher->update(0, newStates, std::vector<ExecutionState *>());
  
  // @KLEE-SEMu         //Temporary TODO
  //ks_watchpoint = true;
  double ks_initTime = util::getWallTime();
  ExecutionState *ks_prevState = nullptr;  
  //~KS
  
  while (!states.empty() && !haltExecution) {
    ExecutionState &state = searcher->selectState();
    KInstruction *ki = state.pc;   
    
    // @KLEE-SEMu
    //Handle infinite loop, This requires BFS search strategy XXX
//#ifdef SEMU_INFINITE_LOOP_BREAK
    if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
      ks_CheckAndBreakInfinitLoop(state, ks_prevState, ks_initTime);
    }
//#endif
    //~KS
    
    stepInstruction(state);

    executeInstruction(state, ki);
    processTimers(&state, MaxInstructionTime);

    checkMemoryUsage();

    // @KLEE-SEMu
    if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
      if (ks_CheckpointingMainCheck(state, ki, false/*is Seeding*/)) {
        processTimers(nullptr, MaxInstructionTime); //Make sure instruction timer is reset
        continue;  // avoid putting back the state in the searcher bellow "updateStates(&state);"
      }
    }
    //~KS

    updateStates(&state);
  }

  delete searcher;
  searcher = 0;
  
  doDumpStates();
}

std::string Executor::getAddressInfo(ExecutionState &state, 
                                     ref<Expr> address) const{
  std::string Str;
  llvm::raw_string_ostream info(Str);
  info << "\taddress: " << address << "\n";
  uint64_t example;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(address)) {
    example = CE->getZExtValue();
  } else {
    ref<ConstantExpr> value;
    bool success = solver->getValue(state, address, value);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    example = value->getZExtValue();
    info << "\texample: " << example << "\n";
    std::pair< ref<Expr>, ref<Expr> > res = solver->getRange(state, address);
    info << "\trange: [" << res.first << ", " << res.second <<"]\n";
  }
  
  MemoryObject hack((unsigned) example);    
  MemoryMap::iterator lower = state.addressSpace.objects.upper_bound(&hack);
  info << "\tnext: ";
  if (lower==state.addressSpace.objects.end()) {
    info << "none\n";
  } else {
    const MemoryObject *mo = lower->first;
    std::string alloc_info;
    mo->getAllocInfo(alloc_info);
    info << "object at " << mo->address
         << " of size " << mo->size << "\n"
         << "\t\t" << alloc_info << "\n";
  }
  if (lower!=state.addressSpace.objects.begin()) {
    --lower;
    info << "\tprev: ";
    if (lower==state.addressSpace.objects.end()) {
      info << "none\n";
    } else {
      const MemoryObject *mo = lower->first;
      std::string alloc_info;
      mo->getAllocInfo(alloc_info);
      info << "object at " << mo->address 
           << " of size " << mo->size << "\n"
           << "\t\t" << alloc_info << "\n";
    }
  }

  return info.str();
}

void Executor::terminateState(ExecutionState &state) {
  if (replayKTest && replayPosition!=replayKTest->numObjects) {
    klee_warning_once(replayKTest,
                      "replay did not consume all objects in test input.");
  }

  interpreterHandler->incPathsExplored();

  std::vector<ExecutionState *>::iterator it =
      std::find(addedStates.begin(), addedStates.end(), &state);
  if (it==addedStates.end()) {
    state.pc = state.prevPC;

    removedStates.push_back(&state);
  } else {
    // never reached searcher, just delete immediately
    std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it3 = 
      seedMap.find(&state);
    if (it3 != seedMap.end())
      seedMap.erase(it3);
    addedStates.erase(it);
    processTree->remove(state.ptreeNode);
    delete &state;
  }
}

void Executor::terminateStateEarly(ExecutionState &state, 
                                   const Twine &message) {
  // @KLEE-SEMu
  if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
    ks_justTerminatedStates.insert(&state);
    return;
  }
  if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::TESTGEN_MODE) {
    // Do not generate test for seeding mutant, the original will have it
    if (state.ks_mutantID > 0 && state.isTestGenMutSeeding) {
      terminateState(state);
      return;
    }
  }
  //~KS

  if (!OnlyOutputStatesCoveringNew || state.coveredNew ||
      (AlwaysOutputSeeds && seedMap.count(&state)))
    interpreterHandler->processTestCase(state, (message + "\n").str().c_str(),
                                        "early");
  
  terminateState(state);
}

void Executor::terminateStateOnExit(ExecutionState &state) {
  // @KLEE-SEMu
  if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
    ks_justTerminatedStates.insert(&state);
    return;
  }
  if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::TESTGEN_MODE) {
    // Do not generate test for seeding mutant, the original will have it
    if (state.ks_mutantID > 0 && state.isTestGenMutSeeding) {
      terminateState(state);
      return;
    }
  }
  //~KS
  if (!OnlyOutputStatesCoveringNew || state.coveredNew || 
      (AlwaysOutputSeeds && seedMap.count(&state)))
    interpreterHandler->processTestCase(state, 0, 0);
  
  terminateState(state);
}

const InstructionInfo & Executor::getLastNonKleeInternalInstruction(const ExecutionState &state,
    Instruction ** lastInstruction) {
  // unroll the stack of the applications state and find
  // the last instruction which is not inside a KLEE internal function
  ExecutionState::stack_ty::const_reverse_iterator it = state.stack.rbegin(),
      itE = state.stack.rend();

  // don't check beyond the outermost function (i.e. main())
  itE--;

  const InstructionInfo * ii = 0;
  if (kmodule->internalFunctions.count(it->kf->function) == 0){
    ii =  state.prevPC->info;
    *lastInstruction = state.prevPC->inst;
    //  Cannot return yet because even though
    //  it->function is not an internal function it might of
    //  been called from an internal function.
  }

  // Wind up the stack and check if we are in a KLEE internal function.
  // We visit the entire stack because we want to return a CallInstruction
  // that was not reached via any KLEE internal functions.
  for (;it != itE; ++it) {
    // check calling instruction and if it is contained in a KLEE internal function
    const Function * f = (*it->caller).inst->getParent()->getParent();
    if (kmodule->internalFunctions.count(f)){
      ii = 0;
      continue;
    }
    if (!ii){
      ii = (*it->caller).info;
      *lastInstruction = (*it->caller).inst;
    }
  }

  if (!ii) {
    // something went wrong, play safe and return the current instruction info
    *lastInstruction = state.prevPC->inst;
    return *state.prevPC->info;
  }
  return *ii;
}

bool Executor::shouldExitOn(enum TerminateReason termReason) {
  std::vector<TerminateReason>::iterator s = ExitOnErrorType.begin();
  std::vector<TerminateReason>::iterator e = ExitOnErrorType.end();

  for (; s != e; ++s)
    if (termReason == *s)
      return true;

  return false;
}

void Executor::terminateStateOnError(ExecutionState &state,
                                     const llvm::Twine &messaget,
                                     enum TerminateReason termReason,
                                     const char *suffix,
                                     const llvm::Twine &info) {
  std::string message = messaget.str();
  static std::set< std::pair<Instruction*, std::string> > emittedErrors;
  Instruction * lastInst;
  const InstructionInfo &ii = getLastNonKleeInternalInstruction(state, &lastInst);
  
  if (EmitAllErrors ||
      emittedErrors.insert(std::make_pair(lastInst, message)).second) {
    if (ii.file != "") {
      klee_message("ERROR: %s:%d: %s", ii.file.c_str(), ii.line, message.c_str());
    } else {
      klee_message("ERROR: (location information missing) %s", message.c_str());
    }
    if (!EmitAllErrors)
      klee_message("NOTE: now ignoring this error at this location");

    std::string MsgString;
    llvm::raw_string_ostream msg(MsgString);
    msg << "Error: " << message << "\n";
    if (ii.file != "") {
      msg << "File: " << ii.file << "\n";
      msg << "Line: " << ii.line << "\n";
      msg << "assembly.ll line: " << ii.assemblyLine << "\n";
    }
    msg << "Stack: \n";
    state.dumpStack(msg);

    std::string info_str = info.str();
    if (info_str != "")
      msg << "Info: \n" << info_str;

    std::string suffix_buf;
    if (!suffix) {
      suffix_buf = TerminateReasonNames[termReason];
      suffix_buf += ".err";
      suffix = suffix_buf.c_str();
    }

    // @KLEE-SEMu
    if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) 
      ; //do nothing
    else // process test case only for original or bounded symbex mutants in test gen mode
      if (!(ExecutionState::ks_getMode() == ExecutionState::KS_Mode::TESTGEN_MODE && state.ks_mutantID > 0 && state.isTestGenMutSeeding)) 
    //~KS
        interpreterHandler->processTestCase(state, msg.str().c_str(), suffix);
  }
  
  // @KLEE-SEMu
  if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::SEMU_MODE) {
    if (!shouldExitOn(termReason)) {
      ks_justTerminatedStates.insert(&state);
      return;
    }
  }
  //~KS
    
  terminateState(state);

  if (shouldExitOn(termReason))
    haltExecution = true;
}

// XXX shoot me
static const char *okExternalsList[] = { "printf", 
                                         "fprintf", 
                                         "puts",
                                         "getpid" };
static std::set<std::string> okExternals(okExternalsList,
                                         okExternalsList + 
                                         (sizeof(okExternalsList)/sizeof(okExternalsList[0])));

void Executor::callExternalFunction(ExecutionState &state,
                                    KInstruction *target,
                                    Function *function,
                                    std::vector< ref<Expr> > &arguments) {
  // check if specialFunctionHandler wants it
  if (specialFunctionHandler->handle(state, function, target, arguments))
    return;
  
  if (NoExternals && !okExternals.count(function->getName())) {
    klee_warning("Calling not-OK external function : %s\n",
               function->getName().str().c_str());
    terminateStateOnError(state, "externals disallowed", User);
    return;
  }

  // normal external function handling path
  // allocate 128 bits for each argument (+return value) to support fp80's;
  // we could iterate through all the arguments first and determine the exact
  // size we need, but this is faster, and the memory usage isn't significant.
  uint64_t *args = (uint64_t*) alloca(2*sizeof(*args) * (arguments.size() + 1));
  memset(args, 0, 2 * sizeof(*args) * (arguments.size() + 1));
  unsigned wordIndex = 2;
  for (std::vector<ref<Expr> >::iterator ai = arguments.begin(), 
       ae = arguments.end(); ai!=ae; ++ai) {
    if (AllowExternalSymCalls) { // don't bother checking uniqueness
      ref<ConstantExpr> ce;
      bool success = solver->getValue(state, *ai, ce);
      assert(success && "FIXME: Unhandled solver failure");
      (void) success;
      ce->toMemory(&args[wordIndex]);
      wordIndex += (ce->getWidth()+63)/64;
    } else {
      ref<Expr> arg = toUnique(state, *ai);
      if (ConstantExpr *ce = dyn_cast<ConstantExpr>(arg)) {
        // XXX kick toMemory functions from here
        ce->toMemory(&args[wordIndex]);
        wordIndex += (ce->getWidth()+63)/64;
      } else {
        terminateStateOnExecError(state, 
                                  "external call with symbolic argument: " + 
                                  function->getName());
        return;
      }
    }
  }

  state.addressSpace.copyOutConcretes();

  if (!SuppressExternalWarnings) {

    std::string TmpStr;
    llvm::raw_string_ostream os(TmpStr);
    os << "calling external: " << function->getName().str() << "(";
    for (unsigned i=0; i<arguments.size(); i++) {
      os << arguments[i];
      if (i != arguments.size()-1)
	os << ", ";
    }
    os << ")";
    
    if (AllExternalWarnings)
      klee_warning("%s", os.str().c_str());
    else
      klee_warning_once(function, "%s", os.str().c_str());
  }
  
  // @KLEE-SEMu
  bool success;
  // XXX keep this only to function that output (input with new process may need more)
  if (semuForkProcessForSegvExternalCalls && function->getName() == "printf") {
    pid_t my_pid;
    int child_status;
    // Use of vfork, be careful about multi threading
    if ((my_pid = ::vfork()) < 0) {
      llvm::errs() << "SEMU@ERROR: fork failure for external call (to avoid Segmentation fault)";
      exit(1);
    }
    if (my_pid == 0) {
      // Child process 
      _Exit((int) !externalDispatcher->executeCall(function, target->inst, args));
    } else {
      // Parent process 
      ::wait(&child_status);
      if (WIFEXITED(child_status)) {
         const int es = WEXITSTATUS(child_status);
         if (es) {
           //perror ("exited with error code");
           success = false;
         } else {
           //perror ("exited normally");
           success = true;
         }
      } else if (WIFSIGNALED(child_status)) {
         const int signum = WTERMSIG(child_status);
         llvm::errs() << "SEMU@Error: External call child process signaled with signum: " << signum << " !\n";
         assert (false);
         exit (1);
      } else {
         llvm::errs() << "SEMU@Error: External call child process other error!\n";
         assert (false);
         exit (1);
      }
    }
  } else
      success = externalDispatcher->executeCall(function, target->inst, args);
  //~KS
  /* // @KLEE-SEMu  -- COMMENT
  bool success = externalDispatcher->executeCall(function, target->inst, args);
  */ // ~KS
  if (!success) {
    terminateStateOnError(state, "failed external call: " + function->getName(),
                          External);
    return;
  }

  if (!state.addressSpace.copyInConcretes()) {
    terminateStateOnError(state, "external modified read-only object",
                          External);
    return;
  }

  LLVM_TYPE_Q Type *resultType = target->inst->getType();
  if (resultType != Type::getVoidTy(getGlobalContext())) {
    ref<Expr> e = ConstantExpr::fromMemory((void*) args, 
                                           getWidthForLLVMType(resultType));
    bindLocal(target, state, e);
  }
}

/***/

ref<Expr> Executor::replaceReadWithSymbolic(ExecutionState &state, 
                                            ref<Expr> e) {
  unsigned n = interpreterOpts.MakeConcreteSymbolic;
  if (!n || replayKTest || replayPath)
    return e;

  // right now, we don't replace symbolics (is there any reason to?)
  if (!isa<ConstantExpr>(e))
    return e;

  if (n != 1 && random() % n)
    return e;

  // create a new fresh location, assert it is equal to concrete value in e
  // and return it.
  
  static unsigned id;
  const Array *array =
      arrayCache.CreateArray("rrws_arr" + llvm::utostr(++id),
                             Expr::getMinBytesForWidth(e->getWidth()));
  ref<Expr> res = Expr::createTempRead(array, e->getWidth());
  ref<Expr> eq = NotOptimizedExpr::create(EqExpr::create(e, res));
  llvm::errs() << "Making symbolic: " << eq << "\n";
  state.addConstraint(eq);
  return res;
}

ObjectState *Executor::bindObjectInState(ExecutionState &state, 
                                         const MemoryObject *mo,
                                         bool isLocal,
                                         const Array *array) {
  ObjectState *os = array ? new ObjectState(mo, array) : new ObjectState(mo);
  state.addressSpace.bindObject(mo, os);

  // Its possible that multiple bindings of the same mo in the state
  // will put multiple copies on this list, but it doesn't really
  // matter because all we use this list for is to unbind the object
  // on function return.
  if (isLocal)
    state.stack.back().allocas.push_back(mo);

  return os;
}

void Executor::executeAlloc(ExecutionState &state,
                            ref<Expr> size,
                            bool isLocal,
                            KInstruction *target,
                            bool zeroMemory,
                            const ObjectState *reallocFrom) {
  size = toUnique(state, size);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(size)) {
    MemoryObject *mo = memory->allocate(CE->getZExtValue(), isLocal, false, 
                                        state.prevPC->inst);
    if (!mo) {
      bindLocal(target, state, 
                ConstantExpr::alloc(0, Context::get().getPointerWidth()));
    } else {
      ObjectState *os = bindObjectInState(state, mo, isLocal);
      if (zeroMemory) {
        os->initializeToZero();
      } else {
        os->initializeToRandom();
      }
      bindLocal(target, state, mo->getBaseExpr());
      
      if (reallocFrom) {
        unsigned count = std::min(reallocFrom->size, os->size);
        for (unsigned i=0; i<count; i++)
          os->write(i, reallocFrom->read8(i));
        state.addressSpace.unbindObject(reallocFrom->getObject());
      }
    }
  } else {
    // XXX For now we just pick a size. Ideally we would support
    // symbolic sizes fully but even if we don't it would be better to
    // "smartly" pick a value, for example we could fork and pick the
    // min and max values and perhaps some intermediate (reasonable
    // value).
    // 
    // It would also be nice to recognize the case when size has
    // exactly two values and just fork (but we need to get rid of
    // return argument first). This shows up in pcre when llvm
    // collapses the size expression with a select.

    ref<ConstantExpr> example;
    bool success = solver->getValue(state, size, example);
    assert(success && "FIXME: Unhandled solver failure");
    (void) success;
    
    // Try and start with a small example.
    Expr::Width W = example->getWidth();
    while (example->Ugt(ConstantExpr::alloc(128, W))->isTrue()) {
      ref<ConstantExpr> tmp = example->LShr(ConstantExpr::alloc(1, W));
      bool res;
      bool success = solver->mayBeTrue(state, EqExpr::create(tmp, size), res);
      assert(success && "FIXME: Unhandled solver failure");      
      (void) success;
      if (!res)
        break;
      example = tmp;
    }

    StatePair fixedSize = fork(state, EqExpr::create(example, size), true);
    
    if (fixedSize.second) { 
      // Check for exactly two values
      ref<ConstantExpr> tmp;
      bool success = solver->getValue(*fixedSize.second, size, tmp);
      assert(success && "FIXME: Unhandled solver failure");      
      (void) success;
      bool res;
      success = solver->mustBeTrue(*fixedSize.second, 
                                   EqExpr::create(tmp, size),
                                   res);
      assert(success && "FIXME: Unhandled solver failure");      
      (void) success;
      if (res) {
        executeAlloc(*fixedSize.second, tmp, isLocal,
                     target, zeroMemory, reallocFrom);
      } else {
        // See if a *really* big value is possible. If so assume
        // malloc will fail for it, so lets fork and return 0.
        StatePair hugeSize = 
          fork(*fixedSize.second, 
               UltExpr::create(ConstantExpr::alloc(1<<31, W), size), 
               true);
        if (hugeSize.first) {
          klee_message("NOTE: found huge malloc, returning 0");
          bindLocal(target, *hugeSize.first, 
                    ConstantExpr::alloc(0, Context::get().getPointerWidth()));
        }
        
        if (hugeSize.second) {

          std::string Str;
          llvm::raw_string_ostream info(Str);
          ExprPPrinter::printOne(info, "  size expr", size);
          info << "  concretization : " << example << "\n";
          info << "  unbound example: " << tmp << "\n";
          terminateStateOnError(*hugeSize.second, "concretized symbolic size",
                                Model, NULL, info.str());
        }
      }
    }

    if (fixedSize.first) // can be zero when fork fails
      executeAlloc(*fixedSize.first, example, isLocal, 
                   target, zeroMemory, reallocFrom);
  }
}

void Executor::executeFree(ExecutionState &state,
                           ref<Expr> address,
                           KInstruction *target) {
  StatePair zeroPointer = fork(state, Expr::createIsZero(address), true);
  if (zeroPointer.first) {
    if (target)
      bindLocal(target, *zeroPointer.first, Expr::createPointer(0));
  }
  if (zeroPointer.second) { // address != 0
    ExactResolutionList rl;
    resolveExact(*zeroPointer.second, address, rl, "free");
    
    for (Executor::ExactResolutionList::iterator it = rl.begin(), 
           ie = rl.end(); it != ie; ++it) {
      const MemoryObject *mo = it->first.first;
      if (mo->isLocal) {
        terminateStateOnError(*it->second, "free of alloca", Free, NULL,
                              getAddressInfo(*it->second, address));
      } else if (mo->isGlobal) {
        terminateStateOnError(*it->second, "free of global", Free, NULL,
                              getAddressInfo(*it->second, address));
      } else {
        it->second->addressSpace.unbindObject(mo);
        if (target)
          bindLocal(target, *it->second, Expr::createPointer(0));
      }
    }
  }
}

void Executor::resolveExact(ExecutionState &state,
                            ref<Expr> p,
                            ExactResolutionList &results, 
                            const std::string &name) {
  // XXX we may want to be capping this?
  ResolutionList rl;
  state.addressSpace.resolve(state, solver, p, rl);
  
  ExecutionState *unbound = &state;
  for (ResolutionList::iterator it = rl.begin(), ie = rl.end(); 
       it != ie; ++it) {
    ref<Expr> inBounds = EqExpr::create(p, it->first->getBaseExpr());
    
    StatePair branches = fork(*unbound, inBounds, true);
    
    if (branches.first)
      results.push_back(std::make_pair(*it, branches.first));

    unbound = branches.second;
    if (!unbound) // Fork failure
      break;
  }

  if (unbound) {
    terminateStateOnError(*unbound, "memory error: invalid pointer: " + name,
                          Ptr, NULL, getAddressInfo(*unbound, p));
  }
}

void Executor::executeMemoryOperation(ExecutionState &state,
                                      bool isWrite,
                                      ref<Expr> address,
                                      ref<Expr> value /* undef if read */,
                                      KInstruction *target /* undef if write */) {
  Expr::Width type = (isWrite ? value->getWidth() : 
                     getWidthForLLVMType(target->inst->getType()));
  unsigned bytes = Expr::getMinBytesForWidth(type);

  if (SimplifySymIndices) {
    if (!isa<ConstantExpr>(address))
      address = state.constraints.simplifyExpr(address);
    if (isWrite && !isa<ConstantExpr>(value))
      value = state.constraints.simplifyExpr(value);
  }

  // fast path: single in-bounds resolution
  ObjectPair op;
  bool success;
  solver->setTimeout(coreSolverTimeout);
  if (!state.addressSpace.resolveOne(state, solver, address, op, success)) {
    address = toConstant(state, address, "resolveOne failure");
    success = state.addressSpace.resolveOne(cast<ConstantExpr>(address), op);
  }
  solver->setTimeout(0);

  if (success) {
    const MemoryObject *mo = op.first;

    if (MaxSymArraySize && mo->size>=MaxSymArraySize) {
      address = toConstant(state, address, "max-sym-array-size");
    }
    
    ref<Expr> offset = mo->getOffsetExpr(address);

    bool inBounds;
    solver->setTimeout(coreSolverTimeout);
    bool success = solver->mustBeTrue(state, 
                                      mo->getBoundsCheckOffset(offset, bytes),
                                      inBounds);
    solver->setTimeout(0);
    if (!success) {
      state.pc = state.prevPC;
      terminateStateEarly(state, "Query timed out (bounds check).");
      return;
    }

    if (inBounds) {
      const ObjectState *os = op.second;
      if (isWrite) {
        if (os->readOnly) {
          terminateStateOnError(state, "memory error: object read only",
                                ReadOnly);
        } else {
          ObjectState *wos = state.addressSpace.getWriteable(mo, os);
          wos->write(offset, value);
        }          
      } else {
        ref<Expr> result = os->read(offset, type);
        
        if (interpreterOpts.MakeConcreteSymbolic)
          result = replaceReadWithSymbolic(state, result);
        
        bindLocal(target, state, result);
      }

      return;
    }
  } 

  // we are on an error path (no resolution, multiple resolution, one
  // resolution with out of bounds)
  
  ResolutionList rl;  
  solver->setTimeout(coreSolverTimeout);
  bool incomplete = state.addressSpace.resolve(state, solver, address, rl,
                                               0, coreSolverTimeout);
  solver->setTimeout(0);
  
  // XXX there is some query wasteage here. who cares?
  ExecutionState *unbound = &state;
  
  for (ResolutionList::iterator i = rl.begin(), ie = rl.end(); i != ie; ++i) {
    const MemoryObject *mo = i->first;
    const ObjectState *os = i->second;
    ref<Expr> inBounds = mo->getBoundsCheckPointer(address, bytes);
    
    StatePair branches = fork(*unbound, inBounds, true);
    ExecutionState *bound = branches.first;

    // bound can be 0 on failure or overlapped 
    if (bound) {
      if (isWrite) {
        if (os->readOnly) {
          terminateStateOnError(*bound, "memory error: object read only",
                                ReadOnly);
        } else {
          ObjectState *wos = bound->addressSpace.getWriteable(mo, os);
          wos->write(mo->getOffsetExpr(address), value);
        }
      } else {
        ref<Expr> result = os->read(mo->getOffsetExpr(address), type);
        bindLocal(target, *bound, result);
      }
    }

    unbound = branches.second;
    if (!unbound)
      break;
  }
  
  // XXX should we distinguish out of bounds and overlapped cases?
  if (unbound) {
    if (incomplete) {
      terminateStateEarly(*unbound, "Query timed out (resolve).");
    } else {
      terminateStateOnError(*unbound, "memory error: out of bound pointer", Ptr,
                            NULL, getAddressInfo(*unbound, address));
    }
  }
}

void Executor::executeMakeSymbolic(ExecutionState &state, 
                                   const MemoryObject *mo,
                                   const std::string &name) {
  // Create a new object state for the memory object (instead of a copy).
  if (!replayKTest) {
    // Find a unique name for this array.  First try the original name,
    // or if that fails try adding a unique identifier.
    unsigned id = 0;
    std::string uniqueName = name;
    while (!state.arrayNames.insert(uniqueName).second) {
      uniqueName = name + "_" + llvm::utostr(++id);
    }
    const Array *array = arrayCache.CreateArray(uniqueName, mo->size);
    bindObjectInState(state, mo, false, array);
    state.addSymbolic(mo, array);
    
    std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
      seedMap.find(&state);
    if (it!=seedMap.end()) { // In seed mode we need to add this as a
                             // binding.
      for (std::vector<SeedInfo>::iterator siit = it->second.begin(), 
             siie = it->second.end(); siit != siie; ++siit) {
        SeedInfo &si = *siit;
        KTestObject *obj = si.getNextInput(mo, NamedSeedMatching);

        if (!obj) {
          if (ZeroSeedExtension) {
            std::vector<unsigned char> &values = si.assignment.bindings[array];
            values = std::vector<unsigned char>(mo->size, '\0');
          } else if (!AllowSeedExtension) {
            terminateStateOnError(state, "ran out of inputs during seeding",
                                  User);
            break;
          }
        } else {
          if (obj->numBytes != mo->size &&
              ((!(AllowSeedExtension || ZeroSeedExtension)
                && obj->numBytes < mo->size) ||
               (!AllowSeedTruncation && obj->numBytes > mo->size))) {
	    std::stringstream msg;
	    msg << "replace size mismatch: "
          << mo->name << "[" << mo->size << "]"
          << " vs " << obj->name << "[" << obj->numBytes << "]"
          << " in test\n";

            terminateStateOnError(state, msg.str(), User);
            break;
          } else {
            std::vector<unsigned char> &values = si.assignment.bindings[array];
            values.insert(values.begin(), obj->bytes, 
                          obj->bytes + std::min(obj->numBytes, mo->size));
            if (ZeroSeedExtension) {
              for (unsigned i=obj->numBytes; i<mo->size; ++i)
                values.push_back('\0');
            }
          }
        }
      }
    }
  } else {
    ObjectState *os = bindObjectInState(state, mo, false);
    if (replayPosition >= replayKTest->numObjects) {
      terminateStateOnError(state, "replay count mismatch", User);
    } else {
      KTestObject *obj = &replayKTest->objects[replayPosition++];
      if (obj->numBytes != mo->size) {
        terminateStateOnError(state, "replay size mismatch", User);
      } else {
        for (unsigned i=0; i<mo->size; i++)
          os->write8(i, obj->bytes[i]);
      }
    }
  }
}

/***/

void Executor::runFunctionAsMain(Function *f,
				 int argc,
				 char **argv,
				 char **envp) {
  std::vector<ref<Expr> > arguments;

  // force deterministic initialization of memory objects
  srand(1);
  srandom(1);
  
  MemoryObject *argvMO = 0;

  // In order to make uclibc happy and be closer to what the system is
  // doing we lay out the environments at the end of the argv array
  // (both are terminated by a null). There is also a final terminating
  // null that uclibc seems to expect, possibly the ELF header?

  int envc;
  for (envc=0; envp[envc]; ++envc) ;

  unsigned NumPtrBytes = Context::get().getPointerWidth() / 8;
  KFunction *kf = kmodule->functionMap[f];
  assert(kf);
  Function::arg_iterator ai = f->arg_begin(), ae = f->arg_end();
  if (ai!=ae) {
    arguments.push_back(ConstantExpr::alloc(argc, Expr::Int32));

    if (++ai!=ae) {
      argvMO = memory->allocate((argc+1+envc+1+1) * NumPtrBytes, false, true,
                                f->begin()->begin());

      if (!argvMO)
        klee_error("Could not allocate memory for function arguments");

      arguments.push_back(argvMO->getBaseExpr());

      if (++ai!=ae) {
        uint64_t envp_start = argvMO->address + (argc+1)*NumPtrBytes;
        arguments.push_back(Expr::createPointer(envp_start));

        if (++ai!=ae)
          klee_error("invalid main function (expect 0-3 arguments)");
      }
    }
  }

  ExecutionState *state = new ExecutionState(kmodule->functionMap[f]);
  
  if (pathWriter) 
    state->pathOS = pathWriter->open();
  if (symPathWriter) 
    state->symPathOS = symPathWriter->open();


  if (statsTracker)
    statsTracker->framePushed(*state, 0);

  assert(arguments.size() == f->arg_size() && "wrong number of arguments");
  for (unsigned i = 0, e = f->arg_size(); i != e; ++i)
    bindArgument(kf, i, *state, arguments[i]);

  if (argvMO) {
    ObjectState *argvOS = bindObjectInState(*state, argvMO, false);

    for (int i=0; i<argc+1+envc+1+1; i++) {
      if (i==argc || i>=argc+1+envc) {
        // Write NULL pointer
        argvOS->write(i * NumPtrBytes, Expr::createPointer(0));
      } else {
        char *s = i<argc ? argv[i] : envp[i-(argc+1)];
        int j, len = strlen(s);
        
        MemoryObject *arg = memory->allocate(len+1, false, true, state->pc->inst);
        if (!arg)
          klee_error("Could not allocate memory for function arguments");
        ObjectState *os = bindObjectInState(*state, arg, false);
        for (j=0; j<len+1; j++)
          os->write8(j, s[j]);

        // Write pointer to newly allocated and initialised argv/envp c-string
        argvOS->write(i * NumPtrBytes, arg->getBaseExpr());
      }
    }
  }
  
  initializeGlobals(*state);

  processTree = new PTree(state);
  state->ptreeNode = processTree->root;
  run(*state);
  delete processTree;
  processTree = 0;

  // hack to clear memory objects
  delete memory;
  memory = new MemoryManager(NULL);

  globalObjects.clear();
  globalAddresses.clear();

  if (statsTracker)
    statsTracker->done();
}

unsigned Executor::getPathStreamID(const ExecutionState &state) {
  assert(pathWriter);
  return state.pathOS.getID();
}

unsigned Executor::getSymbolicPathStreamID(const ExecutionState &state) {
  assert(symPathWriter);
  return state.symPathOS.getID();
}

void Executor::getConstraintLog(const ExecutionState &state, std::string &res,
                                Interpreter::LogType logFormat) {

  std::ostringstream info;

  switch (logFormat) {
  case STP: {
    Query query(state.constraints, ConstantExpr::alloc(0, Expr::Bool));
    char *log = solver->getConstraintLog(query);
    res = std::string(log);
    free(log);
  } break;

  case KQUERY: {
    std::string Str;
    llvm::raw_string_ostream info(Str);
    ExprPPrinter::printConstraints(info, state.constraints);
    res = info.str();
  } break;

  case SMTLIB2: {
    std::string Str;
    llvm::raw_string_ostream info(Str);
    ExprSMTLIBPrinter printer;
    printer.setOutput(info);
    Query query(state.constraints, ConstantExpr::alloc(0, Expr::Bool));
    printer.setQuery(query);
    printer.generateOutput();
    res = info.str();
  } break;

  default:
    klee_warning("Executor::getConstraintLog() : Log format not supported!");
  }
}

bool Executor::getSymbolicSolution(const ExecutionState &state,
                                   std::vector< 
                                   std::pair<std::string,
                                   std::vector<unsigned char> > >
                                   &res) {
  solver->setTimeout(coreSolverTimeout);

  ExecutionState tmp(state);

  // Go through each byte in every test case and attempt to restrict
  // it to the constraints contained in cexPreferences.  (Note:
  // usually this means trying to make it an ASCII character (0-127)
  // and therefore human readable. It is also possible to customize
  // the preferred constraints.  See test/Features/PreferCex.c for
  // an example) While this process can be very expensive, it can
  // also make understanding individual test cases much easier.
  for (unsigned i = 0; i != state.symbolics.size(); ++i) {
    const MemoryObject *mo = state.symbolics[i].first;
    std::vector< ref<Expr> >::const_iterator pi = 
      mo->cexPreferences.begin(), pie = mo->cexPreferences.end();
    for (; pi != pie; ++pi) {
      bool mustBeTrue;
      // Attempt to bound byte to constraints held in cexPreferences
      bool success = solver->mustBeTrue(tmp, Expr::createIsZero(*pi), 
					mustBeTrue);
      // If it isn't possible to constrain this particular byte in the desired
      // way (normally this would mean that the byte can't be constrained to
      // be between 0 and 127 without making the entire constraint list UNSAT)
      // then just continue on to the next byte.
      if (!success) break;
      // If the particular constraint operated on in this iteration through
      // the loop isn't implied then add it to the list of constraints.
      if (!mustBeTrue) tmp.addConstraint(*pi);
    }
    if (pi!=pie) break;
  }

  std::vector< std::vector<unsigned char> > values;
  std::vector<const Array*> objects;
  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    objects.push_back(state.symbolics[i].second);
  bool success = solver->getInitialValues(tmp, objects, values);
  solver->setTimeout(0);
  if (!success) {
    klee_warning("unable to compute initial values (invalid constraints?)!");
    ExprPPrinter::printQuery(llvm::errs(), state.constraints,
                             ConstantExpr::alloc(0, Expr::Bool));
    return false;
  }
  
  for (unsigned i = 0; i != state.symbolics.size(); ++i)
    res.push_back(std::make_pair(state.symbolics[i].first->name, values[i]));
  return true;
}

void Executor::getCoveredLines(const ExecutionState &state,
                               std::map<const std::string*, std::set<unsigned> > &res) {
  res = state.coveredLines;
}

void Executor::doImpliedValueConcretization(ExecutionState &state,
                                            ref<Expr> e,
                                            ref<ConstantExpr> value) {
  abort(); // FIXME: Broken until we sort out how to do the write back.

  if (DebugCheckForImpliedValues)
    ImpliedValue::checkForImpliedValues(solver->solver, e, value);

  ImpliedValueList results;
  ImpliedValue::getImpliedValues(e, value, results);
  for (ImpliedValueList::iterator it = results.begin(), ie = results.end();
       it != ie; ++it) {
    ReadExpr *re = it->first.get();
    
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(re->index)) {
      // FIXME: This is the sole remaining usage of the Array object
      // variable. Kill me.
      const MemoryObject *mo = 0; //re->updates.root->object;
      const ObjectState *os = state.addressSpace.findObject(mo);

      if (!os) {
        // object has been free'd, no need to concretize (although as
        // in other cases we would like to concretize the outstanding
        // reads, but we have no facility for that yet)
      } else {
        assert(!os->readOnly && 
               "not possible? read only object with static read?");
        ObjectState *wos = state.addressSpace.getWriteable(mo, os);
        wos->write(CE, it->second);
      }
    }
  }
}

Expr::Width Executor::getWidthForLLVMType(LLVM_TYPE_Q llvm::Type *type) const {
  return kmodule->targetData->getTypeSizeInBits(type);
}


/*****************************************************************/
/****************** SEMu Only Methods @ START ********************/
/*****************************************************************/


// @KLEE-SEMu
//#define ENABLE_KLEE_SEMU_DEBUG 1

inline llvm::Instruction * Executor::ks_makeArgSym (Module &module, GlobalVariable * &emptyStrAddr, Instruction *insAfter, Value *memAddr, Type *valtype) {
  llvm::Function *f_make_symbolic = module.getFunction("klee_make_symbolic");
  std::vector<Value *> kms_arguments;
  //TODO: How to handle pointer parameters
  //if (valtype->isPointerTy())
  //  continue;
    
  if (!emptyStrAddr) {
    //IRBuilder<> builder(getGlobalContext());
    assert (!module.getNamedGlobal("KLEE_SEMu__klee_make_symbolic_emptyStr") && "KLEE_SEMu__klee_make_symbolic_emptyStr already existent in module");
    emptyStrAddr = dyn_cast<GlobalVariable>(module.getOrInsertGlobal("KLEE_SEMu__klee_make_symbolic_emptyStr", ArrayType::get(Type::getInt8Ty(getGlobalContext()), 4)));
    emptyStrAddr->setAlignment(1);
    emptyStrAddr->setInitializer(ConstantDataArray::getString(getGlobalContext(), "str")); //arg->getName().size()?arg->getName():"str")); //
    emptyStrAddr->setConstant(true);
    //Value *emptyStr = builder.CreateGlobalStringPtr("", "KLEE_SEMu__klee_make_symbolic_emptyStr");
    //emptyStrAddr = emptyStr;
  }
  kms_arguments.clear();
  kms_arguments.push_back(ConstantInt::get(getGlobalContext(), APInt(32, (uint64_t)(0))));
  kms_arguments.push_back(ConstantInt::get(getGlobalContext(), APInt(32, (uint64_t)(0))));
  Instruction *gepStr = GetElementPtrInst::CreateInBounds (emptyStrAddr, kms_arguments);
  gepStr->insertAfter(insAfter);
  Instruction *bitcast8 = new BitCastInst(memAddr, Type::getInt8PtrTy(getGlobalContext()));
  bitcast8->insertAfter(gepStr);
  kms_arguments.clear();
  kms_arguments.push_back(bitcast8);
  unsigned sizeofVal = 0;
  if (valtype->isPointerTy()) {
    sizeofVal = 8;    //TODO
    //errs()<<"@@@@@#" <<valtype->getPointerAddressSpace()<<"\n";
  } else {
    sizeofVal = valtype->getPrimitiveSizeInBits()/8;
  }
  assert (sizeofVal && "Problem getting variable size (for calling klee_make_symbolic)");
  kms_arguments.push_back(ConstantInt::get(getGlobalContext(), APInt(64, (uint64_t)(sizeofVal), false)));
  kms_arguments.push_back(gepStr);
  Instruction *callKMS = CallInst::Create (f_make_symbolic, kms_arguments);
  callKMS->insertAfter(bitcast8);
  
  return callKMS;
}

//This function Set as symbolic the arguments of the entry function (Maybe for 'main' it is better to use command line sym-args)
// This will insert call to 'klee_make_symbolic' over all the arguments.
void Executor::ks_setInitialSymbolics (/*ExecutionState &state, */Module &module, Function &Func)
{
#if 0
  assert (module.getContext() == getGlobalContext() && "");
#endif
  llvm::Function *f_make_symbolic = module.getFunction("klee_make_symbolic");
  
  //The user already added the klee_make_symbolic, no need to proceed
  if (f_make_symbolic)
    return;
    
  //add klee_make_symbolic into the module
  Constant* cf = module.getOrInsertFunction("klee_make_symbolic",
                                                   Type::getVoidTy(getGlobalContext()),
                                                   Type::getInt8PtrTy(getGlobalContext()),
                                                   Type::getInt64Ty(getGlobalContext()),
                                                   Type::getInt8PtrTy(getGlobalContext()),
                                                   NULL);
  f_make_symbolic = llvm::cast<llvm::Function>(cf);
  
  //assert(f_make_symbolic && "'klee_make_symbolic' is not present in module");   //TODO: klee_make_symbolic should not be declare at tis time, do it here
  GlobalVariable *emptyStrAddr = nullptr;
  
  //XXX
  
  // @
  //FOR NOW WE JUST USE SCANF TO MAKE SYMBOLIC (the variable passed to scanf)
  /*const unsigned symStrLen = 10000;  //Maximum size for a string read from stdin (to help make symbolic)
  Function *scanfFunc = module.getFunction("scanf");
  for (auto &Func: module) {
    for (auto &BB: Func) {
      for (auto &Inst: BB) {
        if (CallInst *callI = dyn_cast<CallInst>(&Inst)) {
          if (callI->getCalledFunction() == scanfFunc) { //TODO: consider indirect, maybe getCalledValue is better
            for (arg: callI->) {
              
            }
          }
        }
      }
    }
  }
  return;*/
  
  // @
  //Globals
  //specialFunctionHandler->handle(state, f_make_symbolic, nullptr, kms_arguments);
  
  // @
  //Params
  //XXX For now the pointer passed to function are not yet verified to work. TODO: Implement Lazy Initialization (JPF)
  assert (!Func.isVarArg() && "Variable argument functions are not yet supported by SEMu");   //TODO: add support
  for (Function::arg_iterator arg = Func.arg_begin(), ae = Func.arg_end(); arg != ae; ++arg) {
    assert (arg->getNumUses() <= 1 && "args are actually used mostly once (to store their value)");
    if (arg->getNumUses() == 1) {
      StoreInst * storeArg = dyn_cast<StoreInst>(*(arg->use_begin()));
      assert(storeArg && "The only use of arg should be store");
      Type *valtype = storeArg->getValueOperand()->getType();
      
      ks_makeArgSym (module, emptyStrAddr, storeArg, storeArg->getPointerOperand(), valtype);
    }
  }
}

// get the list of mutants to use in the execution from semuCandidateMutantsFile 
// If the filename is not the empty string, get the mutants list and remove the mutant not in the list from module
void Executor::ks_FilterMutants (llvm::Module *module) {

  // We must have at least a mutant to run SEMU mid selector represents maxID + 1)
  if (dyn_cast<ConstantInt>(ks_mutantIDSelectorGlobal->getInitializer())->getZExtValue() < 2) {
    klee_error("SEMU@ERROR: The module passed contains no mutant!");
    exit(1);
  }

  std::set<ExecutionState::KS_MutantIDType> cand_mut_ids;
  if (!semuCandidateMutantsFile.empty()) {
    std::ifstream ifs(semuCandidateMutantsFile); 
    if (ifs.is_open()) {
      ExecutionState::KS_MutantIDType tmpid;
      ifs >> tmpid;
      while (ifs.good()) {
        cand_mut_ids.insert(tmpid);
        ifs >> tmpid;
      }
      cand_mut_ids.insert(tmpid); // Just in case the user forget the las new line
      ifs.close();
      if (cand_mut_ids.empty()) {
        klee_error("SEMU@ERROR: The candidate mutant list passed is empty!");
        exit(1);
      }
    } else {
      llvm::errs() << "SEMU@ERROR: Unable to open Mutants Candidate file: " << semuCandidateMutantsFile << "\n";
      assert(false);
      exit(1);
    }
  }

  if (cand_mut_ids.empty()) {
    ks_max_mutant_id = dyn_cast<ConstantInt>(ks_mutantIDSelectorGlobal->getInitializer())->getZExtValue() - 1;
    ks_number_of_mutants = ks_max_mutant_id;
    return;
  }

  
  ks_max_mutant_id = 0;
  ks_number_of_mutants = 0;

  // XXX Fix both the mutant fork function for ID range and the switch
  for (auto &Func: *module) {
    for (auto &BB: Func) {
      for (auto iit = BB.begin(), iie = BB.end(); iit != iie;) {
        llvm::Instruction *Instp = &*(iit++);
        if (auto *calli = llvm::dyn_cast<llvm::CallInst>(Instp)) {
          llvm::Function *f = calli->getCalledFunction();
          if (f == ks_mutantIDSelectorGlobal_Func) {
            uint64_t fromMID = dyn_cast<ConstantInt>(calli->getArgOperand(0))->getZExtValue();
            uint64_t toMID = dyn_cast<ConstantInt>(calli->getArgOperand(1))->getZExtValue();
            assert (fromMID <= toMID && "Invalid mutant range");
            
            std::vector<ExecutionState::KS_MutantIDType> fromsCandIds;
            std::vector<ExecutionState::KS_MutantIDType> tosCandIds;
            bool lastIsCand = false;
            for (ExecutionState::KS_MutantIDType mIds = fromMID; mIds <= toMID; mIds++) {
              if (cand_mut_ids.count(mIds) > 0) {
                if (lastIsCand) {
                  tosCandIds[tosCandIds.size() - 1] = mIds;
                } else {
                  fromsCandIds.push_back(mIds);
                  tosCandIds.push_back(mIds);
                  lastIsCand = true;
                }
              } else {
                lastIsCand = false;
              }
            }

            // modify the range and split by cloning and modifying
            // - Case when all mutants are not canditate: Delete the call instruction
            if (fromsCandIds.size() == 0) {
              calli->eraseFromParent();
            } else { 
              // - Case when at least one non candidate. make clones of call inst for other ranges

              // update max mutant id              
              ks_max_mutant_id = std::max(ks_max_mutant_id, 
                                      *std::max_element(tosCandIds.begin(), tosCandIds.end()));

              for (auto i = 0u; i < fromsCandIds.size() - 1; ++i) {
                ks_number_of_mutants += tosCandIds[i] - fromsCandIds[i] + 1;
                auto *clonei = llvm::dyn_cast<llvm::CallInst>(calli->clone());
                clonei->insertBefore(calli);
                clonei->setArgOperand(0, llvm::ConstantInt::get(clonei->getArgOperand(0)->getType(), fromsCandIds[i]));
                clonei->setArgOperand(1, llvm::ConstantInt::get(clonei->getArgOperand(1)->getType(), tosCandIds[i]));
              }
              ks_number_of_mutants += tosCandIds[tosCandIds.size() - 1] - fromsCandIds[fromsCandIds.size() - 1] + 1;
              calli->setArgOperand(0, llvm::ConstantInt::get(calli->getArgOperand(0)->getType(), fromsCandIds[fromsCandIds.size() - 1]));
              calli->setArgOperand(1, llvm::ConstantInt::get(calli->getArgOperand(1)->getType(), tosCandIds[tosCandIds.size() - 1]));
            }
              
          }
        } else if (auto *switchi = llvm::dyn_cast<llvm::SwitchInst>(Instp)) {
          if (auto *ld = llvm::dyn_cast<llvm::LoadInst>(switchi->getCondition())) {
            if (ld->getOperand(0) == ks_mutantIDSelectorGlobal) {
              std::vector<llvm::ConstantInt *> noncand_cases;
              for (llvm::SwitchInst::CaseIt i = switchi->case_begin(),
                                            e = switchi->case_end();
                   i != e; ++i) {
                auto *mutIDConstInt = i.getCaseValue();
                if (cand_mut_ids.count(mutIDConstInt->getZExtValue()) == 0)
                  noncand_cases.push_back(mutIDConstInt);  // to be removed later
              }
              for (auto *caseval : noncand_cases) {
                llvm::SwitchInst::CaseIt cit = switchi->findCaseValue(caseval);
                switchi->removeCase(cit);
              }
            }
          }
        } 
      } // For Inst ...
    } // For BB ...
  } // For Func ...

  // XXX: do not Update the number of mutants in Global mutant ID selectod For it represents HighestID + 1
}

void Executor::ks_mutationPointBranching(ExecutionState &state, 
              std::vector<uint64_t> &mut_IDs) {
  TimerStatIncrementer timer(stats::forkTime);
  unsigned N = mut_IDs.size();
  assert(N);

  if (MaxForks!=~0u && stats::forks >= MaxForks) {
    
    llvm::errs() << "!! Mutants Not Processed due to MaxForks: ";
    for (std::vector<uint64_t>::iterator it = mut_IDs.begin(),
                                       ie = mut_IDs.end(); it != ie; ++it) {
      ExprPPrinter::printSingleExpr(llvm::errs(), ConstantExpr::create(*it, 32));
      llvm::errs() << " ";
    }
    llvm::errs() << "\n";
    return;
    
    /*unsigned next = theRNG.getInt32() % N;
    for (unsigned i=0; i<N; ++i) {
      if (i == next) {
        result.push_back(&state);
      } else {
        result.push_back(NULL);
      }
    }*/
  } else {
    stats::forks += N-1;
    //addedStates.push_back(&state);

    // Mutants just created thus ks_hasToReachPostMutationPoint set to true
    // The mutants created will inherit the value
    if (mut_IDs.size() > 0) {
      state.ks_hasToReachPostMutationPoint = true;
      state.ks_startdepth = state.depth;
    }

    // create the mutants states
    for (std::vector<uint64_t>::iterator it = mut_IDs.begin(),
                                   ie = mut_IDs.end(); it != ie; ++it) {
      // In the test generation mode, do not even generate more mutants 
      // If reached max number of tests
      if (ks_outputTestsCases) {
        auto mapit = mutants2gentestsNum.find(*it);
        if (mapit != mutants2gentestsNum.end() 
                    && mapit->second >= semuMaxNumTestGenPerMutant) {
          // Stop
          continue;
        }
      }

      ExecutionState *ns = state.ks_branchMut();
      addedStates.push_back(ns);
      //result.push_back(ns);
      state.ptreeNode->data = 0;
      std::pair<PTree::Node*,PTree::Node*> res = processTree->split(state.ptreeNode, ns, &state);
      ns->ptreeNode = res.first;
      state.ptreeNode = res.second;
      
      executeMemoryOperation (*ns, true, evalConstant(ks_mutantIDSelectorGlobal), ConstantExpr::create(*it, 32), 0);    //Mutant IDs are 32 bit unsigned int
      ns->ks_mutantID = *it;
      
      ns->ks_originalMutSisterStates = state.ks_curBranchTreeNode;

      // Update active
      (state.ks_numberActiveCmpMutants)++;

      // Free useless space
      ns->ks_VisitedMutPointsSet.clear();
      ns->ks_VisitedMutantsSet.clear();

      // On test generation mode, the newly seen mutant is shadowing original
      // Thus is in seed mode (This is KLEE Shadow implementation (XXX))
      if (ExecutionState::ks_getMode() == ExecutionState::KS_Mode::TESTGEN_MODE)
        ns->isTestGenMutSeeding = true;

      // Handle seed phase. Insert mutant in seedMap with same seed as original
      std::map< ExecutionState*, std::vector<SeedInfo> >::iterator sm_it = 
        seedMap.find(&state);
      if (sm_it != seedMap.end()) {
        seedMap[ns] = sm_it->second;
      }
    }
    // No new mutant created because they reach their limit of number of tests
    if (state.ks_numberActiveCmpMutants == 0 && state.ks_VisitedMutantsSet.size() >= ks_number_of_mutants) {
      // remove from treenode
      state.ks_curBranchTreeNode->exState = nullptr;
      // early terminate
      terminateStateEarly(state, "Original has no possible mutant left");
    }
  }
}


bool Executor::ks_checkfeasiblewithsolver(ExecutionState &state, ref<Expr> bool_expr) {
  bool result;
  bool success = solver->mayBeTrue(state, bool_expr, result);
  assert(success && "KS: Unhandled solver failure");
  (void) success;
  return result;
}

////>
// TODO TODO: Handle state comparison in here
inline bool Executor::ks_outEnvCallDiff (const ExecutionState &a, const ExecutionState &b, std::vector<ref<Expr>> &inStateDiffExp, KScheckFeasible &feasibleChecker) {
  CallInst *acins = dyn_cast<CallInst>(a.pc->inst);
  CallInst *bcins = dyn_cast<CallInst>(b.pc->inst);
  if (!(acins && bcins)) {
    // They are certainly different. Insert true to show that
    inStateDiffExp.push_back(ConstantExpr::alloc(1, Expr::Bool));
    return true;
  }
  if (acins->getCalledFunction() != bcins->getCalledFunction()) { //TODO: consider indirect, maybe getCalledValue is better
    // They are certainly different. Insert true to show that
    inStateDiffExp.push_back(ConstantExpr::alloc(1, Expr::Bool));
    return true;
  }
  if (acins->getNumArgOperands () != bcins->getNumArgOperands ()) {
    // They are certainly different. Insert true to show that
    inStateDiffExp.push_back(ConstantExpr::alloc(1, Expr::Bool));
    return true;
  }
  unsigned numArgs = acins->getNumArgOperands();
  for (unsigned j=0; j<numArgs; ++j) {
    ref<Expr> aArg = eval(&*(a.pc), j+1, const_cast<ExecutionState &>(a)).value;
    ref<Expr> bArg = eval(&*(b.pc), j+1, const_cast<ExecutionState &>(b)).value;
    if (aArg.compare(bArg)) {
#ifdef ENABLE_KLEE_SEMU_DEBUG
      llvm::errs() << "--> External call args differ.\n";
#endif
      // the if to make sure that args expressions are of same type to avoid expr assert
      if (aArg->getWidth()==bArg->getWidth()) {
        ref<Expr> tmpexpr = NeExpr::create(aArg, bArg);
        if (feasibleChecker.isFeasible(tmpexpr)) 
          inStateDiffExp.push_back(tmpexpr);    //XXX: need to do 'or' of all the diff found hre befre returning true?
      } else {
        inStateDiffExp.push_back(ConstantExpr::alloc(1, Expr::Bool));
      }
      //return true;
    }
  }
  if (!inStateDiffExp.empty())
    return true;
  return false;
}
  
// If there is already an env call on the stack (previously checked) 
// Do not check anymore
// XXX this is checked befor KLEE executes the corresponding call instruction
inline bool Executor::ks_isOutEnvCallInvoke (Instruction *cii) {
  static const std::set<std::string> outEnvFuncs = {"printf", "vprintf" "puts", 
                                              "putchar", "putc", "fprintf", 
                                              "vfprintf", "write", "fwrite", 
                                              "fputs", "fputs_unlocked", 
                                              "putchar_unlocked", "fputc", 
                                              "fflush", "perror", "assert", 
                                              "exit", "_exit", "abort", 
                                              "syscall"};
  Function *f = nullptr;
  if (auto *ci = llvm::dyn_cast<llvm::CallInst>(cii))
    f = ci->getCalledFunction();  //TODO: consider indirect, maybe getCalledValue is better
  else if (auto *ii = llvm::dyn_cast<llvm::InvokeInst>(cii))
    f = ii->getCalledFunction();  //TODO: consider indirect, maybe getCalledValue is better
  else
    return false;

  // Out env must be declaration only
  /*static std::set<std::string> tmpextern; //DBG*/
  if (f && f->isDeclaration()) {
    switch(f->getIntrinsicID()) {
      case Intrinsic::trap:
        return true;
      case Intrinsic::not_intrinsic:
        if (outEnvFuncs.count(f->getName()) || ks_customOutEnvFuncNames.count(f->getName()))
          return true;
        break;
      default:
        ;
    }  
  }
  return false;
}

// In the case of call, check the next to execute instruction, which should be state.pc
inline bool Executor::ks_nextIsOutEnv (ExecutionState &state) {
  //if ((uint64_t)state.pc->inst==1) {state.prevPC->inst->getParent()->dump();state.prevPC->inst->dump();} 
  // Is the next instruction to execute an external call that change output
  if (! semuNoConsiderOutEnvForDiffs) {
    if (llvm::dyn_cast_or_null<llvm::UnreachableInst>(state.prevPC->inst) 
                                                                == nullptr) {
      if (ks_isOutEnvCallInvoke(state.pc->inst)) {
        return true;
      }
    }
  }
  return false;
}

inline bool Executor::ks_reachedAMutant(KInstruction *ki) {
  if (ki->inst->getOpcode() == Instruction::Call) {
    Function *f = dyn_cast<CallInst>(ki->inst)->getCalledFunction();
    if (f == ks_mutantIDSelectorGlobal_Func)
      return true;
  }
  return false;
}

#ifdef SEMU_RELMUT_PRED_ENABLED
inline bool Executor::ks_reachedAnOldNew(KInstruction *ki) {
  if (ki->inst->getOpcode() == Instruction::Call) {
    Function *f = dyn_cast<CallInst>(ki->inst)->getCalledFunction();
    if (f == ks_klee_change_function)
      return true;
  }
  return false;
}
#endif

// Check whether the ki is a post mutation point of the state (0) for original
bool Executor::ks_checkAtPostMutationPoint(ExecutionState &state, KInstruction *ki) {
  bool ret = false;
  KInstruction * cur_ki = ki;
  Instruction *i;
  
  if (!state.ks_hasToReachPostMutationPoint) {
    return ret;
  }
  
  if (semuDisableCheckAtPostMutationPoint) {
    state.ks_hasToReachPostMutationPoint = false; 
    return ret;
  } 
  
  do {
    i = cur_ki->inst;
    if (i->getOpcode() == Instruction::Call) {
      Function *f = dyn_cast<CallInst>(i)->getCalledFunction();
      if (f == ks_postMutationPoint_Func) {
        uint64_t fromMID = dyn_cast<ConstantInt>(dyn_cast<CallInst>(i)->getArgOperand(0))->getZExtValue();
        uint64_t toMID = dyn_cast<ConstantInt>(dyn_cast<CallInst>(i)->getArgOperand(1))->getZExtValue();
        if (state.ks_mutantID >= fromMID && state.ks_mutantID <= toMID)
          ret = true;
        // look for the last call in the BB 
        // (since the call was added by mutation tool, there must be following Instruction in BB)
        if (cur_ki != ki) // not the first loop
          stepInstruction(state);
        cur_ki = state.pc;
      } else {
        break;
      }
    } else {
      break;
    }
  } while(true);
  // Handle the cases where the post mutation point
  // function is not inserted (return) or call to
  // a function within mutated statement.
  // XXX if the post mutation is not seen after first
  // fork, we disable post mutation point for the state.
  if (!ret && state.depth > state.ks_startdepth) {
    state.ks_hasToReachPostMutationPoint = false;
  }
  return ret;
}

inline bool Executor::ks_reachedCheckNextDepth(ExecutionState &state) {
  if (state.depth > ks_nextDepthID)
    return true;
  return false;
}

inline bool Executor::ks_reachedCheckMaxDepth(ExecutionState &state) {
  if (state.depth - state.ks_startdepth > semuMaxDepthWP)
    return true;
  return false;
}

// This function must be called after 'stepInstruction' and 'executeInstruction' function call, which respectively
// have set state.pc to next instruction to execute and state.prevPC to the to the just executed instruction 'ki'
// - In the case of return, 'ki' is used and should be the return instruction (checked after return instruction execution)
// ** We also have the option of limiting symbolic exec depth for mutants and such depth would be a watchpoint.
inline bool Executor::ks_watchPointReached (ExecutionState &state, KInstruction *ki) {
#ifdef SEMU_RELMUT_PRED_ENABLED
  // if not reached oldnew, not watchedpoint
  if (state.ks_old_new == 0)
    return false;
#endif
  // No need to check return of non entry func.
  // Change this to enable/disable intermediate return
  const bool noIntermediateRet = true; 

  if (ki->inst->getOpcode() == Instruction::Ret) {
    //ks_watchpoint = false;
    if (! (noIntermediateRet && state.ks_checkRetFunctionEntry01NonEntryNeg() < 0))
      return true;
    return false;
  } 
  return ks_reachedCheckMaxDepth(state);
}
///


inline void Executor::ks_fixTerminatedChildren(ExecutionState *es, llvm::SmallPtrSet<ExecutionState *, 5> const &toremove, 
				                          bool terminateLooseOrig, bool removeLooseOrigFromAddedPreTerm) {
  if (toremove.empty())
    return;

  ks_fixTerminatedChildrenRecursive (es, toremove);

  // let a child mutant state be the new parent of the group in case this parent terminated
  // XXX at this point, all terminated children are removed from its chindren set
  if (toremove.count(es) > 0) {
    if (!es->ks_childrenStates.empty()) {
      auto *newParent = *(es->ks_childrenStates.begin());
      es->ks_childrenStates.erase(newParent);
    
      newParent->ks_childrenStates.insert(es->ks_childrenStates.begin(), es->ks_childrenStates.end());
      assert (newParent->ks_originalMutSisterStates == nullptr);
      newParent->ks_originalMutSisterStates = es->ks_originalMutSisterStates;

      es->ks_originalMutSisterStates = nullptr;
      es->ks_childrenStates.clear();
    } else {
      // No state of this mutant at this sub tree remains, decrease active count
      // And remove the corresponding original if no more mutant on the way (active and upcoming)
      std::vector<ExecutionState *> _sub_states;
      llvm::SmallPtrSet<ExecutionState *, 5>  _toremove;
      es->ks_originalMutSisterStates->getAllStates(_sub_states);
      for (ExecutionState *_s: _sub_states) {
        assert (_s->ks_numberActiveCmpMutants > 0 && "BUG, ks_numberActiveCmpMutants must be > 0");
        (_s->ks_numberActiveCmpMutants)--;
        if (_s->ks_numberActiveCmpMutants == 0 
                && _s->ks_VisitedMutantsSet.size() >= ks_number_of_mutants) {
          _toremove.insert(_s);
          // Add to ks_terminatedBeforeWP
          ks_moveIntoTerminatedBeforeWP(_s);
        }
      }
      // Fixup
      if (_toremove.size() > 0) {
        if(!semuQuiet)
          llvm::errs() << "# SEMU@Status: Removing " << _toremove.size() 
              << " original states with no possible subtree mutant left.\n";
        es->ks_originalMutSisterStates->ks_cleanTerminatedOriginals(_toremove);
      }

      if (removeLooseOrigFromAddedPreTerm) {
        for (auto *es: _toremove) {
          std::vector<ExecutionState *>::iterator it =
                        std::find(addedStates.begin(), addedStates.end(), es);
          if (it != addedStates.end())
            addedStates.erase(it);
        }
      }
            
      if (terminateLooseOrig) {
        // If post compare, make sure to add all original states removed during ks_fixTerminatedChildren
        // into ks_justTerminatedStates
        for (auto *es: _toremove) {
          //std::vector<ExecutionState *>::iterator it =
          //          std::find(addedStates.begin(), addedStates.end(), es);
          //if (it != addedStates.end())
          //  addedStates.erase(it);

          if (ks_terminatedBeforeWP.count(es) > 0)
            ks_terminatedBeforeWP.erase(es);
          terminateState(*es);
        }
      }
    }
  }
}

void Executor::ks_fixTerminatedChildrenRecursive (ExecutionState *pes, llvm::SmallPtrSet<ExecutionState *, 5> const &toremove) {
  // Verify
  if (pes->ks_mutantID > ks_max_mutant_id) {
    llvm::errs() << "\nSEMU@error: Invalid mutant ID. Potential memory corruption (BUG)."
                 << " The value must be no greater than " << ks_max_mutant_id
                 << ", but got " << pes->ks_mutantID << "\n\n";
    assert(false);
  }

  if (pes->ks_childrenStates.empty())
    return;

  std::vector<ExecutionState *> children(pes->ks_childrenStates.begin(), pes->ks_childrenStates.end());
  for (ExecutionState *ces: children) {
    ks_fixTerminatedChildrenRecursive(ces, toremove);
    if (toremove.count(ces) > 0) {
      pes->ks_childrenStates.erase(ces);
      if (! ces->ks_childrenStates.empty()) {
        auto *newparent = *(ces->ks_childrenStates.begin());
        pes->ks_childrenStates.insert(newparent);
        
        ces->ks_childrenStates.erase(newparent);
        newparent->ks_childrenStates.insert(ces->ks_childrenStates.begin(), ces->ks_childrenStates.end());

        ces->ks_originalMutSisterStates = nullptr;
        ces->ks_childrenStates.clear();
      }
    }
  }
}

void Executor::ks_moveIntoTerminatedBeforeWP(ExecutionState *es) {
  if (ks_reachedWatchPoint.count(es) > 0) {
    ks_terminatedBeforeWP.insert(es);
    ks_reachedWatchPoint.erase(es);
  } else if (ks_ongoingExecutionAtWP.count(es) > 0) {
    ks_terminatedBeforeWP.insert(es);
    ks_ongoingExecutionAtWP.erase(es);
  } else if (ks_atPointPostMutation.count(es) > 0) {
    ks_terminatedBeforeWP.insert(es);
    ks_atPointPostMutation.erase(es);
  } else if (ks_reachedOutEnv.count(es) > 0) {
    ks_terminatedBeforeWP.insert(es);
    ks_reachedOutEnv.erase(es);
  } else if (ks_terminatedBeforeWP.count(es) == 0) {
    //llvm::errs() << "Error: Execution state in None of the depth check sets (watchpoint, terminated, ...)\n";
    //throw "Executor::ks_moveIntoTerminatedBeforeWP exception";
  }
}

void Executor::ks_terminateSubtreeMutants(ExecutionState *pes) {
  for (ExecutionState *ces: pes->ks_childrenStates) {
    ks_terminateSubtreeMutants(ces);
    ks_moveIntoTerminatedBeforeWP(ces);
  }
  ks_moveIntoTerminatedBeforeWP(pes);
}

void Executor::ks_getMutParentStates(std::vector<ExecutionState *> &mutParentStates) {
  mutParentStates.clear();
  //TODO TODO: Make efficient
  for(ExecutionState *es: states) {
    if (es->ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  /*
  for(ExecutionState *es: ks_atPointPostMutation) {
    if (es->ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: ks_ongoingExecutionAtWP) {
    if (es->ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: ks_reachedOutEnv) {
    if (es->ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: ks_reachedWatchPoint) {
    if (es->ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: ks_terminatedBeforeWP) {
    if (es->ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: addedStates) {
    if (es->ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: removedStates) {
    if (es->ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }*/
}

void Executor::ks_compareStates (std::vector<ExecutionState *> &remainStates, bool outEnvOnly, bool postMutOnly) {
  assert (!(outEnvOnly & postMutOnly) && 
          "must not have both outEnvOnly and postMutOnly enabled simultaneously");

  llvm::SmallPtrSet<ExecutionState *, 5> postMutOnly_hasdiff;

  std::vector<ExecutionState *> mutParentStates;
  ks_getMutParentStates(mutParentStates);

  std::sort(mutParentStates.begin(), mutParentStates.end(),
            [](const ExecutionState *a, const ExecutionState *b)
            {
                return a->ks_mutantID < b->ks_mutantID;
            });
  
  std::map<ExecutionState *, ref<Expr>> origSuffConstr;
  
  std::vector<ExecutionState *> correspOriginals;
  
  for (ExecutionState *es: mutParentStates) {
    correspOriginals.clear();
    es->ks_originalMutSisterStates->getAllStates(correspOriginals);
    //assert (correspOriginals.size() > 0 && (std::string("Error: Empty original state list - Comparing with mutant ID: ")+std::to_string(es->ks_mutantID)).c_str());

    // TODO: CHECK WHY ORIGINAL FINISHES FIRST
    //for(auto xy: correspOriginals)llvm::errs()<<" "<<xy->ks_mutantID;llvm::errs()<<" **\n";
    //llvm::errs()<<mutParentStates.size()<<" ~~~~~~\n";
    
    // This may happend because an original ma terminate before its corresponding mutant
    // and thus, it will be removed from ks_terminatedBeforeWP at a "next check" before the mutant's watch point
    if (correspOriginals.empty()) {
      if (outEnvOnly) {
        // no original left, no comparison to be done. Let the generation to the checkpoint
        remainStates.clear();
        remainStates.insert(remainStates.begin(), ks_reachedOutEnv.begin(), ks_reachedOutEnv.end());
        return;
      }
      
      // No need to continue with the mutants since original finished
      // Remove the mutants of the subtree from ks_reachedWatchPoint..., add to terminated
#ifdef SEMU_RELMUT_PRED_ENABLED
      // RELMUT: Do not remove induced mutants if old version (the mutants might sill be needed). Of course, in case they are no more needed
      // they will be explored in vain. 
      // FIXME: find a better way to avoid exploration in vain in this case
      if (es->ks_old_new == -1)
#endif
      ks_terminateSubtreeMutants(es);
      //continue;

      // use a sisstate null and the path condition true to just generate the test for the mutant
      origSuffConstr.clear();
      correspOriginals.push_back(nullptr);
      origSuffConstr.insert(std::pair<ExecutionState *, ref<Expr>>(nullptr, ConstantExpr::alloc(1, Expr::Bool)));
    } else {
      //compute constraint for each original (only if the previous is descendent of this). for this, if any leave is common and the the previous subtree is bigger or equal to this, then 
      if (origSuffConstr.find(correspOriginals.at(0)) == origSuffConstr.end() || correspOriginals.size() > origSuffConstr.size()) {
        origSuffConstr.clear();
        for (ExecutionState *tmpes: correspOriginals) {
          ref<Expr> sconstr = ConstantExpr::alloc(1, Expr::Bool);
          for (ConstraintManager::constraint_iterator it = tmpes->constraints.begin(), 
                            ie = tmpes->constraints.end(); it != ie; ++it) {
            sconstr = AndExpr::create(sconstr, *it);
          }
          origSuffConstr.insert(std::pair<ExecutionState *, ref<Expr>>(tmpes, sconstr));
        }
      }
    }

    if (ks_compareRecursive (es, correspOriginals, origSuffConstr, outEnvOnly, postMutOnly, postMutOnly_hasdiff)) {
      // terminate all the states of this mutant by removing them from ks_reachedWatchPoint and adding them to ks_terminatedBeforeWP
      // XXX: For a mutant do we need to generate test for all difference with original or only one (mutant forked from different original have different test generated)?
      // TODO improve this
      /*for(ExecutionState *ites: ks_reachedWatchPoint) {
        if (ites->ks_mutantID == es->ks_mutantID) {
          ks_reachedWatchPoint.erase(ites);
          ks_terminatedBeforeWP.insert(ites); 
        }
      }*/
    }

    /*if (!outEnvOnly) {
      // Remove mutants states that are terminated form their parent's 'children set'
      ks_fixTerminatedChildrenRecursive (es);

      // let a child mutant state be the new parent of the group in case this parent terminated
      // XXX at this point, all terminated children are removed from its chindren set
      if (ks_terminatedBeforeWP.count(es) > 0 && !es->ks_childrenStates.empty()) {
        auto *newParent = *(es->ks_childrenStates.begin());
        es->ks_childrenStates.erase(newParent);
        
        newParent->ks_childrenStates.insert(es->ks_childrenStates.begin(), es->ks_childrenStates.end());
        assert (newParent->ks_originalMutSisterStates == nullptr);
        newParent->ks_originalMutSisterStates = es->ks_originalMutSisterStates;
      }
    }*/
  }

  if (!outEnvOnly) {
    if (!postMutOnly) {
      // Fix original terminated
      if (! mutParentStates.empty()) {
        auto *cands = mutParentStates.front();
        auto *topParent = cands->ks_originalMutSisterStates;
        while (topParent->parent != nullptr)
          topParent = topParent->parent;
        topParent->ks_cleanTerminatedOriginals(ks_terminatedBeforeWP);
      }

    //Temporary
    //remainStates.clear();
    //remainStates.insert(remainStates.begin(), ks_reachedOutEnv.begin(), ks_reachedOutEnv.end());

      // We reached Checkpoint, terminate all mutant states so far (originals are never in reached Checkpoint)
      for (auto *s: ks_reachedWatchPoint) {
        //if (s->ks_mutantID == 0)
          //remainStates.push_back(s);
        //else
        ks_terminatedBeforeWP.insert(s);
      }

      // Remove mutants states that are terminated form their parent's 'children set'
      // And set a new mutant parent if cur parent is terminated
      for (ExecutionState *es: mutParentStates) {
        // Remove mutants states that are terminated form their parent's 'children set'
        ks_fixTerminatedChildren(es, ks_terminatedBeforeWP, false);
      }
      remainStates.clear();
    } else {
      llvm::SmallPtrSet<ExecutionState *, 5> toremove;
      llvm::SmallPtrSet<ExecutionState *, 5> originals;
      for (auto *s: ks_atPointPostMutation) {
        if(s->ks_mutantID != 0 && postMutOnly_hasdiff.count(s) == 0)
          toremove.insert(s);
      }
      // Remove mutants states that are terminated form their parent's 'children set'
      // And set a new mutant parent if cur parent is terminated
      for (ExecutionState *es: mutParentStates) {
        // Remove mutants states that are terminated form their parent's 'children set'
        ks_fixTerminatedChildren(es, toremove, true);
      }

      // put this here because some original may be moved out in ks_fixTerminatedChildren
      for (auto *s: ks_atPointPostMutation) {
        if (s->ks_mutantID == 0)
          originals.insert(s);
      }
        
      // Get stats
      ks_numberOfMutantStatesCheckedAtMutationPoint += toremove.size() + postMutOnly_hasdiff.size();
      ks_numberOfMutantStatesDiscardedAtMutationPoint += toremove.size();

      // terminate the state in toremove
      for (auto *s: toremove) {
        //s->pc = s->prevPC;
        terminateState(*s);
      }

      remainStates.clear();
      remainStates.insert(remainStates.begin(), originals.begin(), originals.end());
      remainStates.insert(remainStates.end(), postMutOnly_hasdiff.begin(), postMutOnly_hasdiff.end());
    }
  } else {
    remainStates.clear();
    remainStates.insert(remainStates.begin(), ks_reachedOutEnv.begin(), ks_reachedOutEnv.end());
  }
  //ks_watchpoint = false;  //temporary
}

bool Executor::ks_diffTwoStates (ExecutionState *mState, ExecutionState *mSisState, 
                                  ref<Expr> &origpathPrefix, 
                                  bool outEnvOnly, int &sDiff, 
                                  std::vector<ref<Expr>> &inStateDiffExp) {
  /**
   * Return the result of the solver.
   */
  static const bool usethesolverfordiff = true;

  inStateDiffExp.clear();
  bool result;
  bool success = solver->mayBeTrue(*mState, origpathPrefix, result);
  assert(success && "KS: Unhandled solver failure");
  (void) success;
  if (result) {
    // Clear diff expr list
    KScheckFeasible feasibleChecker(this, mState, origpathPrefix, usethesolverfordiff);

    // compare these 
    if (mSisState == nullptr) {
      // Do not have corresponding original anymore.
      sDiff |= ExecutionState::KS_StateDiff_t::ksPC_DIFF;
    } else {
      if (mState->ks_numberOfOutEnvSeen != mSisState->ks_numberOfOutEnvSeen) {
        sDiff |= ExecutionState::KS_StateDiff_t::ksOUTENV_DIFF;
      }

      if (outEnvOnly &&  (KInstruction*)(mState->pc) && ks_isOutEnvCallInvoke(mState->pc->inst)) {
        sDiff |= ks_outEnvCallDiff (*mState, *mSisState, inStateDiffExp, feasibleChecker) ? 
                              ExecutionState::KS_StateDiff_t::ksOUTENV_DIFF :
                                                 ExecutionState::KS_StateDiff_t::ksNO_DIFF;
        // XXX: we also compare states (ks_compareStateWith) here or not?

        if (!ExecutionState::ks_isNoDiff(sDiff))
          sDiff |= mState->ks_compareStateWith(*mSisState, ks_mutantIDSelectorGlobal, 
#ifdef SEMU_RELMUT_PRED_ENABLED
                                               ks_isOldVersionGlobal,
#endif
                                               inStateDiffExp, &feasibleChecker, false/*post...*/);
      } else {
        sDiff |= mState->ks_compareStateWith(*mSisState, ks_mutantIDSelectorGlobal, 
#ifdef SEMU_RELMUT_PRED_ENABLED
                                               ks_isOldVersionGlobal,
#endif
                                             inStateDiffExp, &feasibleChecker, true/*post...*/);
        // XXX if mutant terminated and not original or vice versa, set the main return diff
        // TODO: Meke this more efficient
        if (ks_terminatedBeforeWP.count(mSisState) != ks_terminatedBeforeWP.count(mState))
          sDiff |= ExecutionState::KS_StateDiff_t::ksRETCODE_DIFF_MAINFUNC;
      }
    }
    // make sure that the sDiff is not having an error. If error, abort
    ExecutionState::ks_checkNoDiffError(sDiff, mState->ks_mutantID);
  }
  return result;
}

void Executor::ks_createDiffExpr(ExecutionState *mState, ref<Expr> &insdiff, 
                                  ref<Expr> &origpathPrefix,
                                  std::vector<ref<Expr>> &inStateDiffExp) {

  /**
   *  create constraint and return in `insDiff`
   */
  // TODO: improve this so the test constraint is smaller: remove condition for variable
  // that are not of the output (not returned nor printed)
  /**/ //TODO: Should this be included? in introduce error in some test generation (during solving)
  //llvm::errs() << "============================= begin\n";
  //Try with and of all conditions
  insdiff = AndExpr::create(ConstantExpr::alloc(1, Expr::Bool), origpathPrefix);
  for (auto &expr: inStateDiffExp) { 
    //expr->dump(); 
    insdiff = AndExpr::create(insdiff, expr);
  }
  if (!ks_checkfeasiblewithsolver(*mState, insdiff)) {
    //use OR since it is infeasible with and (OR must be feasible because each clause is independently feasible)
    insdiff = ConstantExpr::alloc(0, Expr::Bool);
    for (auto &expr: inStateDiffExp)
      insdiff = OrExpr::create(insdiff, expr);
    insdiff = AndExpr::create(insdiff, origpathPrefix);
  }
  //llvm::errs() << "============================= end\n";
  /*  */
}

#ifdef SEMU_RELMUT_PRED_ENABLED
bool Executor::ks_mutantPrePostDiff (ExecutionState *mState, bool outEnvOnly, 
                                      std::vector<ref<Expr>> &inStateDiffExp, 
                                      ref<Expr> &insdiff) {
  bool can_diff = false;
  
  // 1. Find all states with same mutant_ID as `mState` 
  std::vector<ExecutionState*> preMutStateList;
  // XXX: TODO: improve this:w
  for (auto *s: states) {
    if (s->ks_mutantID == mState->ks_mutantID && s->ks_old_new < 0) 
      preMutStateList.push_back(s);
  }

  // 2. For each state:
  for (auto *preMutState: preMutStateList) {
    //  a) Compare until a pre-post diff is found 
    //  b) check the feasibility of the pre-post diff with `mState`
    //  c) Return the feasible by adding to `insdiff`
    int sDiff = ExecutionState::KS_StateDiff_t::ksNO_DIFF; 
    ref<Expr> prepathPrefix = ConstantExpr::alloc(1, Expr::Bool);
    for (ConstraintManager::constraint_iterator it = preMutState->constraints.begin(), 
                      ie = preMutState->constraints.end(); it != ie; ++it) {
      prepathPrefix = AndExpr::create(prepathPrefix, *it);
    }
    bool result = ks_diffTwoStates (mState, preMutState, prepathPrefix, 
                                                outEnvOnly, sDiff, inStateDiffExp); 
    if (result) {
      if (!ExecutionState::ks_isNoDiff(sDiff)) {
        if (/*outputTestCases 
              &&*/ (!(semuTestgenOnlyForCriticalDiffs || outEnvOnly) 
                    || ExecutionState::ks_isCriticalDiff(sDiff))) {
            ref<Expr> tmp_insdiff;
            ks_createDiffExpr(mState, tmp_insdiff, prepathPrefix, inStateDiffExp);
            if (ks_checkfeasiblewithsolver(*mState, AndExpr::create(insdiff, tmp_insdiff))) {
              insdiff = AndExpr::create(insdiff, tmp_insdiff);
              can_diff = true;
              break;
            }
        }
      }
    }
  }
  return can_diff;
}
#endif

//return true if there is a strong difference (the mutant is killed, stop it)
//XXX: For a mutant do we need to generate test for all difference with original or only one?
bool Executor::ks_compareRecursive (ExecutionState *mState, 
                                    std::vector<ExecutionState *> &mSisStatesVect,
                                    std::map<ExecutionState *, ref<Expr>> &origSuffConstr, 
                                    bool outEnvOnly, bool postMutOnly, llvm::SmallPtrSet<ExecutionState *, 5> &postMutOnly_hasdiff) {

  static unsigned gentestid = 0;
  static unsigned num_failed_testgen = 0;
  bool outputTestCases = ks_outputTestsCases; //false;
  bool doMaxSat = ks_outputStateDiffs; //true;

  std::vector<ref<Expr>> inStateDiffExp;
  bool diffFound = false;

#ifdef SEMU_RELMUT_PRED_ENABLED
  if (mState->ks_old_new > 0 || postMutOnly) { // only post commit version, except postMut
#endif

  // enter if WP (neither outenv nor post mutation) and the state in not ongoing
  // or is outenv and in its data or is post mutation and is in its data
  if ((!outEnvOnly && !postMutOnly && ks_ongoingExecutionAtWP.count(mState) == 0)
      || (outEnvOnly && ks_reachedOutEnv.count(mState) > 0)
      || (postMutOnly && ks_atPointPostMutation.count(mState) > 0)) {
    for (auto mSisState: mSisStatesVect) {
#ifdef SEMU_RELMUT_PRED_ENABLED
      if (mSisState != nullptr && mSisState->ks_old_new <= 0 && !postMutOnly) // No need for pre-commit original, except postMut
        continue; 
#endif
      int sDiff = ExecutionState::KS_StateDiff_t::ksNO_DIFF; 
      ref<Expr> origpathPrefix = origSuffConstr.at(mSisState);
      bool result = ks_diffTwoStates (mState, mSisState, origpathPrefix, 
                                                outEnvOnly, sDiff, inStateDiffExp); 
      if (result) {
        if (ExecutionState::ks_isNoDiff(sDiff)) {
          if (!outEnvOnly/*at check point*/ && doMaxSat) {
            // XXX put out the paths showing no differences as well
            if (mSisState != nullptr)
              ks_checkMaxSat(mState->constraints, mSisState, inStateDiffExp, mState->ks_mutantID, sDiff);
          }
  #ifdef ENABLE_KLEE_SEMU_DEBUG
          llvm::errs() << "<==> a state pair of Original and Mutant-" << mState->ks_mutantID << " are Equivalent.\n\n";
  #endif
        } else {
  #ifdef ENABLE_KLEE_SEMU_DEBUG
          llvm::errs() << "<!=> a state pair of Original and Mutant-" << mState->ks_mutantID << " are Different.\n\n";
  #endif
          diffFound = true;

          // postMutOnly case
          if (postMutOnly) {
            postMutOnly_hasdiff.insert(mState);
            return diffFound;
          }

          // Consider only critical diffs if outEnvOnly is True (semuTestgenOnlyForCriticalDiffs is overriden)
          if (outputTestCases 
                && (!(semuTestgenOnlyForCriticalDiffs || outEnvOnly) 
                      || ExecutionState::ks_isCriticalDiff(sDiff))) {
            // On test generation mode, if the mutant of mState reached maximum quota of tests, 
            // stop comparing. 
            // insert if not yet in map
            auto irp_m = mutants2gentestsNum.insert(std::make_pair(mState->ks_mutantID, 0));
            if (irp_m.first->second >= semuMaxNumTestGenPerMutant) {
              // We reache the test gen quota for the mutant, do nothing
              return diffFound;
            }

            // generate test case of this difference.
            ref<Expr> insdiff;
            ks_createDiffExpr(mState, insdiff, origpathPrefix, inStateDiffExp);

#ifdef SEMU_RELMUT_PRED_ENABLED
            if (ks_mutantPrePostDiff (mState, outEnvOnly, inStateDiffExp, insdiff)) {
#endif

            // XXX create a new mState just to output testcase and destroy after
            ExecutionState *tmp_mState = new ExecutionState(*mState);
            if (! semuDisableStateDiffInTestGen)
                tmp_mState->addConstraint (insdiff); 
            // TODO FIXME: Handle cases where the processTestCase call fail and no test is gen (for the following code: mutants stats uptades)
            interpreterHandler->processTestCase(*tmp_mState, nullptr, nullptr); //std::to_string(mState->ks_mutantID).insert(0,"Mut").c_str());
            delete tmp_mState;
            ////size_t clen = mState->constraints.size();
            ////mState->addConstraint (insdiff); 
            ////interpreterHandler->processTestCase(*mState, nullptr, nullptr); //std::to_string(mState->ks_mutantID).insert(0,"Mut").c_str());
            bool testgen_was_good = ks_writeMutantTestsInfos(mState->ks_mutantID, ++gentestid); //Write info needed to know which test for which mutant 
            ////if (mState->constraints.size() > clen) {
            ////  assert (mState->constraints.size() == clen + 1 && "Expect only one more contraint here, the just added one");
            ////  mState->constraints.back() = ConstantExpr::alloc(1, Expr::Bool);    //set just added constraint to true
            ////}

            if (testgen_was_good) {
              // increment mutant test count 
              irp_m.first->second += 1;
            } else { 
              ++num_failed_testgen;
            }

            // XXX: If reached maximum number ot tests, exit
            if (gentestid - num_failed_testgen >= semuMaxTotalNumTestGen) {
              llvm::errs() << "\nSEMU@COMPLETED: Maximum number of generated tests reached. Done!\n";
              exit(0);
            }

            // Stop mutant if reached its limit
            if (irp_m.first->second >= semuMaxNumTestGenPerMutant) {
              // We reache the test gen quota for the mutant, do nothing
              return diffFound;
            }
            //return true;
#ifdef SEMU_RELMUT_PRED_ENABLED
            }
#endif
          }
          if (doMaxSat) {
            ks_checkMaxSat(mState->constraints, mSisState, inStateDiffExp, mState->ks_mutantID, sDiff);
          }

        }
      } else {
  #ifdef ENABLE_KLEE_SEMU_DEBUG
        llvm::errs() << "# Infesible differential between an original and a Mutant-" << mState->ks_mutantID <<".\n\n";
  #endif
      }
    }
  }
  
#ifdef SEMU_RELMUT_PRED_ENABLED
  }//~ only post commit version
#endif

  //compare children as well
  for (ExecutionState *es: mState->ks_childrenStates) {
    diffFound |= ks_compareRecursive (es, mSisStatesVect, origSuffConstr, outEnvOnly, postMutOnly, postMutOnly_hasdiff);
  }
  
  return diffFound;
}


// This take the path condition common to a mutant and original, together 
// with the conditions of equality, for each state variable, between
// original and mutant
void Executor::ks_checkMaxSat (ConstraintManager const &mutPathCond,
                                ExecutionState const *origState,
                                std::vector<ref<Expr>> &stateDiffExprs,
                                ExecutionState::KS_MutantIDType mutant_id, int sDiff) {
  ConstraintManager const &origPathCond = origState->constraints;
  unsigned nMaxFeasibleDiffs, nMaxFeasibleEqs, nSoftClauses;

  nSoftClauses = stateDiffExprs.size();
  nMaxFeasibleDiffs = nMaxFeasibleEqs = 0;
  
  // XXX Make this better (considering overlap  in both path conds), usig set?
  std::set<ref<Expr>> hardClauses(origPathCond.begin(), origPathCond.end());
  //llvm::errs() << origPathCond.size() << " oooo##$$$\n";
  //llvm::errs() << mutPathCond.size() << " ##$$$\n";
  //mutPathCond.back()->dump(); (*mutPathCond.begin())->dump();
  hardClauses.insert(mutPathCond.begin(), mutPathCond.end());

  // remove the false from soft clauses: 
  // They are certainly not part of maxsat for nSoftClauses
  // remove the true from soft clauses: 
  // They are certainly not part of maxsat for nMaxFeasibleEqs, but added to ...Diffs
  std::vector<unsigned> posToRemove;
  unsigned nFalse = 0, nTrue = 0;
  for (unsigned i = 0; i < stateDiffExprs.size(); ++i) {
    if (stateDiffExprs[i]->isTrue()) {
      ++nTrue;
      posToRemove.push_back(i);
    } else if (stateDiffExprs[i]->isFalse()) {
      ++nFalse;
      posToRemove.push_back(i);
    }
  }
  // remove
  for (auto it = posToRemove.rbegin(), ie = posToRemove.rend(); it != ie; ++it)
    stateDiffExprs.erase(stateDiffExprs.begin() + (*it));
    
  if (! stateDiffExprs.empty()) {
    //Solver *coreSolver = klee::createCoreSolver(CoreSolverToUse);
    //bool ress;
    //std::vector<ref<Expr>> xxx; xxx.insert(xxx.begin(), hardClauses.begin(), hardClauses.end());
    //bool rr = coreSolver->mayBeTrue(Query(ConstraintManager(xxx), stateDiffExprs.back()), ress); 
    //llvm::errs() << rr << " " << ress<< " #\n";

    nMaxFeasibleDiffs = stateDiffExprs.size();
#ifdef KS_Z3MAXSAT_SOLVER__H
    pmaxsat_solver.setSolverTimeout(coreSolverTimeout);
    // TODO fix z3 maxsat and uncoment bellow
    //pmaxsat_solver.checkMaxSat(hardClauses, stateDiffExprs, nMaxFeasibleDiffs, nMaxFeasibleEqs);
#endif //~KS_Z3MAXSAT_SOLVER__H
  }
  
  // update using nTrue and nFalse
  nMaxFeasibleDiffs += nTrue;
  nSoftClauses -= nFalse;

  ks_writeMutantStateData (mutant_id, nSoftClauses, nMaxFeasibleDiffs, nMaxFeasibleEqs, sDiff, origState);
}

void Executor::ks_writeMutantStateData(ExecutionState::KS_MutantIDType mutant_id,
                                unsigned nSoftClauses,
                                unsigned nMaxFeasibleDiffs,
                                unsigned nMaxFeasibleEqs,
                                int sDiff,
                                ExecutionState const *origState) {
  static const std::string fnPrefix("mutant-");
  static const std::string fnSuffix(".semu");
  static std::map<ExecutionState::KS_MutantIDType, std::string> mutantID2outfile;
  //llvm::errs() << "MutantID | nSoftClauses  nMaxFeasibleDiffs  nMaxFeasibleEqs | Diff Type\n";
  //llvm::errs() << mutant_id << " | " << nSoftClauses << "  " << nMaxFeasibleDiffs << "  " << nMaxFeasibleEqs << " | " << sDiff << "\n";
  std::string header;
  std::string out_file_name = mutantID2outfile[mutant_id];
  if (out_file_name.empty()) {
    mutantID2outfile[mutant_id] = out_file_name = 
           interpreterHandler->getOutputFilename(fnPrefix+std::to_string(mutant_id)+fnSuffix);
    header.assign("ellapsedTime(s),MutantID,nSoftClauses,nMaxFeasibleDiffs,nMaxFeasibleEqs,Diff_Type,OrigState,WatchPointID,MaxDepthID\n");
  }

  unsigned ellapsedtime = util::getWallTime() - ks_runStartTime;
  std::ofstream ofs(out_file_name, std::ofstream::out | std::ofstream::app); 
  if (ofs.is_open()) {
    ofs << header << ellapsedtime<<","<<mutant_id << "," << nSoftClauses << "," << nMaxFeasibleDiffs << "," << nMaxFeasibleEqs 
        << "," << sDiff << "," << origState << "," << ks_checkID << "," << ks_nextDepthID <<"\n";
    ofs.close();
  } else {
    llvm::errs() << "Error: Unable to create info file: " << out_file_name 
                 << ". Mutant ID is:" << mutant_id << ".\n";
    assert(false);
    exit(1);
  }
}

bool Executor::ks_writeMutantTestsInfos(ExecutionState::KS_MutantIDType mutant_id, unsigned testid) {
  static const std::string fnPrefix("mutant-");
  static const std::string fnSuffix(".ktestlist");
  static const std::string ktest_suffix(".ktest");
  static std::map<ExecutionState::KS_MutantIDType, std::string> mutantID2outfile;
  static std::string semu_exec_info_file;
  std::string mutant_data_header;
  std::string info_header;

  // put out the semu ececution info
  if (semu_exec_info_file.empty()) {
    semu_exec_info_file = interpreterHandler->getOutputFilename("semu_execution_info.csv");
    info_header.assign("ellapsedTime(s),stateCompareTime(s),#MutStatesForkedFromOriginal,#MutStatesEqWithOrigAtMutPoint\n");
  } 
  double ellapsedtime = util::getWallTime() - ks_runStartTime;
  std::ofstream ofs(semu_exec_info_file, std::ofstream::out | std::ofstream::app); 
  if (ofs.is_open()) {
    ofs << info_header << ellapsedtime << "," << ks_totalStateComparisonTime << ","
                  << ks_numberOfMutantStatesCheckedAtMutationPoint << ","
                  << ks_numberOfMutantStatesDiscardedAtMutationPoint << "\n";
    ofs.close();
  } else {
    llvm::errs() << "Error: Unable to create/open semu info file: " << semu_exec_info_file << ".\n";
    llvm::errs() << ">>> Failbit: " << ofs.fail() << ", Badbit: " << ofs.bad() << ", Eofbit: " << ofs.eof() << ".\n";
    llvm::errs() << ">>> " << strerror(errno) << "\n";
    assert(false);
    exit(1);
  }


  // Make sure that the test was correctly generated
  std::stringstream filename;
  filename << "test" << std::setfill('0') << std::setw(6) << testid << ktest_suffix;
  // In case the test case generation failed, do not add it
  //if (! llvm::sys::fs::exists(interpreterHandler->getOutputFilename(filename.str()))) {
    //return false;
  //}

  // Test is generated okay, update mutant info file
  std::string out_file_name = mutantID2outfile[mutant_id];
  if (out_file_name.empty()) {
    mutantID2outfile[mutant_id] = out_file_name = 
           interpreterHandler->getOutputFilename(fnPrefix+std::to_string(mutant_id)+fnSuffix);
    mutant_data_header.assign("ellapsedTime(s),MutantID,ktest\n");
  }

  ellapsedtime = util::getWallTime() - ks_runStartTime;
  ofs.open(out_file_name, std::ofstream::out | std::ofstream::app); 
  if (ofs.is_open()) {
    ofs << mutant_data_header << ellapsedtime << "," << mutant_id << "," << filename.str() << "\n";
    ofs.close();
  } else {
    llvm::errs() << "Error: Unable to create test info file: " << out_file_name 
                 << ". Mutant ID is:" << mutant_id << ".\n";
    llvm::errs() << ">>> Failbit: " << ofs.fail() << ", Badbit: " << ofs.bad() << ", Eofbit: " << ofs.eof() << ".\n";
    llvm::errs() << ">>> " << strerror(errno) << "\n";
    assert(false);
    exit(1);
  }

  return true;
}

void Executor::ks_loadKQueryConstraints(std::vector<ConstraintManager> &outConstraintsList) {
  for (auto it=semuPrecondFiles.begin(), ie=semuPrecondFiles.end(); it != ie; ++it) {
    std::string kqFilename = *it;
    std::string ErrorStr;
  
#if LLVM_VERSION_CODE < LLVM_VERSION(3,5)
    OwningPtr<MemoryBuffer> MB;
    error_code ec=MemoryBuffer::getFile(kqFilename.c_str(), MB);
    if (ec) {
      llvm::errs() << "Loading Constraints from File '" << kqFilename << "': error: " << ec.message() << "\n";
      exit(1);
    }
#else
    auto MBResult = MemoryBuffer::getFile(kqFilename.c_str());
    if (!MBResult) {
      llvm::errs() << "Loading Constraints from File '" << kqFilename << "': error: " << MBResult.getError().message()
                   << "\n";
      exit(1);
    }
    std::unique_ptr<MemoryBuffer> &MB = *MBResult;
#endif
  
    ExprBuilder *Builder = createDefaultExprBuilder();
    Builder = createConstantFoldingExprBuilder(Builder);
    Builder = createSimplifyingExprBuilder(Builder);

    std::vector<expr::Decl*> Decls;
    expr::Parser *P = expr::Parser::Create(kqFilename, MB.get(), Builder, /*ClearArrayAfterQuery*/false);
    P->SetMaxErrors(20);
    while (expr::Decl *D = P->ParseTopLevelDecl())
    {
      Decls.push_back(D);
    }

    bool success = true;
    if (unsigned N = P->GetNumErrors())
    {
      llvm::errs() << kqFilename << ": parse failure: "
             << N << " errors.\n";
      success = false;
    }

    if (!success)
      exit(1);

    //Loop over the declarations
    for (std::vector<expr::Decl*>::iterator it = Decls.begin(), ie = Decls.end(); it != ie; ++it)
    {
      expr::Decl *D = *it;
      if (expr::QueryCommand *QC = dyn_cast<expr::QueryCommand>(D))
      {
        ConstraintManager constraintM(QC->Constraints);
        outConstraintsList.push_back(constraintM);
        //for (auto it=constraintM.begin(), ie=constraintM.end(); it != ie; ++it)
        //  outConstraints.addConstraint(*it);
      }
    }
    //Clean up - Cancelled because we will still use it
    /*for (std::vector<expr::Decl*>::iterator it = Decls.begin(),
        ie = Decls.end(); it != ie; ++it)
      delete *it;
    delete P;*/
  }
}

/**
 * In Test Generation (TG) mode, check whether the last instruction lead to fork or branch
 **/
bool Executor::ks_hasJustForkedTG (ExecutionState &state, KInstruction *ki) {
  if (!state.ks_childrenStates.empty()) {
    return true;
  }
  return false;
}

void Executor::ks_fourWayForksTG() {
  // TODO implement this
  // TODO XXX Clear the ks_childrenStates of every state
}


/// Avoid infinite loop mutant to run indefinitely: simple fix
inline void Executor::ks_CheckAndBreakInfinitLoop(ExecutionState &curState, ExecutionState *&prevState, double &initTime) {
  if(curState.ks_mutantID > 0) {   //TODO: how about when we reach watch point
    if (prevState != &curState) {
      prevState = &curState;
      initTime = util::getWallTime();
    } else if (semuLoopBreakDelay < util::getWallTime() - initTime) {
      klee_message((std::string("SEMU@WARNING: Loop Break Delay reached for mutant ")+std::to_string(curState.ks_mutantID)).c_str());
      terminateStateEarly(curState, "infinite loop"); //Terminate mutant
      // XXX Will be removed from searcher and processed bellow
      //continue;
    }
  }
}

/// Return true if the state comparison actually happened, false otherwise. This help to know if we need to call updateStates or not
inline bool Executor::ks_CheckpointingMainCheck(ExecutionState &curState, KInstruction *ki, bool isSeeding, uint64_t precond_offset) {
  // // FIXME: We just checked memory and some states will be killed if memory exeeded and we will have some problem in comparison
  // // FIXME: For now we assume that the memory limit must not be exceeded, need to find a way to handle this later
  if (atMemoryLimit) {
    if (semuEnableNoErrorOnMemoryLimit) {
      static double prevWarnMemLimitTime = 0;
      // Print at most one warning per 3 seconds, to avoid clogging
      if (util::getWallTime() - prevWarnMemLimitTime > 3) {
        klee_warning("SEMU@WARNING: reached memory limit and killed states. You could increase memory limit or restrict symbex.");
        prevWarnMemLimitTime = util::getWallTime();
      }
    } else {
      klee_error("SEMU@ERROR: Must not reach memory limit and kill states. increase memory limit or restrict symbex (FIXME)");
      exit(1);
    }
  }

  static std::map<ExecutionState*, std::vector<SeedInfo> > backed_seedMap;

  bool ks_terminated = false;
  bool ks_OutEnvReached = false;
  bool ks_WPReached = false;
  bool ks_atPostMutation = false;
  bool ks_atNextCheck = false;

  // If next instruction is unreachable instruction, terminate the state early
  //if (/*llvm::isa<llvm::UnreachableInst>(curState.pc->inst) || */llvm::isa<llvm::UnreachableInst>(curState.prevPC->inst)) {
  //  terminateStateEarly(curState, "@SEMU: unreachable instruction reached");
  //  ks_terminated = true;
  //} else {
    ks_terminated = !ks_justTerminatedStates.empty();
    // XXX Do not check whe the instruction is PHI or EHPad, except the state terminated
    if (ks_terminated || !(llvm::isa<llvm::PHINode>(ki->inst) /*|| ki->inst->isEHPad()*/)) {
      // A state can have both in ks_atPostMutation and ks_atNextCheck true
      ks_OutEnvReached = ks_nextIsOutEnv (curState);
      ks_WPReached = ks_watchPointReached (curState, ki);
      if (ks_WPReached) {
        if (curState.ks_mutantID == 0) {
          // Original (ks_mutantID == 0) do not have WP
          // instead has next check
          // Note that bot ks_atPostMutation and ks_atNextCheck ar false here
          std::swap(ks_WPReached, ks_atNextCheck);
        }
      } else {
	      ks_atPostMutation = ks_checkAtPostMutationPoint(curState, ki);
        ks_atNextCheck = !ks_atPostMutation && ks_reachedCheckNextDepth(curState);
      }
    }
  //}

  if (ks_terminated | ks_WPReached | ks_OutEnvReached | ks_atPostMutation | ks_atNextCheck) {   //(ks_terminatedBeforeWP.count(&curState) == 1)

    bool curTerminated = ks_justTerminatedStates.count(&curState) > 0;
    
    //remove from searcher or seed map if the curState is concerned (guaranteed for WP and Out Env, but not term: termination while forking)
    if (ks_WPReached | ks_OutEnvReached | curTerminated | ks_atPostMutation | ks_atNextCheck) {
      if (isSeeding) {
        std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
          seedMap.find(&curState);
        if (it != seedMap.end()) {
          // keep state for restoration after compare (seedMap not set un UpdateStates)
          backed_seedMap[&curState] = it->second;  
          seedMap.erase(it);
        }
      } else {
        searcher->update(&curState, std::vector<ExecutionState *>(), std::vector<ExecutionState *>({&curState}));
      }

      // Terminated has priority, then outEnv
      if (! curTerminated) {
        // add to ks_reachedWatchPoint if so
        if (ks_OutEnvReached) {
          ks_reachedOutEnv.insert(&curState);
          curState.ks_numberOfOutEnvSeen++;
        } else if (ks_atPostMutation) {
          ks_atPointPostMutation.insert(&curState);
        } else if (ks_WPReached) { // ! ks_OutEnvReached and !ks_atPointPostMutation and ks_WPReached
          ks_reachedWatchPoint.insert(&curState);
        } else { // ks_atNextCheck or is original
          ks_ongoingExecutionAtWP.insert(&curState);
        }
      } 
    }

    // keep in ks_justTerminatedStates only the states that are in addedStates, to be removed
    // from searcher and seedMap after the following updateStates 
    static std::vector<ExecutionState *> addedNdeleted;
    addedNdeleted.clear();
    if (!ks_justTerminatedStates.empty()) {
      //std::set_intersection(addedStates.begin(), addedStates.end(), ks_justTerminatedStates.begin(), ks_justTerminatedStates.end(), addedNdeleted.begin());
      for (auto *s: addedStates)
        if (ks_justTerminatedStates.count(s))
          addedNdeleted.push_back(s);

      // Update all terminated 
      ks_terminatedBeforeWP.insert(ks_justTerminatedStates.begin(), ks_justTerminatedStates.end());
      if (atMemoryLimit) {
        // remove the state interrupted, to terminate, in checkMemoryUsage from other sets
        for (auto *_es: ks_justTerminatedStates) {
          ks_moveIntoTerminatedBeforeWP(_es);
          if (_es == &curState)
            continue;
          addedNdeleted.push_back(_es); // FIXME: Not really addedNdeleted. Use another similar vector
        }
      }
      ks_justTerminatedStates.clear();
    }

    // Put this here to make sure that any newly added state are considered 
    // (addesStates and removedStates are empty after)
    updateStates(0);

    // Remove addedNdeleted form both searcher and seedMap
    if (! addedNdeleted.empty()) {
      if (isSeeding) {
        for (auto *s: addedNdeleted) {
          std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
            seedMap.find(s);
          if (it != seedMap.end()) {
            seedMap.erase(it);
          }
        }
      } else {
        searcher->update(nullptr, std::vector<ExecutionState *>(), addedNdeleted);
      }
      addedNdeleted.clear();
    }

    bool recheck_for_no_postmutation_remaining = false;

    do {
      // disable rexecution of this loop by default
      recheck_for_no_postmutation_remaining = false;

      //Check if all reached a stop point and compare states
      if (precond_offset + 
          ks_reachedOutEnv.size() + 
          ks_reachedWatchPoint.size() + 
          ks_terminatedBeforeWP.size() +
          ks_atPointPostMutation.size() +
          ks_ongoingExecutionAtWP.size() >= states.size()) {

        // Make sure that on test gen mode, the mutants that reached maximum
        // generated tests are removed
        /*if (ks_outputTestsCases && !ks_hasOutEnv) {
          ks_eliminateMutantStatesWithMaxTests(true);
        }*/

        std::vector<ExecutionState *> remainWPStates;
        ks_checkID++;
        bool ks_hasOutEnv = !ks_reachedOutEnv.empty();
        bool ks_isAtPostMut = (!ks_hasOutEnv && !ks_atPointPostMutation.empty());

        // if all mutants states reached a branching point
        // apply mutant search
        if (!(ks_hasOutEnv || ks_isAtPostMut)) {
          ks_applyMutantSearchStrategy();
        }

        if (ks_reachedOutEnv.empty() && ks_reachedWatchPoint.empty()
            && ks_terminatedBeforeWP.empty() && ks_atPointPostMutation.empty()) {
          // all states are ongoing, no need to compare, just put them back
          // XXX Will be added into added states bellow
        } else {
	  if(!semuQuiet)
            llvm::errs() << "# SEMU@Status: Comparing states: " << states.size() 
                        << " States" << (ks_hasOutEnv?" (OutEnv)":
                            (ks_isAtPostMut?" (PostMutationPoint)":" (Checkpoint)"))
                        << ".\n";
          auto elapsInittime = util::getWallTime();
          ks_compareStates(remainWPStates, ks_hasOutEnv/*outEnvOnly*/, ks_isAtPostMut/*postMutOnly*/);
          if(!semuQuiet)
	    llvm::errs() << "# SEMU@Status: State Comparison Done! (" << (util::getWallTime() - elapsInittime) << " seconds)\n";
          ks_totalStateComparisonTime += (util::getWallTime() - elapsInittime);
        }
        
        //continue the execution
        if (ks_isAtPostMut) {
          // print stats
          if(!semuQuiet)
	    llvm::errs() << "# SEMU@Status: Aggregated post mutation Discarded/created mutants states: "
                        << ks_numberOfMutantStatesDiscardedAtMutationPoint << "/"
                        << ks_numberOfMutantStatesCheckedAtMutationPoint << "\n";
          if (remainWPStates.empty()) {
            // XXX Make sure there is something in addedStates
            if (addedStates.empty())
              recheck_for_no_postmutation_remaining = true;
            /*unsigned numOriginals = 0;
            for (auto *s: ks_atPointPostMutation)
              if (s->ks_mutantID == 0)
                ++numOriginals;
            llvm::errs() << "\n>> "
                  << "(BUG) No remaining after post Mutation point check. "
                  << "Number of original states in ks_atPointPostMutation is: "
                  << numOriginals
                  << "\n\n";
            assert (false && 
                    "There must be remaining after post mutation point check");*/
          } else {
            // XXX Make sure there is something in addedStates.
            // keep the first to add in the worse case
            ExecutionState * r_backup = remainWPStates.front();
            // disable ks_hasToReachPostMutationPoint
            r_backup->ks_hasToReachPostMutationPoint = false;
            for (auto it=remainWPStates.begin()+1, ie=remainWPStates.end();
                      it != ie; ++it) {
              // disable ks_hasToReachPostMutationPoint
              (*it)->ks_hasToReachPostMutationPoint = false;
              // The state that at the same time reached postmutation 
              // and next check are added into ongoing
              if (ks_reachedCheckNextDepth(*(*it)))
                ks_ongoingExecutionAtWP.insert(*it);
              else
                addedStates.push_back(*it);
            }
            if (addedStates.empty()) {
              addedStates.push_back(r_backup);
            } else {
              if (ks_reachedCheckNextDepth(*r_backup))
                ks_ongoingExecutionAtWP.insert(r_backup);
              else
                addedStates.push_back(r_backup);
            }
          }
        } else {
          addedStates.insert(addedStates.end(), remainWPStates.begin(), remainWPStates.end());
        }
          
        // If outenv is empty, it means that every state reached checkpoint,
        // We can then termintate crash states and clear the term and WP sets
        if (!ks_hasOutEnv) { 
          if (!ks_isAtPostMut) {
            for (SmallPtrSet<ExecutionState *, 5>::iterator it = ks_terminatedBeforeWP.begin(), 
                  ie = ks_terminatedBeforeWP.end(); it != ie; ++it ) {
              //(*it)->pc = (*it)->prevPC;
              terminateState (**it);
            }
            // Clear
            ks_terminatedBeforeWP.clear();
            ks_nextDepthID++; 

            // Clear
            ks_reachedWatchPoint.clear();

            // add back ongoing states
            addedStates.insert(addedStates.end(), ks_ongoingExecutionAtWP.begin(), ks_ongoingExecutionAtWP.end());
            // Clear
            ks_ongoingExecutionAtWP.clear();
          } else {
            // Clear
            ks_atPointPostMutation.clear();
          }

	        if(!semuQuiet)
            llvm::errs() << "# SEMU@Status: After nextdepth point ID=" << (ks_nextDepthID-1) 
                        << " There are " << addedStates.size() 
                        <<" States remaining (seeding is "
                        <<(isSeeding?"True":"False")<<")!\n";

        } else { // there should be no terminated state
          if (ks_reachedOutEnv.size() != remainWPStates.size()) {
            klee_error("SEMU@ERROR: BUG, states reaching outenv different after compare states");
            exit(1);
          }
          // Clear
          ks_reachedOutEnv.clear();
        }

        // Make sure that on test gen mode, the mutants that reached maximum
        // generated tests are removed. 
        // For now, only do that when not outev checking, because if causes a segfault when 
        // there is a single mutant (issue: https://github.com/thierry-tct/KLEE-SEMu/issues/1)
        if (ks_outputTestsCases && !ks_hasOutEnv) {
          ks_eliminateMutantStatesWithMaxTests(false);
        }

        // add all terminated states to the searcher so that update won't assert that the states are not in searcher
        // XXX The searcher is empty here. This is necessary because in updateStates, removedStates must be in searcher
        if (searcher)
          searcher->update(0, removedStates/*adding*/, std::vector<ExecutionState *>());

        // in seeding mode, since seedMap is not augmented in updateState,
        // we update it with remaining states (in addesStates vector) before updateStates
        if (isSeeding) {
          assert ((/*ks_hasOutEnv || */seedMap.empty()) && "SeedMap must be empty at checks"/*point"*/);
          for(auto *s: addedStates) {
            std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
              backed_seedMap.find(s);
            assert (it != backed_seedMap.end() && "A state is not in backed seed map but remains."); 
            seedMap[s] = it->second;
            backed_seedMap.erase(it);
          }
          // if checkpoint, clear backed_seedMap.
          if (!ks_hasOutEnv && !ks_isAtPostMut)
            backed_seedMap.clear();
        } else {
          backed_seedMap.clear(); // seed mode already passed, clear any ramining
        }

        // take account of addedStates and removedStates
        updateStates(0);
      } else {
        // in Seen Mode, the seedMap must not be empty
        if (isSeeding && seedMap.empty()) {
          llvm::errs() << "\n>> (BUG) States size: "<<states.size()<<". Sum of check stages sizes: " 
                       << (ks_reachedOutEnv.size() + ks_reachedWatchPoint.size() 
                                              + ks_terminatedBeforeWP.size() 
                                              + ks_atPointPostMutation.size()
                                              + ks_ongoingExecutionAtWP.size() 
                                              + precond_offset)
                        <<".\n";
          klee_error("SEMU@ERROR: on seeding phase, the seedMap is empty while some some states are not at a check stage of preconditioned");
          exit(1);
        }
      }
    } while(recheck_for_no_postmutation_remaining);

    return true;
  }
  return false;
}


void Executor::ks_heuristicbasedContinueStates (std::vector<ExecutionState*> const &statelist,
                            std::vector<ExecutionState*> &toContinue,
                            std::vector<ExecutionState*> &toStop) {
  if (statelist.empty())
    return;

  // determine the number of mutants to continue
  assert(semuPostCheckpointMutantStateContinueProba >= 0.0 
        && semuPostCheckpointMutantStateContinueProba <= 1.0 
        && "Error, invalid semuPostCheckpointMutantStateContinueProba. Must be between 0.0 adn 1.0");
  unsigned keep = std::round(statelist.size() * semuPostCheckpointMutantStateContinueProba);
  toContinue.clear();
  toStop.clear();
  toContinue.assign(statelist.begin(), statelist.end());
  //std::srand ( unsigned ( std::time(0) ) );
  // Random is enabled when no other is enabled
  std::random_shuffle (toContinue.begin(), toContinue.end());
  //std::srand(1); //revert
  if (semuApplyMinDistToOutputForMutContinue)
    std::sort(toContinue.begin(), toContinue.end(), ks_getMinDistToOutput);
  toStop.assign(toContinue.begin()+keep, toContinue.end());
  toContinue.resize(keep);
}

void Executor::ks_process_closestout_recursive(llvm::CallGraphNode *cgnode,
                  std::map<llvm::CallGraphNode*, unsigned> &visited_cgnodes) {
  static const unsigned MaxLens = 1000000000;
  bool useInstDist = !semuUseBBForDistance;

  if (visited_cgnodes.count(cgnode))
    return;

  llvm::Function * func = cgnode->getFunction();

  // visit
  if (!func || func->isDeclaration()) {
    visited_cgnodes[cgnode] = 0; 
    return;
  }
  visited_cgnodes[cgnode] = MaxLens;

  std::map<llvm::Function*, llvm::CallGraphNode*> func2cgnode;
  // Compute every dependency
  for (auto cg_it = cgnode->begin(); cg_it != cgnode->end(); ++cg_it) {
    ks_process_closestout_recursive(cg_it->second, visited_cgnodes);
    func2cgnode[cg_it->second->getFunction()] = cg_it->second;
  }

  // Compute for this one using the dependencies
  std::vector<llvm::Instruction *> outInsts;

  for (auto &BB: *func) {
    for (auto &Inst: BB) {
      ks_instruction2closestout_distance[&Inst] = MaxLens;
      if (ks_isOutEnvCallInvoke (&Inst)) {
        outInsts.push_back(&Inst);
        ks_instruction2closestout_distance[&Inst] = 0;
      } else if (auto *ret = llvm::dyn_cast<llvm::ReturnInst>(&Inst)) {
        outInsts.push_back(&Inst);
        ks_instruction2closestout_distance[&Inst] = 0;
      } else if (auto *unreach = llvm::dyn_cast<llvm::UnreachableInst>(&Inst)) {
        outInsts.push_back(&Inst);
        ks_instruction2closestout_distance[&Inst] = 0;
      }
    }
  }

  assert (outInsts.size() > 0 && "function having no outenv and no ret? (BUG)");

  // start from outInst and go backward
  std::queue<llvm::Instruction *> workQ;
  for (auto *inst: outInsts)
    workQ.push(inst);

  while (!workQ.empty()) { // BFS
    auto *inst = workQ.front();
    workQ.pop();
    unsigned inc = 0;
    unsigned this_i_dist;
    llvm::Instruction *bb_1st_inst = &(inst->getParent()->front());
    if (inst != bb_1st_inst) {
      for (llvm::Instruction *I = inst->getPrevNode(); I; I = ((I==bb_1st_inst)?0:I->getPrevNode())) {
        //if (!llvm::isa<llvm::DbgInfoIntrinsic>(I))
        if (useInstDist) {
          inc++;
        }
        llvm::Function *tmpF = nullptr;
        if (auto * ci = llvm::dyn_cast<llvm::CallInst>(I)) {
          tmpF = ci->getCalledFunction();
        } else if (auto * ii = llvm::dyn_cast<llvm::InvokeInst>(I)) {
          tmpF = ii->getCalledFunction();
        }
        if (tmpF) {
          inc += visited_cgnodes[func2cgnode[tmpF]];
        }
        this_i_dist = std::min(ks_instruction2closestout_distance[inst] + inc, MaxLens);
        if (this_i_dist < ks_instruction2closestout_distance[I])
          ks_instruction2closestout_distance[I] = this_i_dist;
      }
    }

    // update values of previouus basic block terminators
    llvm::BasicBlock *bb = inst->getParent();
    inc++;
    for (auto pred=llvm::pred_begin(bb); pred != llvm::pred_end(bb); ++pred) {
      llvm::Instruction *term = (*pred)->getTerminator();
      if (!term)
        term = &((*pred)->back());
      if (!term)
        assert (false && "block without instruction??");
      llvm::Function *tmpF = nullptr;
      if (auto * ci = llvm::dyn_cast<llvm::CallInst>(term)) {
        tmpF = ci->getCalledFunction();
      } else if (auto * ii = llvm::dyn_cast<llvm::InvokeInst>(term)) {
        tmpF = ii->getCalledFunction();
      }
      if (tmpF) {
        inc += visited_cgnodes[func2cgnode[tmpF]];
      }
      this_i_dist = std::min(ks_instruction2closestout_distance[inst] + inc, MaxLens);
      if (this_i_dist < ks_instruction2closestout_distance[term]) {
        ks_instruction2closestout_distance[term] = this_i_dist;
        workQ.push(term);
      }
    }
    if (bb == &(func->getEntryBlock())) {
      visited_cgnodes[cgnode] = std::min(visited_cgnodes[cgnode], ks_instruction2closestout_distance[inst] + inc - 1);
    } 
  }
}

void Executor::ks_initialize_ks_instruction2closestout_distance(llvm::Module* mod) {
  ks_instruction2closestout_distance.clear();
  if (semuApplyMinDistToOutputForMutContinue) {
    // Compute Call graph
#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 5)
    llvm::CallGraph call_graph(*mod);
#else
    llvm::CallGraph call_graph;
    call_graph.runOnModule(*mod);
#endif

    std::map<llvm::CallGraphNode*, unsigned> cgnode_to_midist_out_ret;
    do {
      auto *root = call_graph.getRoot();
      ks_process_closestout_recursive(root, cgnode_to_midist_out_ret);
    } while (false);
  }
}

std::map<llvm::Instruction*, unsigned> Executor::ks_instruction2closestout_distance;

bool Executor::ks_getMinDistToOutput(ExecutionState *lh, ExecutionState *rh) {
  // We safely use pc not prevPC because 
  // the states here must be executing (not terminated)
  // and pc is actually what will be executed next
  llvm::Instruction *lhInst = lh->pc->inst;
  llvm::Instruction *rhInst = rh->pc->inst;
  return ks_instruction2closestout_distance[lhInst] < ks_instruction2closestout_distance[rhInst];
}

void Executor::ks_applyMutantSearchStrategy() {
  // whether to apply the search also for the original (discard some states)
  // XXX Keep this to False. if set to true, there is need to update the
  // Original children state tree to have no problem on state comparison
  const bool original_too = false;
  // Whether to apply for a mutant ID regardless of whether 
  //ganerated from same original state
  const bool mutant_alltogether = true;
  std::map<ExecutionState::KS_MutantIDType, std::vector<ExecutionState*>> 
                                                    mutId2states;
  for (auto *s: ks_reachedWatchPoint) {
    if (original_too || s->ks_mutantID != 0)
      mutId2states[s->ks_mutantID].push_back(s);
  }

  // Update seenCheckpoint
  for (auto &p: mutId2states)
    for (auto *s: p.second)
      s->ks_numSeenCheckpoints++;
  
  // add ongoing too if enabled
  if (mutant_alltogether) {
    for (auto *s: ks_ongoingExecutionAtWP) {
      // TODO Optimize the first condition with iterator
      if (mutId2states.count(s->ks_mutantID) > 0 && 
              (original_too || s->ks_mutantID != 0)) {
        mutId2states[s->ks_mutantID].push_back(s);
      }
    }
  }

  std::vector<ExecutionState*> toContinue, toStop;
  llvm::SmallPtrSet<ExecutionState *, 5> toTerminate;
  for (auto p: mutId2states) {
    toContinue.clear();
    toStop.clear();
    // XXX Choose strategy here
    ks_heuristicbasedContinueStates(p.second, toContinue, toStop);

    // Handle continue and stop
    for (auto *s: toContinue) {
      ks_ongoingExecutionAtWP.insert(s);
      if (ks_reachedWatchPoint.count(s)) {
        s->ks_startdepth = s->depth;
        ks_reachedWatchPoint.erase(s);
      }
    }
    for (auto *s: toStop) {
      if (ks_reachedWatchPoint.count(s)) {
        if (!toContinue.empty() && // make sure that not everything is removed
                s->ks_numSeenCheckpoints <= semuGenTestForDiscardedFromCheckNum) {
          ks_reachedWatchPoint.erase(s);
          toTerminate.insert(s);
        }
      }
      if (ks_ongoingExecutionAtWP.count(s)) {
        ks_ongoingExecutionAtWP.erase(s);
        toTerminate.insert(s);
      }
    }
  }

  std::vector<ExecutionState *> parStates;
  if (! toTerminate.empty())
    ks_getMutParentStates(parStates);

  // Terminate the muts in toTerminate
  for (auto *es: parStates) {
    ks_fixTerminatedChildren(es, toTerminate, true);
  }
  for (auto *es: toTerminate) {
    //es->pc = es->prevPC;
    terminateState(*es);
  }
}

void Executor::ks_eliminateMutantStatesWithMaxTests(bool pre_compare) {
  std::set<ExecutionState::KS_MutantIDType> reached_max_tg;
  for (auto mp: mutants2gentestsNum)
    if (mp.second >= semuMaxNumTestGenPerMutant) {
      assert (mp.first > 0 && "Original must not be found here (BUG)");
      reached_max_tg.insert(mp.first);
    }

  if (reached_max_tg.size() > 0) {
    llvm::SmallPtrSet<ExecutionState *, 5> toTerminate;
    if (pre_compare) {
      llvm::SmallPtrSet<ExecutionState *, 5> tmp_one_toTerminate;
      tmp_one_toTerminate.clear();
      for (auto m_it=ks_reachedOutEnv.begin(); 
                                      m_it != ks_reachedOutEnv.end();++m_it) {
        if (reached_max_tg.count((*m_it)->ks_mutantID) > 0) {
          tmp_one_toTerminate.insert(*m_it);
        } 
      }
      for (auto *s: tmp_one_toTerminate) {
        ks_reachedOutEnv.erase(s);
        toTerminate.insert(s);
      }

      tmp_one_toTerminate.clear();
      for (auto m_it=ks_reachedWatchPoint.begin(); 
                                      m_it != ks_reachedWatchPoint.end();++m_it) {
        if (reached_max_tg.count((*m_it)->ks_mutantID) > 0) {
          tmp_one_toTerminate.insert(*m_it);
        } 
      }
      for (auto *s: tmp_one_toTerminate) {
        ks_reachedWatchPoint.erase(s);
        toTerminate.insert(s);
      }

      tmp_one_toTerminate.clear();
      for (auto m_it=ks_terminatedBeforeWP.begin(); 
                                      m_it != ks_terminatedBeforeWP.end();++m_it) {
        if (reached_max_tg.count((*m_it)->ks_mutantID) > 0) {
          tmp_one_toTerminate.insert(*m_it);
        } 
      }
      for (auto *s: tmp_one_toTerminate) {
        ks_terminatedBeforeWP.erase(s);
        toTerminate.insert(s);
      }

      tmp_one_toTerminate.clear();
      for (auto m_it=ks_atPointPostMutation.begin(); 
                                      m_it != ks_atPointPostMutation.end();++m_it) {
        if (reached_max_tg.count((*m_it)->ks_mutantID) > 0) {
          tmp_one_toTerminate.insert(*m_it);
        }
      }
      for (auto *s: tmp_one_toTerminate) {
        ks_atPointPostMutation.erase(s);
        toTerminate.insert(s);
      }

      tmp_one_toTerminate.clear();
      for (auto m_it=ks_ongoingExecutionAtWP.begin(); 
                                      m_it != ks_ongoingExecutionAtWP.end();++m_it) {
        if (reached_max_tg.count((*m_it)->ks_mutantID) > 0) {
          tmp_one_toTerminate.insert(*m_it);
        }
      }
      for (auto *s: tmp_one_toTerminate) {
        ks_ongoingExecutionAtWP.erase(s);
        toTerminate.insert(s);
      }
    } else {
      llvm::SmallPtrSet<ExecutionState *, 5> tmp_one_toTerminate;
      for (auto m_it=addedStates.begin(); 
                                      m_it != addedStates.end();++m_it) {
        if (reached_max_tg.count((*m_it)->ks_mutantID) > 0) {
          tmp_one_toTerminate.insert(*m_it);
        }
      }
      for (auto *s: tmp_one_toTerminate) {
        std::vector<ExecutionState *>::iterator it =
              std::find(addedStates.begin(), addedStates.end(), s);
        addedStates.erase(it);
        toTerminate.insert(s);
      }
    }

    std::vector<ExecutionState *> parStates;
    if (! toTerminate.empty()) {
      ks_getMutParentStates(parStates);

      // Terminate the discarded
      for (auto *es: parStates) {
        ks_fixTerminatedChildren(es, toTerminate, true, !pre_compare);
      }
      for (auto *es: toTerminate) {
        // FIXME: memory corruption the 
        //es->pc = es->prevPC;
        terminateState(*es);
        //ks_terminatedBeforeWP.insert(es);
      }
    }
  }
}

#ifdef SEMU_RELMUT_PRED_ENABLED
// get a new ID for a version split node
inline unsigned long long Executor::ks_get_next_oldnew_split_id() {
  return ks_current_oldnew_split_id++;
}

void Executor::ks_odlNewPrepareModule (llvm::Module *mod) {
  // - Set isold global
  if (mod->getNamedGlobal(ks_isOldVersionName)) {
    llvm::errs() << "The gobal variable '" << ks_isOldVersionName
                 << "' already present in code!\n";
    assert(false &&  "ERROR: Module already mutated!");
    exit(1);
  }
  mod->getOrInsertGlobal(ks_isOldVersionName,
                           llvm::Type::getInt1Ty(getGlobalContext()));
  ks_isOldVersionGlobal = mod->getNamedGlobal(ks_isOldVersionName);
  //ks_isOldVersionGlobal->setAlignment(4);
  ks_isOldVersionGlobal->setInitializer(llvm::ConstantInt::get(
                            getGlobalContext(), llvm::APInt(1, 0, false)));

  // - check and prepare klee_change
  ks_klee_change_function = mod->getFunction(ks_klee_change_funtion_Name);
  if (!ks_klee_change_function || ks_klee_change_function->arg_size() != 2) {
    llvm::errs() << 
            "ERROR: klee_change missing in relevant mutant prediction mode\n";
    assert (false);
    exit(1);
  }
  // change the body of klee_change
  ks_klee_change_function->deleteBody();
  llvm::BasicBlock *block = llvm::BasicBlock::Create(
                        getGlobalContext(), "entry", ks_klee_change_function);
  llvm::Function::arg_iterator args = ks_klee_change_function->arg_begin();
  llvm::Value *old_v = llvm::dyn_cast<llvm::Value>(args++);
  llvm::Value *new_v = llvm::dyn_cast<llvm::Value>(args++);
	
  //llvm::IRBuilder<> builder(block);
  // load ks_isOldVersionGlobal
  //llvm::Value *isold = builder.CreateLoad(ks_isOldVersionGlobal);
  llvm::LoadInst *isold = new LoadInst(ks_isOldVersionGlobal);
  block->getInstList().push_back(isold);
  // create select
  //llvm::Value *selection = builder.CreateSelect(isold, old_v, new_v);
  llvm::SelectInst *selection = SelectInst::Create(isold, old_v, new_v);
  block->getInstList().push_back(selection);
  ////builder.CreateBinOp(llvm::Instruction::Mul, x, y, "tmp");
  //builder.CreateRet(selection);
  block->getInstList().push_back(ReturnInst::Create(getGlobalContext(), selection));
}

void Executor::ks_oldNewBranching(ExecutionState &state) {
  assert (state.ks_old_new == 0);
  TimerStatIncrementer timer(stats::forkTime);

  state.ks_startdepth = state.depth;

  if (MaxForks!=~0u && stats::forks >= MaxForks) {
    
    llvm::errs() << "!! OldNew Not Processed due to MaxForks.\n";
    return;
    
    /*unsigned next = theRNG.getInt32() % N;
    for (unsigned i=0; i<N; ++i) {
      if (i == next) {
        result.push_back(&state);
      } else {
        result.push_back(NULL);
      }
    }*/
  } else {
    stats::forks += 1;
    //addedStates.push_back(&state);

    // In the test generation mode, do not even generate more mutants 
    // If reached max number of tests
    if (ks_outputTestsCases) {
      auto mapit = mutants2gentestsNum.find(state.ks_mutantID);
      if (mapit != mutants2gentestsNum.end() 
                  && mapit->second >= semuMaxNumTestGenPerMutant) {
        // Stop
        // remove from treenode
        state.ks_curBranchTreeNode->exState = nullptr;
        // early terminate
        terminateStateEarly(state, "Original has no possible mutant left");
        return;
      }
    }
    
    // get old_new_split_id
    state.ks_oldnew_split_id = ks_get_next_oldnew_split_id();

    ExecutionState *ns = state.branch();
    addedStates.push_back(ns);
    //result.push_back(ns);
    state.ptreeNode->data = 0;
    std::pair<PTree::Node*,PTree::Node*> res = processTree->split(state.ptreeNode, ns, &state);
    ns->ptreeNode = res.first;
    state.ptreeNode = res.second;
    
    executeMemoryOperation (*ns, true, evalConstant(ks_isOldVersionGlobal), ConstantExpr::create(1, 1), 0);    // isOld is a boolean (1 bit int). 1 for True
    executeMemoryOperation (state, true, evalConstant(ks_isOldVersionGlobal), ConstantExpr::create(0, 1), 0);    // isOld is a boolean (1 bit int). 0 for False
    ns->ks_old_new = -1;
    state.ks_old_new = 1;
    
    // Handle seed phase. Insert old in seedMap with same seed as new
    std::map< ExecutionState*, std::vector<SeedInfo> >::iterator sm_it = 
      seedMap.find(&state);
    if (sm_it != seedMap.end()) {
      seedMap[ns] = sm_it->second;
    }
  }
}
#endif

/**/
// This function is called when running Under Const
bool Executor::ks_lazyInitialize (ExecutionState &state, KInstruction *ki) {
  return true;
}

//~KS

/*****************************************************************/
/*~~~~~~~~~~~~~~~~ SEMu Only Methods @ END ~~~~~~~~~~~~~~~~~~~~~~*/
/*****************************************************************/

///

Interpreter *Interpreter::create(const InterpreterOptions &opts,
                           InterpreterHandler *ih) {
  return new Executor(opts, ih);
}
