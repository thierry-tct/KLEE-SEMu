
#include "ExecutionStateHelperSemu.h"
#include "../ExecutionState.h"

#include "ExecutorHelperSemu.h"
#include "../Executor.h"
#include "../CoreStats.h"
#include "../PTree.h"
#include "../TimingSolver.h"
#include "../Searcher.h"
#include "../SeedInfo.h"
#include "../GetElementPtrTypeIterator.h"


#include "klee/Expr/Parser/Lexer.h"
#include "klee/Expr/Parser/Parser.h"
#include "klee/Expr/ExprBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/IR/CFG.h"

#include "klee/Expr/Expr.h"
#include "klee/Core/Interpreter.h"
#include "klee/Statistics/TimerStatIncrementer.h"
#include "klee/Solver/Common.h"
#include "klee/Expr/Assignment.h"
#include "klee/Expr/ExprPPrinter.h"
#include "klee/Expr/ExprSMTLIBPrinter.h"
#include "klee/Expr/ExprUtil.h"
#include "klee/Config/Version.h"
#include "klee/ADT/KTest.h"
#include "klee/ADT/RNG.h"
#include "klee/Module/Cell.h"
#include "klee/Module/InstructionInfoTable.h"
#include "klee/Module/KInstruction.h"
#include "klee/Module/KModule.h"
#include "klee/Support/ErrorHandling.h"
#include "klee/Support/FloatEvaluation.h"
#include "klee/System/Time.h"
#include "klee/System/MemoryUsage.h"
#include "klee/Solver/SolverStats.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#if LLVM_VERSION_CODE < LLVM_VERSION(8, 0)
#include "llvm/IR/CallSite.h"
#endif
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#if LLVM_VERSION_CODE >= LLVM_VERSION(10, 0)
#include "llvm/Support/TypeSize.h"
#else
typedef unsigned TypeSize;
#endif
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <queue>
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

ExecutorHelperSemu::ExecutorHelperSemu(Executor *parent, unsigned maxForksArg) {
  parentE = parent;
  cfgSemuPreconditionLength = semuPreconditionLength;
  cfgSemuUseOnlyBranchForDepth = semuUseOnlyBranchForDepth;
  cfgSemuForkProcessForSegvExternalCalls = semuForkProcessForSegvExternalCalls;
  MaxForks = maxForksArg;
}

void ExecutorHelperSemu::ks_initializeModule(llvm::Module *module, const Interpreter::ModuleOptions &opts) {
  // @KLEE-SEMu
  ExecutionStateHelperSemu::ks_setMode(semuShadowTestGeneration ? ExecutionStateHelperSemu::KS_Mode::TESTGEN_MODE: ExecutionStateHelperSemu::KS_Mode::SEMU_MODE);
  ks_outputTestsCases = (semuMaxNumTestGenPerMutant > 0 || semuMaxTotalNumTestGen > 0);
  ks_outputStateDiffs = !ks_outputTestsCases;

  // Handle case where only one of semuMaxTotalNumTestGen and semuMaxNumTestGenPerMutant is > 0
  if (ks_outputTestsCases) {
    if (semuMaxTotalNumTestGen == 0)
      semuMaxTotalNumTestGen = (unsigned) -1;  // Very high value
    else // semuMaxNumTestGenPerMutant == 0
      semuMaxNumTestGenPerMutant = (unsigned) -1; // Very high value
  }

  ks_entryFunction = module->getFunction(opts.EntryPoint);
  if (semuSetEntryFuncArgsSymbolic) {
    ks_setInitialSymbolics (*module, *ks_entryFunction);   
  }

  ks_verifyModule(module);

  // Add custom outenv
  for (auto it=semuCustomOutEnvFunction.begin(), ie=semuCustomOutEnvFunction.end(); it != ie; ++it)
    ks_customOutEnvFuncNames.insert(*it);
  
  ks_FilterMutants(module);
  ks_initialize_ks_instruction2closestout_distance(module);
#ifdef SEMU_RELMUT_PRED_ENABLED
  ks_oldNewPrepareModule(module);
#endif
  //~KS
}

void ExecutorHelperSemu::ks_verifyModule(llvm::Module *module) {
  // @KLEE-SEMu
  // Reverify since module was changed
  ks_mutantIDSelectorGlobal = module->getNamedGlobal(ks_mutantIDSelectorName);
  if (!ks_mutantIDSelectorGlobal) {
    assert (false && "@KLEE-SEMu - ERROR: The module is unmutated(no mutant ID selector global var)");
    klee_error ("@KLEE-SEMu - ERROR: The module is unmutated(no mutant ID selector global var)");
  }
  //Make sure that the value of the mutIDSelector global variable is 0 (original) or it has value the number of mutants + 1
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
}

/*****************************************************************/
/****************** SEMu Only Methods @ START ********************/
/*****************************************************************/


// @KLEE-SEMu
//#define ENABLE_KLEE_SEMU_DEBUG 1

inline llvm::Instruction * ExecutorHelperSemu::ks_makeArgSym (Module &module, GlobalVariable * &emptyStrAddr, Instruction *insAfter, Value *memAddr, Type *valtype) {
  llvm::Function *f_make_symbolic = module.getFunction("klee_make_symbolic");
  std::vector<Value *> kms_arguments;
  //TODO: How to handle pointer parameters
  //if (valtype->isPointerTy())
  //  continue;
    
  LLVMContext getGlobalContext;
  if (!emptyStrAddr) {
    //IRBuilder<> builder(getGlobalContext);
    assert (!module.getNamedGlobal("KLEE_SEMu__klee_make_symbolic_emptyStr") && "KLEE_SEMu__klee_make_symbolic_emptyStr already existent in module");
    emptyStrAddr = dyn_cast<GlobalVariable>(module.getOrInsertGlobal("KLEE_SEMu__klee_make_symbolic_emptyStr", ArrayType::get(Type::getInt8Ty(getGlobalContext), 4)));
#if (LLVM_VERSION_MAJOR >= 10)
    emptyStrAddr->setAlignment(llvm::MaybeAlign(1));
#else
    emptyStrAddr->setAlignment(1);
#endif
    emptyStrAddr->setInitializer(ConstantDataArray::getString(getGlobalContext, "str")); //arg->getName().size()?arg->getName():"str")); //
    emptyStrAddr->setConstant(true);
    //Value *emptyStr = builder.CreateGlobalStringPtr("", "KLEE_SEMu__klee_make_symbolic_emptyStr");
    //emptyStrAddr = emptyStr;
  }
  kms_arguments.clear();
  kms_arguments.push_back(ConstantInt::get(getGlobalContext, APInt(32, (uint64_t)(0))));
  kms_arguments.push_back(ConstantInt::get(getGlobalContext, APInt(32, (uint64_t)(0))));
  Instruction *gepStr = GetElementPtrInst::CreateInBounds (emptyStrAddr, kms_arguments);
  gepStr->insertAfter(insAfter);
  Instruction *bitcast8 = new BitCastInst(memAddr, Type::getInt8PtrTy(getGlobalContext));
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
  kms_arguments.push_back(ConstantInt::get(getGlobalContext, APInt(64, (uint64_t)(sizeofVal), false)));
  kms_arguments.push_back(gepStr);
  Instruction *callKMS = CallInst::Create (f_make_symbolic, kms_arguments);
  callKMS->insertAfter(bitcast8);
  
  return callKMS;
}

//This function Set as symbolic the arguments of the entry function (Maybe for 'main' it is better to use command line sym-args)
// This will insert call to 'klee_make_symbolic' over all the arguments.
void ExecutorHelperSemu::ks_setInitialSymbolics (/*ExecutionState &state, */Module &module, Function &Func)
{
  llvm::LLVMContext getGlobalContext;
#if 0
  assert (module.getContext() == getGlobalContext && "");
#endif
  llvm::Function *f_make_symbolic = module.getFunction("klee_make_symbolic");
  
  //The user already added the klee_make_symbolic, no need to proceed
  if (f_make_symbolic)
    return;
    
  //add klee_make_symbolic into the module
  Value* cf = module.getOrInsertFunction("klee_make_symbolic",
                                                   Type::getVoidTy(getGlobalContext),
                                                   Type::getInt8PtrTy(getGlobalContext),
                                                   Type::getInt64Ty(getGlobalContext),
                                                   Type::getInt8PtrTy(getGlobalContext)
                                                   ).getCallee ();
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
void ExecutorHelperSemu::ks_FilterMutants (llvm::Module *module) {

  // We must have at least a mutant to run SEMU mid selector represents maxID + 1)
  if (dyn_cast<ConstantInt>(ks_mutantIDSelectorGlobal->getInitializer())->getZExtValue() < 2) {
    klee_error("SEMU@ERROR: The module passed contains no mutant!");
    exit(1);
  }

  std::set<ExecutionStateHelperSemu::KS_MutantIDType> cand_mut_ids;
  if (!semuCandidateMutantsFile.empty()) {
    std::ifstream ifs(semuCandidateMutantsFile); 
    if (ifs.is_open()) {
      ExecutionStateHelperSemu::KS_MutantIDType tmpid;
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
            
            std::vector<ExecutionStateHelperSemu::KS_MutantIDType> fromsCandIds;
            std::vector<ExecutionStateHelperSemu::KS_MutantIDType> tosCandIds;
            bool lastIsCand = false;
            for (ExecutionStateHelperSemu::KS_MutantIDType mIds = fromMID; mIds <= toMID; mIds++) {
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
                auto *mutIDConstInt = (*i).getCaseValue();
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

void ExecutorHelperSemu::ks_mutationPointBranching(ExecutionState &state, 
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
    //parentE->addedStates.push_back(&state);

    // Mutants just created thus ks_hasToReachPostMutationPoint set to true
    // The mutants created will inherit the value
    if (mut_IDs.size() > 0) {
      state.semuESHelper.ks_hasToReachPostMutationPoint = true;
      state.semuESHelper.ks_startdepth = state.depth;
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

      ExecutionState *ns = state.semuESHelper.ks_branchMut();
      parentE->addedStates.push_back(ns);
      //result.push_back(ns);
      parentE->processTree->attach(state.ptreeNode, ns, &state);
      
      parentE->executeMemoryOperation (*ns, true, parentE->evalConstant(ks_mutantIDSelectorGlobal), 
                              ConstantExpr::create(*it, 8*sizeof(ExecutionStateHelperSemu::KS_MutantIDType)), 0); 
      ns->semuESHelper.ks_mutantID = *it;
      
      ns->semuESHelper.ks_originalMutSisterStates = state.semuESHelper.ks_curBranchTreeNode;

      // Update active
      (state.semuESHelper.ks_numberActiveCmpMutants)++;

      // Free useless space
      ns->semuESHelper.ks_VisitedMutPointsSet.clear();
      ns->semuESHelper.ks_VisitedMutantsSet.clear();

      // On test generation mode, the newly seen mutant is shadowing original
      // Thus is in seed mode (This is KLEE Shadow implementation (XXX))
      if (ExecutionStateHelperSemu::ks_getMode() == ExecutionStateHelperSemu::KS_Mode::TESTGEN_MODE)
        ns->semuESHelper.isTestGenMutSeeding = true;

      // Handle seed phase. Insert mutant in seedMap with same seed as original
      std::map< ExecutionState*, std::vector<SeedInfo> >::iterator sm_it = 
        parentE->seedMap.find(&state);
      if (sm_it != parentE->seedMap.end()) {
        parentE->seedMap[ns] = sm_it->second;
      }
    }
    // No new mutant created because they reach their limit of number of tests
    if (state.semuESHelper.ks_numberActiveCmpMutants == 0 && state.semuESHelper.ks_VisitedMutantsSet.size() >= ks_number_of_mutants) {
      // remove from treenode
      state.semuESHelper.ks_curBranchTreeNode->exState = nullptr;
      // early terminate
      parentE->terminateStateEarly(state, "Original has no possible mutant left");
    }
  }
}


bool ExecutorHelperSemu::ks_checkfeasiblewithsolver(ExecutionState &state, ref<Expr> bool_expr) {
  bool result;
  bool success = parentE->solver->mayBeTrue(state.constraints, bool_expr, result, state.queryMetaData);
  assert(success && "KS: Unhandled solver failure");
  (void) success;
  return result;
}

////>
// TODO TODO: Handle state comparison in here
inline bool ExecutorHelperSemu::ks_outEnvCallDiff (const ExecutionState &a, const ExecutionState &b, std::vector<ref<Expr>> &inStateDiffExp, KScheckFeasible &feasibleChecker) {
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
    ref<Expr> aArg = parentE->eval(&*(a.pc), j+1, const_cast<ExecutionState &>(a)).value;
    ref<Expr> bArg = parentE->eval(&*(b.pc), j+1, const_cast<ExecutionState &>(b)).value;
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
inline bool ExecutorHelperSemu::ks_isOutEnvCallInvoke (Instruction *cii) {
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
        if (outEnvFuncs.count(f->getName().str()) || ks_customOutEnvFuncNames.count(f->getName().str()))
          return true;
        break;
      default:
        ;
    }  
  }
  return false;
}

// In the case of call, check the next to execute instruction, which should be state.pc
inline bool ExecutorHelperSemu::ks_nextIsOutEnv (ExecutionState &state) {
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

bool ExecutorHelperSemu::ks_reachedAMutant(KInstruction *ki) {
  if (ki->inst->getOpcode() == Instruction::Call) {
    Function *f = dyn_cast<CallInst>(ki->inst)->getCalledFunction();
    if (f == ks_mutantIDSelectorGlobal_Func)
      return true;
  }
  return false;
}

#ifdef SEMU_RELMUT_PRED_ENABLED
bool ExecutorHelperSemu::ks_reachedAnOldNew(KInstruction *ki) {
  if (ki->inst->getOpcode() == Instruction::Call) {
    Function *f = dyn_cast<CallInst>(ki->inst)->getCalledFunction();
    if (f == ks_klee_change_function)
      return true;
  }
  return false;
}
#endif

// Check whether the ki is a post mutation point of the state (0) for original
bool ExecutorHelperSemu::ks_checkAtPostMutationPoint(ExecutionState &state, KInstruction *ki) {
  bool ret = false;
  KInstruction * cur_ki = ki;
  Instruction *i;
  
  if (!state.semuESHelper.ks_hasToReachPostMutationPoint) {
    return ret;
  }
  
  if (semuDisableCheckAtPostMutationPoint) {
    state.semuESHelper.ks_hasToReachPostMutationPoint = false; 
    return ret;
  } 
  
  do {
    i = cur_ki->inst;
    if (i->getOpcode() == Instruction::Call) {
      Function *f = dyn_cast<CallInst>(i)->getCalledFunction();
      if (f == ks_postMutationPoint_Func) {
        uint64_t fromMID = dyn_cast<ConstantInt>(dyn_cast<CallInst>(i)->getArgOperand(0))->getZExtValue();
        uint64_t toMID = dyn_cast<ConstantInt>(dyn_cast<CallInst>(i)->getArgOperand(1))->getZExtValue();
        if (state.semuESHelper.ks_mutantID >= fromMID && state.semuESHelper.ks_mutantID <= toMID)
          ret = true;
        // look for the last call in the BB 
        // (since the call was added by mutation tool, there must be following Instruction in BB)
        if (cur_ki != ki) // not the first loop
          parentE->stepInstruction(state);
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
  if (!ret && state.depth > state.semuESHelper.ks_startdepth) {
    state.semuESHelper.ks_hasToReachPostMutationPoint = false;
  }
  return ret;
}

inline bool ExecutorHelperSemu::ks_reachedCheckNextDepth(ExecutionState &state) {
  if (state.depth > ks_nextDepthID)
    return true;
  return false;
}

inline bool ExecutorHelperSemu::ks_reachedCheckMaxDepth(ExecutionState &state) {
  if (state.depth - state.semuESHelper.ks_startdepth > semuMaxDepthWP)
    return true;
  return false;
}

// This function must be called after 'stepInstruction' and 'executeInstruction' function call, which respectively
// have set state.pc to next instruction to execute and state.prevPC to the to the just executed instruction 'ki'
// - In the case of return, 'ki' is used and should be the return instruction (checked after return instruction execution)
// ** We also have the option of limiting symbolic exec depth for mutants and such depth would be a watchpoint.
inline bool ExecutorHelperSemu::ks_watchPointReached (ExecutionState &state, KInstruction *ki) {
#ifdef SEMU_RELMUT_PRED_ENABLED
  // if not reached oldnew, not watchedpoint
  if (state.semuESHelper.ks_old_new == 0)
    return false;
#endif
  // No need to check return of non entry func.
  // Change this to enable/disable intermediate return
  const bool noIntermediateRet = true; 

  if (ki->inst->getOpcode() == Instruction::Ret) {
    //ks_watchpoint = false;
    if (! (noIntermediateRet && state.semuESHelper.ks_checkRetFunctionEntry01NonEntryNeg() < 0))
      return true;
    return false;
  } 
  return ks_reachedCheckMaxDepth(state);
}
///


inline void ExecutorHelperSemu::ks_fixTerminatedChildren(ExecutionState *es, llvm::SmallPtrSet<ExecutionState *, 5> const &toremove, 
				                          bool terminateLooseOrig, bool removeLooseOrigFromAddedPreTerm) {
  if (toremove.empty())
    return;

  ks_fixTerminatedChildrenRecursive (es, toremove);

  // let a child mutant state be the new parent of the group in case this parent terminated
  // XXX at this point, all terminated children are removed from its chindren set
  if (toremove.count(es) > 0) {
    if (!es->semuESHelper.ks_childrenStates.empty()) {
      auto *newParent = *(es->semuESHelper.ks_childrenStates.begin());
      es->semuESHelper.ks_childrenStates.erase(newParent);
    
      newParent->semuESHelper.ks_childrenStates.insert(es->semuESHelper.ks_childrenStates.begin(), es->semuESHelper.ks_childrenStates.end());
      assert (newParent->ks_originalMutSisterStates == nullptr);
      newParent->semuESHelper.ks_originalMutSisterStates = es->semuESHelper.ks_originalMutSisterStates;

      es->semuESHelper.ks_originalMutSisterStates = nullptr;
      es->semuESHelper.ks_childrenStates.clear();
    } else {
      // No state of this mutant at this sub tree remains, decrease active count
      // And remove the corresponding original if no more mutant on the way (active and upcoming)
      std::vector<ExecutionState *> _sub_states;
      llvm::SmallPtrSet<ExecutionState *, 5>  _toremove;
      es->semuESHelper.ks_originalMutSisterStates->getAllStates(_sub_states);
      for (ExecutionState *_s: _sub_states) {
        assert (_s->ks_numberActiveCmpMutants > 0 && "BUG, ks_numberActiveCmpMutants must be > 0");
        (_s->semuESHelper.ks_numberActiveCmpMutants)--;
        if (_s->semuESHelper.ks_numberActiveCmpMutants == 0 
                && _s->semuESHelper.ks_VisitedMutantsSet.size() >= ks_number_of_mutants) {
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
        es->semuESHelper.ks_originalMutSisterStates->ks_cleanTerminatedOriginals(_toremove);
      }

      if (removeLooseOrigFromAddedPreTerm) {
        for (auto *es: _toremove) {
          std::vector<ExecutionState *>::iterator it =
                        std::find(parentE->addedStates.begin(), parentE->addedStates.end(), es);
          if (it != parentE->addedStates.end())
            parentE->addedStates.erase(it);
        }
      }
            
      if (terminateLooseOrig) {
        // If post compare, make sure to add all original states removed during ks_fixTerminatedChildren
        // into ks_justTerminatedStates
        for (auto *es: _toremove) {
          //std::vector<ExecutionState *>::iterator it =
          //          std::find(parentE->addedStates.begin(), parentE->addedStates.end(), es);
          //if (it != parentE->addedStates.end())
          //  parentE->addedStates.erase(it);

          if (ks_terminatedBeforeWP.count(es) > 0)
            ks_terminatedBeforeWP.erase(es);
          parentE->terminateState(*es);
        }
      }
    }
  }
}

void ExecutorHelperSemu::ks_fixTerminatedChildrenRecursive (ExecutionState *pes, llvm::SmallPtrSet<ExecutionState *, 5> const &toremove) {
  // Verify
  if (pes->semuESHelper.ks_mutantID > ks_max_mutant_id) {
    llvm::errs() << "\nSEMU@error: Invalid mutant ID. Potential memory corruption (BUG)."
                 << " The value must be no greater than " << ks_max_mutant_id
                 << ", but got " << pes->semuESHelper.ks_mutantID << "\n\n";
    assert(false);
  }

  if (pes->semuESHelper.ks_childrenStates.empty())
    return;

  std::vector<ExecutionState *> children(pes->semuESHelper.ks_childrenStates.begin(), pes->semuESHelper.ks_childrenStates.end());
  for (ExecutionState *ces: children) {
    ks_fixTerminatedChildrenRecursive(ces, toremove);
    if (toremove.count(ces) > 0) {
      pes->semuESHelper.ks_childrenStates.erase(ces);
      if (! ces->semuESHelper.ks_childrenStates.empty()) {
        auto *newparent = *(ces->semuESHelper.ks_childrenStates.begin());
        pes->semuESHelper.ks_childrenStates.insert(newparent);
        
        ces->semuESHelper.ks_childrenStates.erase(newparent);
        newparent->semuESHelper.ks_childrenStates.insert(ces->semuESHelper.ks_childrenStates.begin(), ces->semuESHelper.ks_childrenStates.end());

        ces->semuESHelper.ks_originalMutSisterStates = nullptr;
        ces->semuESHelper.ks_childrenStates.clear();
      }
    }
  }
}

void ExecutorHelperSemu::ks_moveIntoTerminatedBeforeWP(ExecutionState *es) {
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
    //throw "ExecutorHelperSemu::ks_moveIntoTerminatedBeforeWP exception";
  }
}

void ExecutorHelperSemu::ks_terminateSubtreeMutants(ExecutionState *pes) {
  for (ExecutionState *ces: pes->semuESHelper.ks_childrenStates) {
    ks_terminateSubtreeMutants(ces);
    ks_moveIntoTerminatedBeforeWP(ces);
  }
  ks_moveIntoTerminatedBeforeWP(pes);
}

void ExecutorHelperSemu::ks_getMutParentStates(std::vector<ExecutionState *> &mutParentStates) {
  mutParentStates.clear();
  //TODO TODO: Make efficient
  for(ExecutionState *es: parentE->states) {
    if (es->semuESHelper.ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  /*
  for(ExecutionState *es: ks_atPointPostMutation) {
    if (es->semuESHelper.ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: ks_ongoingExecutionAtWP) {
    if (es->semuESHelper.ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: ks_reachedOutEnv) {
    if (es->semuESHelper.ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: ks_reachedWatchPoint) {
    if (es->semuESHelper.ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: ks_terminatedBeforeWP) {
    if (es->semuESHelper.ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: parentE->addedStates) {
    if (es->semuESHelper.ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }
  for(ExecutionState *es: parentE->removedStates) {
    if (es->semuESHelper.ks_originalMutSisterStates != nullptr)
      mutParentStates.push_back(es);
  }*/
}

void ExecutorHelperSemu::ks_compareStates (std::vector<ExecutionState *> &remainStates, bool outEnvOnly, bool postMutOnly) {
  assert (!(outEnvOnly & postMutOnly) && 
          "must not have both outEnvOnly and postMutOnly enabled simultaneously");

  llvm::SmallPtrSet<ExecutionState *, 5> postMutOnly_hasdiff;

  std::vector<ExecutionState *> mutParentStates;
  ks_getMutParentStates(mutParentStates);

  std::sort(mutParentStates.begin(), mutParentStates.end(),
            [](const ExecutionState *a, const ExecutionState *b)
            {
                return a->semuESHelper.ks_mutantID < b->semuESHelper.ks_mutantID;
            });
  
  std::map<ExecutionState *, ref<Expr>> origSuffConstr;
  
  std::vector<ExecutionState *> correspOriginals;
  
  for (ExecutionState *es: mutParentStates) {
    correspOriginals.clear();
    es->semuESHelper.ks_originalMutSisterStates->getAllStates(correspOriginals);
    //assert (correspOriginals.size() > 0 && (std::string("Error: Empty original state list - Comparing with mutant ID: ")+std::to_string(es->semuESHelper.ks_mutantID)).c_str());

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
      if (es->semuESHelper.ks_old_new == -1)
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
          for (ConstraintSet::constraint_iterator it = tmpes->constraints.begin(), 
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
        if (ites->semuESHelper.ks_mutantID == es->semuESHelper.ks_mutantID) {
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
      if (ks_terminatedBeforeWP.count(es) > 0 && !es->semuESHelper.ks_childrenStates.empty()) {
        auto *newParent = *(es->semuESHelper.ks_childrenStates.begin());
        es->semuESHelper.ks_childrenStates.erase(newParent);
        
        newParent->ks_childrenStates.insert(es->semuESHelper.ks_childrenStates.begin(), es->semuESHelper.ks_childrenStates.end());
        assert (newParent->ks_originalMutSisterStates == nullptr);
        newParent->ks_originalMutSisterStates = es->semuESHelper.ks_originalMutSisterStates;
      }
    }*/
  }

  if (!outEnvOnly) {
    if (!postMutOnly) {
      // Fix original terminated
      if (! mutParentStates.empty()) {
        auto *cands = mutParentStates.front();
        auto *topParent = cands->semuESHelper.ks_originalMutSisterStates;
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
        if(s->semuESHelper.ks_mutantID != 0 && postMutOnly_hasdiff.count(s) == 0)
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
        if (s->semuESHelper.ks_mutantID == 0)
          originals.insert(s);
      }
        
      // Get stats
      ks_numberOfMutantStatesCheckedAtMutationPoint += toremove.size() + postMutOnly_hasdiff.size();
      ks_numberOfMutantStatesDiscardedAtMutationPoint += toremove.size();

      // terminate the state in toremove
      for (auto *s: toremove) {
        //s->pc = s->prevPC;
        parentE->terminateState(*s);
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

bool ExecutorHelperSemu::ks_diffTwoStates (ExecutionState *mState, ExecutionState *mSisState, 
                                  ref<Expr> &origpathPrefix, 
                                  bool outEnvOnly, int &sDiff, 
                                  std::vector<ref<Expr>> &inStateDiffExp) {
  /**
   * Return the result of the solver.
   */
  static const bool usethesolverfordiff = true;

  inStateDiffExp.clear();
  bool result;
  bool success = parentE->solver->mayBeTrue(mState->constraints, origpathPrefix, result, mState->queryMetaData);
  assert(success && "KS: Unhandled solver failure");
  (void) success;
  if (result) {
    // Clear diff expr list
    KScheckFeasible feasibleChecker(this, mState, origpathPrefix, usethesolverfordiff);

    // compare these 
    if (mSisState == nullptr) {
      // Do not have corresponding original anymore.
      sDiff |= ExecutionStateHelperSemu::KS_StateDiff_t::ksPC_DIFF;
    } else {
      if (mState->semuESHelper.ks_numberOfOutEnvSeen != mSisState->semuESHelper.ks_numberOfOutEnvSeen) {
        sDiff |= ExecutionStateHelperSemu::KS_StateDiff_t::ksOUTENV_DIFF;
      }

      if (outEnvOnly &&  (KInstruction*)(mState->pc) && ks_isOutEnvCallInvoke(mState->pc->inst)) {
        sDiff |= ks_outEnvCallDiff (*mState, *mSisState, inStateDiffExp, feasibleChecker) ? 
                              ExecutionStateHelperSemu::KS_StateDiff_t::ksOUTENV_DIFF :
                                                 ExecutionStateHelperSemu::KS_StateDiff_t::ksNO_DIFF;
        // XXX: we also compare states (ks_compareStateWith) here or not?

        if (!ExecutionStateHelperSemu::ks_isNoDiff(sDiff))
          sDiff |= mState->semuESHelper.ks_compareStateWith(*mSisState, ks_mutantIDSelectorGlobal, 
#ifdef SEMU_RELMUT_PRED_ENABLED
                                               ks_isOldVersionGlobal,
#endif
                                               inStateDiffExp, &feasibleChecker, false/*post...*/);
      } else {
        sDiff |= mState->semuESHelper.ks_compareStateWith(*mSisState, ks_mutantIDSelectorGlobal, 
#ifdef SEMU_RELMUT_PRED_ENABLED
                                               ks_isOldVersionGlobal,
#endif
                                             inStateDiffExp, &feasibleChecker, true/*post...*/);
        // XXX if mutant terminated and not original or vice versa, set the main return diff
        // TODO: Meke this more efficient
        if (ks_terminatedBeforeWP.count(mSisState) != ks_terminatedBeforeWP.count(mState))
          sDiff |= ExecutionStateHelperSemu::KS_StateDiff_t::ksRETCODE_DIFF_MAINFUNC;
      }
    }
    // make sure that the sDiff is not having an error. If error, abort
    ExecutionStateHelperSemu::ks_checkNoDiffError(sDiff, mState->semuESHelper.ks_mutantID);
  }
  return result;
}

void ExecutorHelperSemu::ks_createDiffExpr(ExecutionState *mState, ref<Expr> &insdiff, 
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
bool ExecutorHelperSemu::ks_mutantPrePostDiff (ExecutionState *mState, bool outEnvOnly, 
                                      std::vector<ref<Expr>> &inStateDiffExp, 
                                      ref<Expr> &insdiff) {
  bool can_diff = false;
  
  // 1. Find all states with same mutant_ID as `mState` 
  std::vector<ExecutionState*> preMutStateList;
  // XXX: TODO: improve this:w
  for (auto *s: parentE->states) {
    if (s->semuESHelper.ks_mutantID == mState->semuESHelper.ks_mutantID && s->semuESHelper.ks_old_new < 0) 
      preMutStateList.push_back(s);
  }

  // 2. For each state:
  for (auto *preMutState: preMutStateList) {
    //  a) Compare until a pre-post diff is found 
    //  b) check the feasibility of the pre-post diff with `mState`
    //  c) Return the feasible by adding to `insdiff`
    int sDiff = ExecutionStateHelperSemu::KS_StateDiff_t::ksNO_DIFF; 
    ref<Expr> prepathPrefix = ConstantExpr::alloc(1, Expr::Bool);
    for (ConstraintSet::constraint_iterator it = preMutState->constraints.begin(), 
                      ie = preMutState->constraints.end(); it != ie; ++it) {
      prepathPrefix = AndExpr::create(prepathPrefix, *it);
    }
    bool result = ks_diffTwoStates (mState, preMutState, prepathPrefix, 
                                                outEnvOnly, sDiff, inStateDiffExp); 
    if (result) {
      if (!ExecutionStateHelperSemu::ks_isNoDiff(sDiff)) {
        if (/*outputTestCases 
              &&*/ (!(semuTestgenOnlyForCriticalDiffs || outEnvOnly) 
                    || ExecutionStateHelperSemu::ks_isCriticalDiff(sDiff))) {
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
bool ExecutorHelperSemu::ks_compareRecursive (ExecutionState *mState, 
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
  if (mState->semuESHelper.ks_old_new > 0 || postMutOnly) { // only post commit version, except postMut
#endif

  // enter if WP (neither outenv nor post mutation) and the state in not ongoing
  // or is outenv and in its data or is post mutation and is in its data
  if ((!outEnvOnly && !postMutOnly && ks_ongoingExecutionAtWP.count(mState) == 0)
      || (outEnvOnly && ks_reachedOutEnv.count(mState) > 0)
      || (postMutOnly && ks_atPointPostMutation.count(mState) > 0)) {
    for (auto mSisState: mSisStatesVect) {
#ifdef SEMU_RELMUT_PRED_ENABLED
      if (mSisState != nullptr && mSisState->semuESHelper.ks_old_new <= 0 && !postMutOnly) // No need for pre-commit original, except postMut
        continue; 
#endif
      int sDiff = ExecutionStateHelperSemu::KS_StateDiff_t::ksNO_DIFF; 
      ref<Expr> origpathPrefix = origSuffConstr.at(mSisState);
      bool result = ks_diffTwoStates (mState, mSisState, origpathPrefix, 
                                                outEnvOnly, sDiff, inStateDiffExp); 
      if (result) {
        if (ExecutionStateHelperSemu::ks_isNoDiff(sDiff)) {
          if (!outEnvOnly/*at check point*/ && doMaxSat) {
            // XXX put out the paths showing no differences as well
            if (mSisState != nullptr)
              ks_checkMaxSat(mState->constraints, mSisState, inStateDiffExp, mState->semuESHelper.ks_mutantID, sDiff);
          }
  #ifdef ENABLE_KLEE_SEMU_DEBUG
          llvm::errs() << "<==> a state pair of Original and Mutant-" << mState->semuESHelper.ks_mutantID << " are Equivalent.\n\n";
  #endif
        } else {
  #ifdef ENABLE_KLEE_SEMU_DEBUG
          llvm::errs() << "<!=> a state pair of Original and Mutant-" << mState->semuESHelper.ks_mutantID << " are Different.\n\n";
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
                      || ExecutionStateHelperSemu::ks_isCriticalDiff(sDiff))) {
            // On test generation mode, if the mutant of mState reached maximum quota of tests, 
            // stop comparing. 
            // insert if not yet in map
            auto irp_m = mutants2gentestsNum.insert(std::make_pair(mState->semuESHelper.ks_mutantID, 0));
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
            ExecutionState *tmp_mState = mState->semuESHelper.cloneParentES();
            if (! semuDisableStateDiffInTestGen)
                tmp_mState->addConstraint (insdiff); 
            // TODO FIXME: Handle cases where the processTestCase call fail and no test is gen (for the following code: mutants stats uptades)
            parentE->interpreterHandler->processTestCase(*tmp_mState, nullptr, nullptr); //std::to_string(mState->semuESHelper.ks_mutantID).insert(0,"Mut").c_str());
            delete tmp_mState;
            ////size_t clen = mState->constraints.size();
            ////mState->addConstraint (insdiff); 
            ////interpreterHandler->processTestCase(*mState, nullptr, nullptr); //std::to_string(mState->semuESHelper.ks_mutantID).insert(0,"Mut").c_str());
            bool testgen_was_good = ks_writeMutantTestsInfos(mState->semuESHelper.ks_mutantID, ++gentestid); //Write info needed to know which test for which mutant 
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
            ks_checkMaxSat(mState->constraints, mSisState, inStateDiffExp, mState->semuESHelper.ks_mutantID, sDiff);
          }

        }
      } else {
  #ifdef ENABLE_KLEE_SEMU_DEBUG
        llvm::errs() << "# Infesible differential between an original and a Mutant-" << mState->semuESHelper.ks_mutantID <<".\n\n";
  #endif
      }
    }
  }
  
#ifdef SEMU_RELMUT_PRED_ENABLED
  }//~ only post commit version
#endif

  //compare children as well
  for (ExecutionState *es: mState->semuESHelper.ks_childrenStates) {
    diffFound |= ks_compareRecursive (es, mSisStatesVect, origSuffConstr, outEnvOnly, postMutOnly, postMutOnly_hasdiff);
  }
  
  return diffFound;
}


// This take the path condition common to a mutant and original, together 
// with the conditions of equality, for each state variable, between
// original and mutant
void ExecutorHelperSemu::ks_checkMaxSat (ConstraintSet const &mutPathCond,
                                ExecutionState const *origState,
                                std::vector<ref<Expr>> &stateDiffExprs,
                                ExecutionStateHelperSemu::KS_MutantIDType mutant_id, int sDiff) {
  ConstraintSet const &origPathCond = origState->constraints;
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
    pmaxsat_solver.setSolverTimeout(parentE->coreSolverTimeout.toSeconds());
    // TODO fix z3 maxsat and uncoment bellow
    //pmaxsat_solver.checkMaxSat(hardClauses, stateDiffExprs, nMaxFeasibleDiffs, nMaxFeasibleEqs);
#endif //~KS_Z3MAXSAT_SOLVER__H
  }
  
  // update using nTrue and nFalse
  nMaxFeasibleDiffs += nTrue;
  nSoftClauses -= nFalse;

  ks_writeMutantStateData (mutant_id, nSoftClauses, nMaxFeasibleDiffs, nMaxFeasibleEqs, sDiff, origState);
}

void ExecutorHelperSemu::ks_writeMutantStateData(ExecutionStateHelperSemu::KS_MutantIDType mutant_id,
                                unsigned nSoftClauses,
                                unsigned nMaxFeasibleDiffs,
                                unsigned nMaxFeasibleEqs,
                                int sDiff,
                                ExecutionState const *origState) {
  static const std::string fnPrefix("mutant-");
  static const std::string fnSuffix(".semu");
  static std::map<ExecutionStateHelperSemu::KS_MutantIDType, std::string> mutantID2outfile;
  //llvm::errs() << "MutantID | nSoftClauses  nMaxFeasibleDiffs  nMaxFeasibleEqs | Diff Type\n";
  //llvm::errs() << mutant_id << " | " << nSoftClauses << "  " << nMaxFeasibleDiffs << "  " << nMaxFeasibleEqs << " | " << sDiff << "\n";
  std::string header;
  std::string out_file_name = mutantID2outfile[mutant_id];
  if (out_file_name.empty()) {
    mutantID2outfile[mutant_id] = out_file_name = 
           parentE->interpreterHandler->getOutputFilename(fnPrefix+std::to_string(mutant_id)+fnSuffix);
    header.assign("ellapsedTime(s),MutantID,nSoftClauses,nMaxFeasibleDiffs,nMaxFeasibleEqs,Diff_Type,OrigState,WatchPointID,MaxDepthID\n");
  }

  double ellapsedtime = (time::getWallTime() - ks_runStartTime).toSeconds();
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

bool ExecutorHelperSemu::ks_writeMutantTestsInfos(ExecutionStateHelperSemu::KS_MutantIDType mutant_id, unsigned testid) {
  static const std::string fnPrefix("mutant-");
  static const std::string fnSuffix(".ktestlist");
  static const std::string ktest_suffix(".ktest");
  static std::map<ExecutionStateHelperSemu::KS_MutantIDType, std::string> mutantID2outfile;
  static std::string semu_exec_info_file;
  std::string mutant_data_header;
  std::string info_header;

  // put out the semu ececution info
  if (semu_exec_info_file.empty()) {
    semu_exec_info_file = parentE->interpreterHandler->getOutputFilename("semu_execution_info.csv");
    info_header.assign("ellapsedTime(s),stateCompareTime(s),#MutStatesForkedFromOriginal,#MutStatesEqWithOrigAtMutPoint\n");
  } 
  double ellapsedtime = (time::getWallTime() - ks_runStartTime).toSeconds();
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
           parentE->interpreterHandler->getOutputFilename(fnPrefix+std::to_string(mutant_id)+fnSuffix);
    mutant_data_header.assign("ellapsedTime(s),MutantID,ktest\n");
  }

  ellapsedtime = (time::getWallTime() - ks_runStartTime).toSeconds();
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

void ExecutorHelperSemu::ks_loadKQueryConstraints(std::vector<ConstraintSet> &outConstraintsList) {
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
        outConstraintsList.emplace_back(QC->Constraints);
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
bool ExecutorHelperSemu::ks_hasJustForkedTG (ExecutionState &state, KInstruction *ki) {
  if (!state.semuESHelper.ks_childrenStates.empty()) {
    return true;
  }
  return false;
}

void ExecutorHelperSemu::ks_fourWayForksTG() {
  // TODO implement this
  // TODO XXX Clear the ks_childrenStates of every state
}


/// Avoid infinite loop mutant to run indefinitely: simple fix
void ExecutorHelperSemu::ks_CheckAndBreakInfinitLoop(ExecutionState &curState, ExecutionState *&prevState, time::Point &initTime) {
  if(curState.semuESHelper.ks_mutantID > 0) {   //TODO: how about when we reach watch point
    if (prevState != &curState) {
      prevState = &curState;
      initTime = time::getWallTime();
    } else if (time::seconds(semuLoopBreakDelay) < time::getWallTime() - initTime) {
      klee_message((std::string("SEMU@WARNING: Loop Break Delay reached for mutant ")+std::to_string(curState.semuESHelper.ks_mutantID)).c_str());
      parentE->terminateStateEarly(curState, "infinite loop"); //Terminate mutant
      // XXX Will be removed from searcher and processed bellow
      //continue;
    }
  }
}

/// Return true if the state comparison actually happened, false otherwise. This help to know if we need to call updateStates or not
bool ExecutorHelperSemu::ks_CheckpointingMainCheck(ExecutionState &curState, KInstruction *ki, bool isSeeding, uint64_t precond_offset) {
  // // FIXME: We just checked memory and some states will be killed if memory exeeded and we will have some problem in comparison
  // // FIXME: For now we assume that the memory limit must not be exceeded, need to find a way to handle this later
  if (parentE->atMemoryLimit) {
    if (semuEnableNoErrorOnMemoryLimit) {
      static time::Point prevWarnMemLimitTime = time::getWallTime();
      // Print at most one warning per 3 seconds, to avoid clogging
      if (time::getWallTime() - prevWarnMemLimitTime > time::seconds(3)) {
        klee_warning("SEMU@WARNING: reached memory limit and killed states. You could increase memory limit or restrict symbex.");
        prevWarnMemLimitTime = time::getWallTime();
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
  //  parentE->terminateStateEarly(curState, "@SEMU: unreachable instruction reached");
  //  ks_terminated = true;
  //} else {
    ks_terminated = !ks_justTerminatedStates.empty();
    // XXX Do not check whe the instruction is PHI or EHPad, except the state terminated
    if (ks_terminated || !(llvm::isa<llvm::PHINode>(ki->inst) /*|| ki->inst->isEHPad()*/)) {
      // A state can have both in ks_atPostMutation and ks_atNextCheck true
      ks_OutEnvReached = ks_nextIsOutEnv (curState);
      ks_WPReached = ks_watchPointReached (curState, ki);
      if (ks_WPReached) {
        if (curState.semuESHelper.ks_mutantID == 0) {
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
          parentE->seedMap.find(&curState);
        if (it != parentE->seedMap.end()) {
          // keep state for restoration after compare (seedMap not set un UpdateStates)
          backed_seedMap[&curState] = it->second;  
          parentE->seedMap.erase(it);
        }
      } else {
        parentE->searcher->update(&curState, std::vector<ExecutionState *>(), std::vector<ExecutionState *>({&curState}));
      }

      // Terminated has priority, then outEnv
      if (! curTerminated) {
        // add to ks_reachedWatchPoint if so
        if (ks_OutEnvReached) {
          ks_reachedOutEnv.insert(&curState);
          curState.semuESHelper.ks_numberOfOutEnvSeen++;
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
      //std::set_intersection(parentE->addedStates.begin(), parentE->addedStates.end(), ks_justTerminatedStates.begin(), ks_justTerminatedStates.end(), addedNdeleted.begin());
      for (auto *s: parentE->addedStates)
        if (ks_justTerminatedStates.count(s))
          addedNdeleted.push_back(s);

      // Update all terminated 
      ks_terminatedBeforeWP.insert(ks_justTerminatedStates.begin(), ks_justTerminatedStates.end());
      if (parentE->atMemoryLimit) {
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
    parentE->updateStates(0);

    // Remove addedNdeleted form both searcher and seedMap
    if (! addedNdeleted.empty()) {
      if (isSeeding) {
        for (auto *s: addedNdeleted) {
          std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
            parentE->seedMap.find(s);
          if (it != parentE->seedMap.end()) {
            parentE->seedMap.erase(it);
          }
        }
      } else {
        parentE->searcher->update(nullptr, std::vector<ExecutionState *>(), addedNdeleted);
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
          ks_ongoingExecutionAtWP.size() >= parentE->states.size()) {

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
            llvm::errs() << "# SEMU@Status: Comparing states: " << parentE->states.size() 
                        << " States" << (ks_hasOutEnv?" (OutEnv)":
                            (ks_isAtPostMut?" (PostMutationPoint)":" (Checkpoint)"))
                        << ".\n";
          auto elapsInittime = time::getWallTime();
          ks_compareStates(remainWPStates, ks_hasOutEnv/*outEnvOnly*/, ks_isAtPostMut/*postMutOnly*/);
          if(!semuQuiet)
	    llvm::errs() << "# SEMU@Status: State Comparison Done! (" << (time::getWallTime() - elapsInittime).toSeconds() << " seconds)\n";
          ks_totalStateComparisonTime += (time::getWallTime() - elapsInittime).toSeconds();
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
            if (parentE->addedStates.empty())
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
            r_backup->semuESHelper.ks_hasToReachPostMutationPoint = false;
            for (auto it=remainWPStates.begin()+1, ie=remainWPStates.end();
                      it != ie; ++it) {
              // disable ks_hasToReachPostMutationPoint
              (*it)->semuESHelper.ks_hasToReachPostMutationPoint = false;
              // The state that at the same time reached postmutation 
              // and next check are added into ongoing
              if (ks_reachedCheckNextDepth(*(*it)))
                ks_ongoingExecutionAtWP.insert(*it);
              else
                parentE->addedStates.push_back(*it);
            }
            if (parentE->addedStates.empty()) {
              parentE->addedStates.push_back(r_backup);
            } else {
              if (ks_reachedCheckNextDepth(*r_backup))
                ks_ongoingExecutionAtWP.insert(r_backup);
              else
                parentE->addedStates.push_back(r_backup);
            }
          }
        } else {
          parentE->addedStates.insert(parentE->addedStates.end(), remainWPStates.begin(), remainWPStates.end());
        }
          
        // If outenv is empty, it means that every state reached checkpoint,
        // We can then termintate crash states and clear the term and WP sets
        if (!ks_hasOutEnv) { 
          if (!ks_isAtPostMut) {
            for (SmallPtrSet<ExecutionState *, 5>::iterator it = ks_terminatedBeforeWP.begin(), 
                  ie = ks_terminatedBeforeWP.end(); it != ie; ++it ) {
              //(*it)->pc = (*it)->prevPC;
              parentE->terminateState (**it);
            }
            // Clear
            ks_terminatedBeforeWP.clear();
            ks_nextDepthID++; 

            // Clear
            ks_reachedWatchPoint.clear();

            // add back ongoing states
            parentE->addedStates.insert(parentE->addedStates.end(), ks_ongoingExecutionAtWP.begin(), ks_ongoingExecutionAtWP.end());
            // Clear
            ks_ongoingExecutionAtWP.clear();
          } else {
            // Clear
            ks_atPointPostMutation.clear();
          }

	        if(!semuQuiet)
            llvm::errs() << "# SEMU@Status: After nextdepth point ID=" << (ks_nextDepthID-1) 
                        << " There are " << parentE->addedStates.size() 
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
        if (parentE->searcher)
          parentE->searcher->update(0, parentE->removedStates/*adding*/, std::vector<ExecutionState *>());

        // in seeding mode, since seedMap is not augmented in updateState,
        // we update it with remaining states (in addesStates vector) before updateStates
        if (isSeeding) {
          assert ((/*ks_hasOutEnv || */parentE->seedMap.empty()) && "SeedMap must be empty at checks"/*point"*/);
          for(auto *s: parentE->addedStates) {
            std::map< ExecutionState*, std::vector<SeedInfo> >::iterator it = 
              backed_seedMap.find(s);
            assert (it != backed_seedMap.end() && "A state is not in backed seed map but remains."); 
            parentE->seedMap[s] = it->second;
            backed_seedMap.erase(it);
          }
          // if checkpoint, clear backed_seedMap.
          if (!ks_hasOutEnv && !ks_isAtPostMut)
            backed_seedMap.clear();
        } else {
          backed_seedMap.clear(); // seed mode already passed, clear any ramining
        }

        // take account of addedStates and removedStates
        parentE->updateStates(0);
      } else {
        // in Seen Mode, the seedMap must not be empty
        if (isSeeding && parentE->seedMap.empty()) {
          llvm::errs() << "\n>> (BUG) States size: "<< parentE->states.size()<<". Sum of check stages sizes: " 
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


void ExecutorHelperSemu::ks_heuristicbasedContinueStates (std::vector<ExecutionState*> const &statelist,
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

void ExecutorHelperSemu::ks_process_closestout_recursive(llvm::CallGraphNode *cgnode,
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

void ExecutorHelperSemu::ks_initialize_ks_instruction2closestout_distance(llvm::Module* mod) {
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
      auto *root = call_graph[ks_entryFunction];
      ks_process_closestout_recursive(root, cgnode_to_midist_out_ret);
    } while (false);
  }
}

std::map<llvm::Instruction*, unsigned> ExecutorHelperSemu::ks_instruction2closestout_distance;

bool ExecutorHelperSemu::ks_getMinDistToOutput(ExecutionState *lh, ExecutionState *rh) {
  // We safely use pc not prevPC because 
  // the states here must be executing (not terminated)
  // and pc is actually what will be executed next
  llvm::Instruction *lhInst = lh->pc->inst;
  llvm::Instruction *rhInst = rh->pc->inst;
  return ks_instruction2closestout_distance[lhInst] < ks_instruction2closestout_distance[rhInst];
}

void ExecutorHelperSemu::ks_applyMutantSearchStrategy() {
  // whether to apply the search also for the original (discard some states)
  // XXX Keep this to False. if set to true, there is need to update the
  // Original children state tree to have no problem on state comparison
  const bool original_too = false;
  // Whether to apply for a mutant ID regardless of whether 
  //ganerated from same original state
  const bool mutant_alltogether = true;
  std::map<ExecutionStateHelperSemu::KS_MutantIDType, std::vector<ExecutionState*>> 
                                                    mutId2states;
  for (auto *s: ks_reachedWatchPoint) {
    if (original_too || s->semuESHelper.ks_mutantID != 0)
      mutId2states[s->semuESHelper.ks_mutantID].push_back(s);
  }

  // Update seenCheckpoint
  for (auto &p: mutId2states)
    for (auto *s: p.second)
      s->semuESHelper.ks_numSeenCheckpoints++;
  
  // add ongoing too if enabled
  if (mutant_alltogether) {
    for (auto *s: ks_ongoingExecutionAtWP) {
      // TODO Optimize the first condition with iterator
      if (mutId2states.count(s->semuESHelper.ks_mutantID) > 0 && 
              (original_too || s->semuESHelper.ks_mutantID != 0)) {
        mutId2states[s->semuESHelper.ks_mutantID].push_back(s);
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
        s->semuESHelper.ks_startdepth = s->depth;
        ks_reachedWatchPoint.erase(s);
      }
    }
    for (auto *s: toStop) {
      if (ks_reachedWatchPoint.count(s)) {
        if (!toContinue.empty() && // make sure that not everything is removed
                s->semuESHelper.ks_numSeenCheckpoints <= semuGenTestForDiscardedFromCheckNum) {
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
    parentE->terminateState(*es);
  }
}

void ExecutorHelperSemu::ks_eliminateMutantStatesWithMaxTests(bool pre_compare) {
  std::set<ExecutionStateHelperSemu::KS_MutantIDType> reached_max_tg;
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
        if (reached_max_tg.count((*m_it)->semuESHelper.ks_mutantID) > 0) {
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
        if (reached_max_tg.count((*m_it)->semuESHelper.ks_mutantID) > 0) {
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
        if (reached_max_tg.count((*m_it)->semuESHelper.ks_mutantID) > 0) {
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
        if (reached_max_tg.count((*m_it)->semuESHelper.ks_mutantID) > 0) {
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
        if (reached_max_tg.count((*m_it)->semuESHelper.ks_mutantID) > 0) {
          tmp_one_toTerminate.insert(*m_it);
        }
      }
      for (auto *s: tmp_one_toTerminate) {
        ks_ongoingExecutionAtWP.erase(s);
        toTerminate.insert(s);
      }
    } else {
      llvm::SmallPtrSet<ExecutionState *, 5> tmp_one_toTerminate;
      for (auto m_it=parentE->addedStates.begin(); 
                                      m_it != parentE->addedStates.end();++m_it) {
        if (reached_max_tg.count((*m_it)->semuESHelper.ks_mutantID) > 0) {
          tmp_one_toTerminate.insert(*m_it);
        }
      }
      for (auto *s: tmp_one_toTerminate) {
        std::vector<ExecutionState *>::iterator it =
              std::find(parentE->addedStates.begin(), parentE->addedStates.end(), s);
        parentE->addedStates.erase(it);
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
        parentE->terminateState(*es);
        //ks_terminatedBeforeWP.insert(es);
      }
    }
  }
}

#ifdef SEMU_RELMUT_PRED_ENABLED
// get a new ID for a version split node
inline unsigned long long ExecutorHelperSemu::ks_get_next_oldnew_split_id() {
  return ks_current_oldnew_split_id++;
}

void ExecutorHelperSemu::ks_oldNewPrepareModule (llvm::Module *mod) {
  LLVMContext getGlobalContext;
  // - Set isold global
  if (mod->getNamedGlobal(ks_isOldVersionName)) {
    llvm::errs() << "The gobal variable '" << ks_isOldVersionName
                 << "' already present in code!\n";
    assert(false &&  "ERROR: Module already mutated!");
    exit(1);
  }
  mod->getOrInsertGlobal(ks_isOldVersionName,
                           llvm::Type::getInt1Ty(getGlobalContext));
  ks_isOldVersionGlobal = mod->getNamedGlobal(ks_isOldVersionName);
  //ks_isOldVersionGlobal->setAlignment(4);
  ks_isOldVersionGlobal->setInitializer(llvm::ConstantInt::get(
                            getGlobalContext, llvm::APInt(1, 0, false)));

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
                        getGlobalContext, "entry", ks_klee_change_function);
  llvm::Function::arg_iterator args = ks_klee_change_function->arg_begin();
  llvm::Value *old_v = llvm::dyn_cast<llvm::Value>(args++);
  llvm::Value *new_v = llvm::dyn_cast<llvm::Value>(args++);
	
  //llvm::IRBuilder<> builder(block);
  // load ks_isOldVersionGlobal
  //llvm::Value *isold = builder.CreateLoad(ks_isOldVersionGlobal);
  llvm::LoadInst *isold = new LoadInst(
#if (LLVM_VERSION_MAJOR >= 10)
    ks_isOldVersionGlobal->getType()->getPointerElementType(),
#endif
    ks_isOldVersionGlobal,
    "ks_isOldVersionGlobal",
    block
  );
  // create select
  //llvm::Value *selection = builder.CreateSelect(isold, old_v, new_v);
  llvm::SelectInst *selection = SelectInst::Create(isold, old_v, new_v);
  block->getInstList().push_back(selection);
  ////builder.CreateBinOp(llvm::Instruction::Mul, x, y, "tmp");
  //builder.CreateRet(selection);
  block->getInstList().push_back(ReturnInst::Create(getGlobalContext, selection));
}

void ExecutorHelperSemu::ks_oldNewBranching(ExecutionState &state) {
  assert (state.semuESHelper.ks_old_new == 0);
  TimerStatIncrementer timer(stats::forkTime);

  state.semuESHelper.ks_startdepth = state.depth;

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
    //parentE->addedStates.push_back(&state);

    // In the test generation mode, do not even generate more mutants 
    // If reached max number of tests
    if (ks_outputTestsCases) {
      auto mapit = mutants2gentestsNum.find(state.semuESHelper.ks_mutantID);
      if (mapit != mutants2gentestsNum.end() 
                  && mapit->second >= semuMaxNumTestGenPerMutant) {
        // Stop
        // remove from treenode
        state.semuESHelper.ks_curBranchTreeNode->exState = nullptr;
        // early terminate
        parentE->terminateStateEarly(state, "Original has no possible mutant left");
        return;
      }
    }
    
    // get old_new_split_id
    state.semuESHelper.ks_oldnew_split_id = ks_get_next_oldnew_split_id();

    ExecutionState *ns = state.branch();
    parentE->addedStates.push_back(ns);
    //result.push_back(ns);
    parentE->processTree->attach(state.ptreeNode, ns, &state);
    
    parentE->executeMemoryOperation (*ns, true, parentE->evalConstant(ks_isOldVersionGlobal), ConstantExpr::create(1, 1), 0);    // isOld is a boolean (1 bit int). 1 for True
    parentE->executeMemoryOperation (state, true, parentE->evalConstant(ks_isOldVersionGlobal), ConstantExpr::create(0, 1), 0);    // isOld is a boolean (1 bit int). 0 for False
    ns->semuESHelper.ks_old_new = -1;
    state.semuESHelper.ks_old_new = 1;
    
    // Handle seed phase. Insert old in seedMap with same seed as new
    std::map< ExecutionState*, std::vector<SeedInfo> >::iterator sm_it = 
      parentE->seedMap.find(&state);
    if (sm_it != parentE->seedMap.end()) {
      parentE->seedMap[ns] = sm_it->second;
    }
  }
}
#endif

/**/
// This function is called when running Under Const
bool ExecutorHelperSemu::ks_lazyInitialize (ExecutionState &state, KInstruction *ki) {
  return true;
}

//~KS

/*****************************************************************/
/*~~~~~~~~~~~~~~~~ SEMu Only Methods @ END ~~~~~~~~~~~~~~~~~~~~~~*/
/*****************************************************************/