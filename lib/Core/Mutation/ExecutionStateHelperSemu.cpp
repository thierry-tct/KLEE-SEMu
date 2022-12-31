
#include "ExecutionStateHelperSemu.h"
#include "../ExecutionState.h"
#include "klee/Module/Cell.h"
#include "klee/Module/KInstruction.h"
#include "klee/Module/KModule.h"

#include "../Memory.h"

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
#include "llvm/IR/Function.h"
#else
#include "llvm/Function.h"
#endif
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace klee;

ExecutionStateHelperSemu::ExecutionStateHelperSemu(ExecutionState *parent, bool DebugLogStateMergeArg) : 
  ks_mutantID(0),
  ks_originalMutSisterStates(nullptr),
  ks_curBranchTreeNode(new KS_OrigBranchTreeNode(nullptr, parent)),
  ks_hasToReachPostMutationPoint(false),
  ks_startdepth(0),
  ks_numberOfOutEnvSeen(0),
  ks_numberActiveCmpMutants(0),
  DebugLogStateMerge(DebugLogStateMergeArg),
  parentES(parent) {
}

ExecutionStateHelperSemu::ExecutionStateHelperSemu(ExecutionState *parent, const ExecutionStateHelperSemu &other, bool DebugLogStateMergeArg) :
  ks_mutantID(other.ks_mutantID),
  ks_originalMutSisterStates(nullptr),
  ks_curBranchTreeNode(nullptr),
  ks_hasToReachPostMutationPoint(other.ks_hasToReachPostMutationPoint),
  ks_startdepth(other.ks_startdepth),
  ks_numberOfOutEnvSeen(other.ks_numberOfOutEnvSeen),
  ks_numberActiveCmpMutants(other.ks_numberActiveCmpMutants),
#ifdef SEMU_RELMUT_PRED_ENABLED
  ks_old_new(other.ks_old_new),
  ks_oldnew_split_id(other.ks_oldnew_split_id),
#endif
  DebugLogStateMerge(DebugLogStateMergeArg),
  parentES(parent) {
}

// @KLEE-SEMu
ExecutionStateHelperSemu::KS_Mode ExecutionStateHelperSemu::ks_mode = ExecutionStateHelperSemu::KS_Mode::SEMU_MODE;

// Generate mutant state from Original
ExecutionState *ExecutionStateHelperSemu::ks_branchMut() {
  // Do not increase depth since this is not KLEE's normal fork
  // control mutant state explosion
  //depth++;  

  ExecutionState *falseState = new ExecutionState(*parentES);
  falseState->setID();
  falseState->coveredNew = false;
  falseState->coveredLines.clear();

  return falseState;
}


// This function help to know whetehr the function that is returning is 
// an entry point function (return 0 if 'main' and 1 if '__user_main' [which just returned and was removed from stack]) 
// or not(return negative number)
int ExecutionStateHelperSemu::ks_checkRetFunctionEntry01NonEntryNeg() {
  if (parentES->stack.size() == 2)
    return (parentES->stack.at(1).kf->function->getName() == "__uClibc_main"
            && parentES->prevPC->inst->getParent()->getParent()->getName() == "__user_main")
            ? 1: -1;
  else
    return (! parentES->stack.back().caller)? 0: -1;
}
  
bool ExecutionStateHelperSemu::ks_stackHasAnyFunctionOf(std::set<std::string> &funcnames) {
  for (auto &sf: parentES->stack)
    if (funcnames.count(sf.kf->function->getName().str()) > 0)
      return true;
  return false;
}

int ExecutionStateHelperSemu::ks_compareStateWith (const ExecutionState &b, llvm::Value *MutantIDSelectDeclIns, 
#ifdef SEMU_RELMUT_PRED_ENABLED
                                               llvm::Value *IsOldVersionDeclIns,
#endif
                                         std::vector<ref<Expr>> &inStateDiffExp, KScheckFeasibleBase *feasibleChecker, 
                                         bool postExec, bool checkRegs/*=false*/) {
  //if (pc != b.pc)     //Commented beacause some states may terminate early but should still be considered(they may have different PC)
  //  return false;
  
  const bool checkLocals = true;

  // The returned code
  int returnedCode = ksNO_DIFF;
  
  KInstIterator aPC, bPC;
  if (postExec) {
    aPC = parentES->prevPC;
    bPC = b.prevPC;
  } else {
    aPC = parentES->pc;
    bPC = b.pc;
  }

  //TODO>
  //> First: They should have same PC (They both executed the watch point)
  if (aPC != bPC) { 
#ifdef ENABLE_KLEE_SEMU_DEBUG
    llvm::errs() << "--> aPC != bPC\n";  
#endif
    // They are certainly different. Insert true to show that
    inStateDiffExp.push_back(ConstantExpr::alloc(1, Expr::Bool));
    returnedCode |= ksPC_DIFF;

    if (!aPC || !bPC) 
      return returnedCode;

    // XXX: Necessary - Stop if the 2 states are in different functions
    // We make this relax because the same statement can have 2 locations
    // one in original code, one in mutant code. and a mutant is limited within a function
    if (aPC->inst->getParent()->getParent() != bPC->inst->getParent()->getParent())
      return returnedCode;
  } 

    // we cannot compare states if the two have different call stack: Therefore the asserts
  if (false) //XXX: disable this for now, 
  {
    std::vector<StackFrame>::const_iterator itA = parentES->stack.begin();
    std::vector<StackFrame>::const_iterator itB = b.stack.begin();
    while (itA!=parentES->stack.end() && itB!=b.stack.end()) {
      // XXX vaargs?
      if (itA->caller != itB->caller || itA->kf != itB->kf) {
#ifdef ENABLE_KLEE_SEMU_DEBUG
        llvm::errs() << "--> itA->caller!=itB->caller || itA->kf!=itB->kf\n";
#endif
        llvm::errs() << "# A's Stack:";
        parentES->dumpStack(llvm::errs()); 
        llvm::errs() << "# B's Stack:";
        b.dumpStack(llvm::errs()); 
        //llvm::errs() << itA->caller << " " << itB->caller << "\n";
        //itA->caller->inst->dump(); itB->caller->inst->dump();
        //llvm::errs() << (itA->caller != itB->caller) << " " << (itA->kf != itB->kf) << "\n";
        assert (false && "@SEMU-ERROR: Different call stack: diff func (wrong watch point)");
        llvm::errs() << "\n@SEMU-ERROR: Different call stack: diff func (wrong watch point)\n";
        return ksFAILURE_BUG;
      }
      ++itA;
      ++itB;
    }
    if (itA!=parentES->stack.end() || itB!=b.stack.end()) {
#ifdef ENABLE_KLEE_SEMU_DEBUG
      llvm::errs() << "--> itA!=stack.end() || itB!=b.stack.end()\n";
#endif
      llvm::errs() << "# A's Stack:";
      parentES->dumpStack(llvm::errs()); 
      llvm::errs() << "# B's Stack:";
      b.dumpStack(llvm::errs()); 
      assert (false && "@SEMU-ERROR: Different call stack: diff length (wrong watch point)");
      llvm::errs() << "\n@SEMU-ERROR: Different call stack: diff length (wrong watch point)\n";
      return ksFAILURE_BUG;
    }
  }

  //Here we are sure that both have same call stack
  //> Second is the watchpoint a ret instruction?
  if (llvm::isa<llvm::ReturnInst>(bPC->inst)) { //make sure that both watch points are same type
    if (llvm::ReturnInst *ri = llvm::dyn_cast<llvm::ReturnInst>(aPC->inst)) {
      const char *mainFName[] = {"main", "__user_main"};
      int noEntry_NegEntry0_1 = ks_checkRetFunctionEntry01NonEntryNeg();
      
      if (noEntry_NegEntry0_1 >= 0) { //Entry point Function
        if (ri->getParent()->getParent()->getName() == mainFName[noEntry_NegEntry0_1]) { //Check only return values
          if (ks_lastReturnedVal.compare(b.semuESHelper.ks_lastReturnedVal)) {
  #ifdef ENABLE_KLEE_SEMU_DEBUG
            llvm::errs() << "--> return Codes differ: main function.\n";
  #endif
            ref<Expr> tmpexpr = NeExpr::create(ks_lastReturnedVal, b.semuESHelper.ks_lastReturnedVal);
            if (feasibleChecker->isFeasible(tmpexpr)) {
              inStateDiffExp.push_back(tmpexpr);
              returnedCode |= ksRETCODE_DIFF_MAINFUNC;
            }
          }
          else {
            //return ksNO_DIFF;
          }
        } else {  //Check return val and globals
          if (ks_lastReturnedVal.compare(b.semuESHelper.ks_lastReturnedVal)) {
  #ifdef ENABLE_KLEE_SEMU_DEBUG
            llvm::errs() << "--> return Codes differ: other entry point function.\n";
  #endif
            ref<Expr> tmpexpr = NeExpr::create(ks_lastReturnedVal, b.semuESHelper.ks_lastReturnedVal);
            if (feasibleChecker->isFeasible(tmpexpr)) {
              inStateDiffExp.push_back(tmpexpr);
              returnedCode |= ksRETCODE_DIFF_ENTRYFUNC;
            }
          }
          //checkLocals  = false;
        }
      } else {  //Check both returned val and Globals and locals
        if (ks_lastReturnedVal.compare(b.semuESHelper.ks_lastReturnedVal)) {
  #ifdef ENABLE_KLEE_SEMU_DEBUG
          llvm::errs() << "--> return Codes differ: non entry point function.\n";
  #endif
          ref<Expr> tmpexpr = NeExpr::create(ks_lastReturnedVal, b.semuESHelper.ks_lastReturnedVal);
          if (feasibleChecker->isFeasible(tmpexpr)) {
            inStateDiffExp.push_back(tmpexpr);
            returnedCode |= ksRETCODE_DIFF_OTHERFUNC;
          }
        }
        //checkLocals  = true;
      }
    } 
  }

  // XXX is it even possible for these to differ? does it matter? probably
  // implies difference in object states?
  // check this only when we also compare local vars
  if (checkLocals && parentES->symbolics!=b.symbolics) {
#ifdef ENABLE_KLEE_SEMU_DEBUG
    llvm::errs() << "--> symbolics!=b.symbolics\n";
#endif
    assert (false && "@SEMU-ERROR: Different in symbolics. BUG?");
    llvm::errs() << "\n@SEMU-ERROR: Different in symbolics. BUG?\n";
    inStateDiffExp.push_back(ConstantExpr::alloc(1, Expr::Bool));
    returnedCode |= ksSYMBOLICS_DIFF;
    return returnedCode;
  }

  // We cannot merge if addresses would resolve differently in the
  // states. This means:
  // 
  // 1. Any objects created since the branch in either object must
  // have been free'd.
  //
  // 2. We cannot have free'd any pre-existing object in one state
  // and not the other

  if (DebugLogStateMerge) {
    llvm::errs() << "\tchecking object states\n";
    llvm::errs() << "A: " << parentES->addressSpace.objects << "\n";
    llvm::errs() << "B: " << b.addressSpace.objects << "\n";
  }
    
  std::set<const MemoryObject*> mutated;
  MemoryMap::iterator ai = parentES->addressSpace.objects.begin();
  MemoryMap::iterator bi = b.addressSpace.objects.begin();
  MemoryMap::iterator ae = parentES->addressSpace.objects.end();
  MemoryMap::iterator be = b.addressSpace.objects.end();
  for (; ai!=ae && bi!=be; ++ai, ++bi) {
    if (ai->first != bi->first) {
      if (DebugLogStateMerge) {
        if (ai->first < bi->first) {
          llvm::errs() << "\t\tB misses binding for: " << ai->first->id << "\n";
        } else {
          llvm::errs() << "\t\tA misses binding for: " << bi->first->id << "\n";
        }
      }
#ifdef ENABLE_KLEE_SEMU_DEBUG
      llvm::errs() << "--> ai->first != bi->first\n";
#endif
      // They are certainly different. Insert true to show that
      inStateDiffExp.push_back(ConstantExpr::alloc(1, Expr::Bool));
      //return ksVARS_DIFF;
    } else {
    
      if (ai->first->isLocal && !checkLocals) {
        continue;
      }
      
      if (ai->second.get() != bi->second.get()
          && (*ai->second).read(0, ai->first->size*8).compare((*bi->second).read(0, bi->first->size*8)) != 0 
#ifdef SEMU_RELMUT_PRED_ENABLED
          && ai->first->allocSite != IsOldVersionDeclIns
#endif
          && ai->first->allocSite != MutantIDSelectDeclIns) {
        ref<Expr> tmpexpr = NeExpr::create((*ai->second).read(0, ai->first->size*8), (*bi->second).read(0, bi->first->size*8));
        if (feasibleChecker->isFeasible(tmpexpr)) {
          if (DebugLogStateMerge)
            llvm::errs() << "\t\tmutated: " << ai->first->id << "\n";
          mutated.insert(ai->first);
          inStateDiffExp.push_back(tmpexpr);
        }
      }
    }
  }
  if (mutated.size() > 0) {
    //llvm::errs() << "Global objects mutated(" << mutated.size() << ")\n";
    returnedCode |= ksVARS_DIFF;
  }
  if (ai!=ae || bi!=be) {
    if (DebugLogStateMerge)
      llvm::errs() << "\t\tmappings differ\n";
#ifdef ENABLE_KLEE_SEMU_DEBUG
    llvm::errs() << "--> ai!=ae || bi!=be\n";
#endif
    // They are certainly different. Insert true to show that
    inStateDiffExp.push_back(ConstantExpr::alloc(1, Expr::Bool));
    returnedCode |= ksVARS_DIFF;
  }
  
  if (checkRegs) {
    std::vector<StackFrame>::iterator itA = parentES->stack.begin();
    std::vector<StackFrame>::const_iterator itB = b.stack.begin();
    for (; itA!=parentES->stack.end(); ++itA, ++itB) {
      StackFrame &af = *itA;
      const StackFrame &bf = *itB;
      for (unsigned i=0; i<af.kf->numRegisters; i++) {
        ref<Expr> &av = af.locals[i].value;
        const ref<Expr> &bv = bf.locals[i].value;
        if (av.isNull() || bv.isNull()) {
          //These ref=gisters wont be used later
        } else {
          if (av.compare(bv) != 0) {  
#ifdef ENABLE_KLEE_SEMU_DEBUG
            llvm::errs() << "--> Registers Differs\n";
#endif
            ref<Expr> tmpexpr = NeExpr::create(av, bv);
            if (feasibleChecker->isFeasible(tmpexpr)) {
              inStateDiffExp.push_back(tmpexpr);
              returnedCode |= ksVARS_DIFF;
            }
          }
        }
      }
    }
  }

  return returnedCode;
}
//~KS

void ExecutionStateHelperSemu::ks_stateBranchPostProcessing(ExecutionState *falseState) {
  // @KLEE-SEMu
  if (ks_mutantID == 0) {
    if (ks_getMode() == KS_Mode::SEMU_MODE) {
      assert(ks_curBranchTreeNode->lchild == 0 
              && ks_curBranchTreeNode->rchild == 0 
              && ks_curBranchTreeNode->exState == this 
              && "Left child and right child should be NULL here");
      ks_curBranchTreeNode->lchild = new KS_OrigBranchTreeNode(ks_curBranchTreeNode, ks_curBranchTreeNode->exState);
      ks_curBranchTreeNode->rchild = new KS_OrigBranchTreeNode(ks_curBranchTreeNode, falseState);
      falseState->semuESHelper.ks_curBranchTreeNode = ks_curBranchTreeNode->rchild;
      ks_curBranchTreeNode->exState = nullptr;
      ks_curBranchTreeNode = ks_curBranchTreeNode->lchild;
      
      //when branching original, copy this so that children know that what was already mutated
      falseState->semuESHelper.ks_VisitedMutPointsSet = ks_VisitedMutPointsSet;    
      falseState->semuESHelper.ks_VisitedMutantsSet = ks_VisitedMutantsSet;    
    } else {
      // KS_Mode::TESTGEN_MODE
      // just add falseState to children so that we can do 4 way fork with mutants
      ks_childrenStates.insert(falseState);
    }
  } else {
    if (ks_getMode() == KS_Mode::SEMU_MODE) {
      ks_childrenStates.insert(falseState);
    } else {
      // KS_Mode::TESTGEN_MODE
      falseState->semuESHelper.isTestGenMutSeeding = isTestGenMutSeeding;
      // if not seeding no need to add child, since normal symbex of the mutant
      if (isTestGenMutSeeding)
        ks_childrenStates.insert(falseState);
    }
  }
  //~KS
}

ExecutionState *ExecutionStateHelperSemu::cloneParentES() {
  return new ExecutionState(*parentES);
}
