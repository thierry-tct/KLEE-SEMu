
#ifndef EXECUTION_STATE_HELPER_SEMU
#define EXECUTION_STATE_HELPER_SEMU

#include "klee/Expr.h"

// @KLEE-SEMu
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/SmallPtrSet.h"
//~ KS

namespace klee /*ks*/ {

class ExecutionState;

class ExecutionStateHelperSemu {
private:
  friend class ExecutionState;
  ExecutionState *parentES;  

  bool DebugLogStateMerge;

public:
  ExecutionStateHelperSemu(ExecutionState *parent, bool DebugLogStateMergeArg);
  ExecutionStateHelperSemu(ExecutionState *parent, const ExecutionStateHelperSemu &other, bool DebugLogStateMergeArg);

private:
  ExecutionStateHelperSemu(): parentES(0) {}

/*****************************************************************/
/****************** SEMu Only Elements @ START *******************/
/*****************************************************************/

// @KLEE-SEMu Fields KS
public:
  class KScheckFeasibleBase {
  public:
    virtual bool isFeasible(ref<Expr> bool_expr) = 0;
  };

  enum KS_Mode {
    SEMU_MODE=0x0, 
    TESTGEN_MODE=0x1
  };

  typedef unsigned KS_MutantIDType;
  
  enum KS_StateDiff_t {
    ksNO_DIFF=0x00, 
    ksVARS_DIFF=0x01, 
    ksRETCODE_DIFF_OTHERFUNC=0x02, 
    ksRETCODE_DIFF_ENTRYFUNC=0x04, 
    ksRETCODE_DIFF_MAINFUNC=0x08, 
    ksOUTENV_DIFF=0x10, 
    ksSYMBOLICS_DIFF=0x20, 
    ksPC_DIFF=0x40, 
    ksFAILURE_BUG=0x80/*A bug in program*/
  };
  
  inline static bool ks_isCriticalDiff (int sdiff)
  {
    return (sdiff >= ksRETCODE_DIFF_ENTRYFUNC && sdiff <= ksPC_DIFF);           //XXX: should SYMBOLICS and ENTRYFUNC be here?
  }
  inline static bool ks_isNoDiff (int sdiff)
  {
    return (sdiff == ksNO_DIFF); 
  }
  
  inline static void ks_checkNoDiffError(int sdiff, KS_MutantIDType mutant_id)
  {
    if (sdiff >= ksFAILURE_BUG) {
      llvm::errs() << "\nSEMU@ERROR: error in semu execution - Mutant ID: " << mutant_id << ", error code: 1\n";
      exit(1); //ERROR
    }
  }

  struct KS_OrigBranchTreeNode {
    struct KS_OrigBranchTreeNode * parent;
    struct KS_OrigBranchTreeNode * lchild;
    struct KS_OrigBranchTreeNode * rchild;
    ExecutionState * exState;
    KS_OrigBranchTreeNode(struct KS_OrigBranchTreeNode *p, ExecutionState *e): parent(p), lchild(0), rchild(0), exState(e){}
    ~KS_OrigBranchTreeNode() {}    
    void getAllStates(std::vector<ExecutionState *> &vect) { //DFS
      if (exState) {
        vect.push_back(exState);
        assert (lchild == 0 && rchild == 0 && "lchild and rchild should be null here");
      }
      else {
        if (lchild)
          lchild->getAllStates(vect);
        if (rchild)
          rchild->getAllStates(vect);
      }
    }
    // TODO: implements this and call it. return true if fine (not terminated)
    bool ks_cleanTerminatedOriginals(llvm::SmallPtrSet<ExecutionState *, 5> const &ks_terminatedBeforeWP) {
      if (exState && ks_terminatedBeforeWP.count(exState) > 0)
        exState = nullptr;
      if (lchild) {
        if (lchild->exState) {
          if (ks_terminatedBeforeWP.count(lchild->exState) > 0) {
            //delete lchild;
            //lchild = nullptr;
            lchild->exState = nullptr;
          }  
        } else {
          lchild->ks_cleanTerminatedOriginals(ks_terminatedBeforeWP);
        }
      }
      if (rchild) {
        if (rchild->exState) {
          if (ks_terminatedBeforeWP.count(rchild->exState) > 0) {
            //delete rchild;
            //rchild = nullptr;
            rchild->exState = nullptr;
          }
        } else {
          rchild->ks_cleanTerminatedOriginals(ks_terminatedBeforeWP);
        }
      }
      return true;
    }
  };
  
  // Mode of execution
  static KS_Mode ks_mode;
  static inline void ks_setMode(KS_Mode mode) {ks_mode = mode;}
  static inline KS_Mode ks_getMode() {return ks_mode;}

  //Version ID: 0 for original; 1, 2, ... for mutants
  KS_MutantIDType ks_mutantID;
 
  //The last returned value: Help compare states when the watch point is end of the function
  ref<Expr> ks_lastReturnedVal;

  // On the test generation mode (Shadow based), is the mutant state seeding(shadowing originale)
  // Or executing bounded symbolic execution. True for seeding, false for bounded symbex
  // XXX changed in functions branch() and ks_branchMut()'s caller
  bool isTestGenMutSeeding;
  
  // pointer to the original state Sp from where this state Sm was 
  // originated (PathCondition of Sp includes PathCondition of Sm)
  struct KS_OrigBranchTreeNode *ks_originalMutSisterStates;
  
  struct KS_OrigBranchTreeNode *ks_curBranchTreeNode;
  
  //States with same mutants ID branched from this state : This is empty at creation (thus when resulting from branch)
  llvm::SmallPtrSet<ExecutionState *, 5> ks_childrenStates;
  
  // keep the mutation switch statement that has been executed previously
  // by this state or its parents. (help to know whether branch mutants or not)
  llvm::SmallPtrSet<llvm::Instruction *, 5> ks_VisitedMutPointsSet;         //CallInst
  std::set<KS_MutantIDType> ks_VisitedMutantsSet;

  // For the original program: Store the number of mutants That compare with this original state
  KS_MutantIDType ks_numberActiveCmpMutants = 0;

  // Say whether the postmutationpoint need to be reached for this state
  // initially true for mutant state, 
  // but change as mutants are found (for original)
  bool ks_hasToReachPostMutationPoint = false;

  unsigned ks_startdepth;

  // Useful for cases where original reach outenv but not the mutant
  unsigned ks_numberOfOutEnvSeen = 0;

  unsigned ks_numSeenCheckpoints = 0;
  
  ExecutionState *ks_branchMut();

  void ks_stateBranchPostProcessing(ExecutionState *falseState);
  
  // This function help to know whetehr the function that is returning is 
  // an entry point function (return 0 if 'main' and 1 if '__user_main') 
  // or not(return negative number)
  int ks_checkRetFunctionEntry01NonEntryNeg();

  bool ks_stackHasAnyFunctionOf(std::set<std::string> &funcnames);

  // Post exec say whether the comparison is done after the chekpoint instruction execution
  int ks_compareStateWith (const ExecutionState &b, llvm::Value *MutantIDSelectDeclIns, 
#ifdef SEMU_RELMUT_PRED_ENABLED
                                               llvm::Value *IsOldVersionDeclIns,
#endif
                           std::vector<ref<Expr>> &inStateDiffExp, KScheckFeasibleBase *feasibleChecker, 
                           bool postExec=true, bool checkRegs=false);
  
#ifdef SEMU_RELMUT_PRED_ENABLED
  // record the version represented by the state
  // 0 means common to both old and new, -1 means old, 1 means new
  char ks_old_new = 0;
  
  // store the id of the point where the version split occued for this state 
  unsigned long long ks_oldnew_split_id;
#endif
//~KS

/*****************************************************************/
/*~~~~~~~~~~~~~~~~~ SEMu Only Elements @ END ~~~~~~~~~~~~~~~~~~~~*/
/*****************************************************************/
};
}

#endif // EXECUTION_STATE_HELPER_SEMU
