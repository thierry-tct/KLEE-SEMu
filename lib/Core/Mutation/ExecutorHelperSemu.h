#ifndef EXECUTOR_HELPER_SEMU_H
#define EXECUTOR_HELPER_SEMU_H

#include "klee/ExecutionState.h"
#include "klee/Interpreter.h"
#include "ExecutionStateHelperSemu.h"

#include "llvm/Analysis/CallGraph.h"

#include <unistd.h>
#include <sys/wait.h>

#ifdef ENABLE_Z3 
#include "KS_Z3MaxSat.h"
#endif

namespace klee /*ks*/ {
class Executor;

class ExecutorHelperSemu {
private:
  friend class Executor;
  Executor *parentE;

  // configs 
  int cfgSemuPreconditionLength;
  bool cfgSemuUseOnlyBranchForDepth;
  bool cfgSemuForkProcessForSegvExternalCalls;

  unsigned MaxForks;

public:
  ExecutorHelperSemu(Executor *parent, unsigned maxForksArg);

/*****************************************************************/
/****************** SEMu Only Elements @ START *******************/
/*****************************************************************/

  // @KLEE-SEMu
public:
  class KScheckFeasible: public ExecutionStateHelperSemu::KScheckFeasibleBase {
  private:
    ExecutorHelperSemu *executorHelper;
    ExecutionState *mutState;
    ref<Expr> originalCommonPrefix;
    bool usethesolver;
  public:
    KScheckFeasible(ExecutorHelperSemu *exH, ExecutionState *ms, ref<Expr> origPref, bool usesolver): 
      executorHelper(exH), 
      mutState(ms), 
      originalCommonPrefix(origPref),
      usethesolver(usesolver) {}
    bool isFeasible(ref<Expr> bool_expr) {
      if (!usethesolver || bool_expr->isTrue()) 
        return true; 
      return executorHelper->ks_checkfeasiblewithsolver(*mutState, 
                                                  AndExpr::create(originalCommonPrefix, 
                                                                  bool_expr));
    }
  };

  bool ks_checkfeasiblewithsolver(ExecutionState &state, ref<Expr> bool_expr);

private:
  const char *ks_mutantIDSelectorName = "klee_semu_GenMu_Mutant_ID_Selector";
  const char *ks_mutantIDSelectorName_Func = "klee_semu_GenMu_Mutant_ID_Selector_Func";
  const char *ks_postMutationPointFuncName = "klee_semu_GenMu_Post_Mutation_Point_Func";
  llvm::GlobalVariable *ks_mutantIDSelectorGlobal;
  llvm::Function *ks_mutantIDSelectorGlobal_Func;
  llvm::Function *ks_postMutationPoint_Func;
  llvm::SmallPtrSet<ExecutionState *, 5> ks_reachedWatchPoint;
  llvm::SmallPtrSet<ExecutionState *, 5> ks_terminatedBeforeWP;
  llvm::SmallPtrSet<ExecutionState *, 5> ks_reachedOutEnv;
  // The mutant and original states that just got to mutation point and passed
  llvm::SmallPtrSet<ExecutionState *, 5> ks_atPointPostMutation;
  // The states that will still execute after the current WP
  llvm::SmallPtrSet<ExecutionState *, 5> ks_ongoingExecutionAtWP;

  llvm::SmallPtrSet<ExecutionState *, 5> ks_justTerminatedStates;

  llvm::Function * ks_entryFunction;
  
  //Use to decide to skip watch point (to avoid infinite loop): 
  // true <=> watch the point, false <=> do not watch the point (TODO: temporary)
  //bool ks_watchpoint;

  unsigned long ks_checkID=0;
  unsigned long ks_nextDepthID=1;

  double ks_runStartTime;

  bool ks_outputTestsCases;
  bool ks_outputStateDiffs;

  // Some stats
  unsigned long ks_numberOfMutantStatesDiscardedAtMutationPoint=0;
  unsigned long ks_numberOfMutantStatesCheckedAtMutationPoint=0;
  double ks_totalStateComparisonTime=0.0;

  //unsigned maxNumTestsPerMutants = 0;
  // Set of mutants that reached the maximum number of generated tests per mutants
  std::map<ExecutionStateHelperSemu::KS_MutantIDType, unsigned> mutants2gentestsNum;

  // Used by strategy that prioritize the mutants states closer to the output
  static std::map<llvm::Instruction*, unsigned> ks_instruction2closestout_distance;

  // will be set to the maximum mutant id when FilterMutants
  ExecutionStateHelperSemu::KS_MutantIDType ks_max_mutant_id=0;
  ExecutionStateHelperSemu::KS_MutantIDType ks_number_of_mutants=0;
  
  // Specify custom function as output env
  std::set<std::string> ks_customOutEnvFuncNames;

#ifdef KS_Z3MAXSAT_SOLVER__H
  // Partial Max Sat Solver
  // Make this with cache
  PartialMaxSATSolver pmaxsat_solver;
#endif //~KS_Z3MAXSAT_SOLVER__H

#ifdef SEMU_RELMUT_PRED_ENABLED
  unsigned long long ks_current_oldnew_split_id = 0;
  const char *ks_isOldVersionName = "klee_semu_GenMu_Is_Old_Version_Bool";
  const char *ks_klee_change_funtion_Name = "klee_change";
  llvm::GlobalVariable *ks_isOldVersionGlobal;
  llvm::Function *ks_klee_change_function;
#endif

public:
  /// Create new states where each constraint is that of the input state
  /// and return the results. The input state is *NOT* included
  /// in the results. Each state in the result correspond to a mutant.
  /// Note that the output vector may include
  /// NULL pointers for states which were unable to be created.
  void ks_mutationPointBranching(ExecutionState &state, 
              std::vector<uint64_t> &mut_IDs);
  
  bool ks_nextIsOutEnv (ExecutionState &state);
  bool ks_reachedAMutant(KInstruction *ki);
  bool ks_reachedAnOldNew(KInstruction *ki);
  bool ks_checkAtPostMutationPoint(ExecutionState &state, KInstruction *ki);
  inline bool ks_reachedCheckNextDepth(ExecutionState &state);
  bool ks_reachedCheckMaxDepth(ExecutionState &state);

  // Check whether we the last instruction forked - in seed Mode Test Generation (TG)
  bool ks_hasJustForkedTG (ExecutionState &state, KInstruction *ki);

  // make shadow's four way fork to execute the mutants with bounded symbex (for TG)
  void ks_fourWayForksTG();

  // Test wheather we reached the point to compare the states
  bool ks_watchPointReached (ExecutionState &state, KInstruction *ki);
  
  inline void ks_fixTerminatedChildren(ExecutionState *pes, llvm::SmallPtrSet<ExecutionState *, 5> const &toremove, 
				             bool terminateLooseOrig=false, bool removeLooseOrigFromAddedPreTerm=false); 
  void ks_fixTerminatedChildrenRecursive (ExecutionState *pes, llvm::SmallPtrSet<ExecutionState *, 5> const &toremove); 
  void ks_moveIntoTerminatedBeforeWP(ExecutionState *es); 
  void ks_terminateSubtreeMutants(ExecutionState *pes); 
  void ks_getMutParentStates(std::vector<ExecutionState *> &mutParentStates);
  void ks_compareStates (std::vector<ExecutionState *> &remainStates, bool outEnvOnly=false, bool postMutOnly=false);
  bool ks_diffTwoStates (ExecutionState *mState, ExecutionState *mSisState, 
                                  ref<Expr> &origSuffConstr, 
                                  bool outEnvOnly, int &sDiff, 
                                  std::vector<ref<Expr>> &inStateDiffExp);
  void ks_createDiffExpr(ExecutionState *mState, ref<Expr> &insdiff, 
                                  ref<Expr> &origpathPrefix,
                                  std::vector<ref<Expr>> &inStateDiffExp);
  bool ks_compareRecursive (ExecutionState *mState, std::vector<ExecutionState *> &mSisStatesVect, 
                          std::map<ExecutionState *, ref<Expr>> &origSuffConstr, bool outEnvOnly, 
                          bool postMutOnly, llvm::SmallPtrSet<ExecutionState *, 5> &postMutOnly_hasdiff);
  
  void ks_FilterMutants (llvm::Module *module);

  void ks_setInitialSymbolics (llvm::Module &module, llvm::Function &Func);
  
  inline llvm::Instruction * ks_makeArgSym (llvm::Module &module, llvm::GlobalVariable * &emptyStrAddr, llvm::Instruction *insAfter, llvm::Value *memAddr, llvm::Type *valtype);
  
  bool ks_outEnvCallDiff (const ExecutionState &a, const ExecutionState &b, std::vector<ref<Expr>> &inStateDiffExp, KScheckFeasible &feasibleChecker);
  
  bool ks_isOutEnvCallInvoke (llvm::Instruction *ci);

  // This take the path condition common to a mutant and original, together 
  // with the conditions of equality, for each state variable, between
  // original and mutant
  void ks_checkMaxSat (ConstraintManager const &mutPathCond,
                       ExecutionState const *origState,
                       std::vector<ref<Expr>> &stateDiffExprs,
                       ExecutionStateHelperSemu::KS_MutantIDType mutant_id, 
                       int sDiff); 
  void ks_writeMutantStateData(ExecutionStateHelperSemu::KS_MutantIDType mutant_id,
                                unsigned nSoftClauses,
                                unsigned nMaxFeasibleDiffs,
                                unsigned nMaxFeasibleEqs,
                                int sDiff,
                                ExecutionState const *origState);  

  bool ks_writeMutantTestsInfos(ExecutionStateHelperSemu::KS_MutantIDType mutant_id, unsigned testid); 

  void ks_loadKQueryConstraints(std::vector<ConstraintManager> &outConstraintsList);

  void ks_CheckAndBreakInfinitLoop(ExecutionState &curState, ExecutionState *&prevState, double &initTime);

  bool ks_CheckpointingMainCheck(ExecutionState &curState, KInstruction *ki, bool isSeeding, uint64_t precond_offset=0);

  void ks_heuristicbasedContinueStates (std::vector<ExecutionState*> const &statelist,
                                std::vector<ExecutionState*> &toContinue,
                                std::vector<ExecutionState*> &toStop);

  void ks_process_closestout_recursive(llvm::CallGraphNode *cgnode,
                              std::map<llvm::CallGraphNode*, unsigned> &visited_cgnodes);

  void ks_initialize_ks_instruction2closestout_distance(llvm::Module *mod);

  static bool ks_getMinDistToOutput(ExecutionState *lh, ExecutionState *rh);

  void ks_applyMutantSearchStrategy();

  void ks_eliminateMutantStatesWithMaxTests(bool pre_compare=false);

  bool ks_lazyInitialize (ExecutionState &state, KInstruction *ki);

  void ks_initializeModule(llvm::Module *module, const Interpreter::ModuleOptions &opts);

  void ks_verifyModule(llvm::Module *module);

#ifdef SEMU_RELMUT_PRED_ENABLED
  inline unsigned long long ks_get_next_oldnew_split_id();
  void ks_oldNewBranching(ExecutionState &state); 
  void ks_odlNewPrepareModule (llvm::Module *mod);
  bool ks_mutantPrePostDiff (ExecutionState *mState, bool outEnvOnly, 
                                      std::vector<ref<Expr>> &inStateDiffExp, 
                                      ref<Expr> &insdiff);
  //DONE TCT: 
  // X. Implement compare states
  // - add 'use_klee_change' PL arguments to control precondition length
  // - reach checkpoint for mutant post commit version only when both mutant and change are seen
  // - adapt the comparison to seach others and compare accordingly and use the condition
#endif
  //~KS

/*****************************************************************/
/*~~~~~~~~~~~~~~~~~ SEMu Only Elements @ END ~~~~~~~~~~~~~~~~~~~~*/
/*****************************************************************/
};
}

#endif // EXECUTOR_HELPER_SEMU_H
