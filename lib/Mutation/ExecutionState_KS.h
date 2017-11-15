//===-- ExecutionState.h ----------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_EXECUTIONSTATE_H
#define KLEE_EXECUTIONSTATE_H

#include "klee/Constraints.h"
#include "klee/Expr.h"
#include "klee/Internal/ADT/TreeStream.h"

// @KLEE-SEMu
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/SmallPtrSet.h"
//~ KS

// FIXME: We do not want to be exposing these? :(
#include "../../lib/Core/AddressSpace.h"
#include "klee/Internal/Module/KInstIterator.h"

#include <map>
#include <set>
#include <vector>

namespace klee {
class Array;
class CallPathNode;
struct Cell;
struct KFunction;
struct KInstruction;
class MemoryObject;
class PTreeNode;
struct InstructionInfo;

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MemoryMap &mm);

struct StackFrame {
  KInstIterator caller;
  KFunction *kf;
  CallPathNode *callPathNode;

  std::vector<const MemoryObject *> allocas;
  Cell *locals;

  /// Minimum distance to an uncovered instruction once the function
  /// returns. This is not a good place for this but is used to
  /// quickly compute the context sensitive minimum distance to an
  /// uncovered instruction. This value is updated by the StatsTracker
  /// periodically.
  unsigned minDistToUncoveredOnReturn;

  // For vararg functions: arguments not passed via parameter are
  // stored (packed tightly) in a local (alloca) memory object. This
  // is setup to match the way the front-end generates vaarg code (it
  // does not pass vaarg through as expected). VACopy is lowered inside
  // of intrinsic lowering.
  MemoryObject *varargs;

  StackFrame(KInstIterator caller, KFunction *kf);
  StackFrame(const StackFrame &s);
  ~StackFrame();
};

/// @brief ExecutionState representing a path under exploration
class ExecutionState {
public:
  typedef std::vector<StackFrame> stack_ty;
  
// @KLEE-SEMu Fields KS
public:
  typedef unsigned KS_MutantIDType;
  
  enum KS_StateDiff_t {ksNO_DIFF=0, ksVARS_DIFF=1, ksPC_DIFF=2, ksSYMBOLICS_DIFF=3, ksOUTENV_DIFF=4, ksRETCODE_DIFF_MAINFUNC=5, ksRETCODE_DIFF_ENTRYFUNC=6, ksRETCODE_DIFF_OTHERFUNC=7};
  
  inline static bool ks_isCriticalDiff (KS_StateDiff_t sdiff)
  {
    return (sdiff >= ksPC_DIFF && sdiff <= ksRETCODE_DIFF_ENTRYFUNC);           //XXX: should SYMBOLICS and ENTRYFUNC be here?
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
  };
  
  //Version ID: 0 for original; 1, 2, ... for mutants
  KS_MutantIDType ks_mutantID;
  
  //The last returned value: Help compare states when the watch point is end of the function
  ref<Expr> ks_lastReturnedVal;
  
  //pointer to the original state Sp from where this state Sm was 
  //originated (PathCondition of Sp includes PathCondition of Sm)
  struct KS_OrigBranchTreeNode *ks_originalMutSisterStates;
  
  struct KS_OrigBranchTreeNode *ks_curBranchTreeNode;
  
  //TODO: use a union for ks_childrenStates and ks_VisitedMutPointSet, as they are exclusive: the first for mutants and the second for original
  
  //States with same mutants ID branched from this state : This is empty at creation (thus when resulting from branch)
  llvm::SmallPtrSet<ExecutionState *, 5> ks_childrenStates;
  
  // keep the mutation switch statement that has been executed previously
  // by this state or its parents. (help to know whether branch mutants or not)
  llvm::SmallPtrSet<llvm::Instruction *, 5> ks_VisitedMutPointsSet;         //CallInst
  
  ExecutionState *ks_branchMut();
  
  enum KS_StateDiff_t ks_compareStateWith (const ExecutionState &b, llvm::Value *MutantIDSelectDeclIns, ref<Expr> &inStateDiffExp, bool checkRegs=false);
  
//~KS

private:
  // unsupported, use copy constructor
  ExecutionState &operator=(const ExecutionState &);

  std::map<std::string, std::string> fnAliases;

public:
  // Execution - Control Flow specific

  /// @brief Pointer to instruction to be executed after the current
  /// instruction
  KInstIterator pc;

  /// @brief Pointer to instruction which is currently executed
  KInstIterator prevPC;

  /// @brief Stack representing the current instruction stream
  stack_ty stack;

  /// @brief Remember from which Basic Block control flow arrived
  /// (i.e. to select the right phi values)
  unsigned incomingBBIndex;

  // Overall state of the state - Data specific

  /// @brief Address space used by this state (e.g. Global and Heap)
  AddressSpace addressSpace;

  /// @brief Constraints collected so far
  ConstraintManager constraints;

  /// Statistics and information

  /// @brief Costs for all queries issued for this state, in seconds
  mutable double queryCost;

  /// @brief Weight assigned for importance of this state.  Can be
  /// used for searchers to decide what paths to explore
  double weight;

  /// @brief Exploration depth, i.e., number of times KLEE branched for this state
  unsigned depth;

  /// @brief History of complete path: represents branches taken to
  /// reach/create this state (both concrete and symbolic)
  TreeOStream pathOS;

  /// @brief History of symbolic path: represents symbolic branches
  /// taken to reach/create this state
  TreeOStream symPathOS;

  /// @brief Counts how many instructions were executed since the last new
  /// instruction was covered.
  unsigned instsSinceCovNew;

  /// @brief Whether a new instruction was covered in this state
  bool coveredNew;

  /// @brief Disables forking for this state. Set by user code
  bool forkDisabled;

  /// @brief Set containing which lines in which files are covered by this state
  std::map<const std::string *, std::set<unsigned> > coveredLines;

  /// @brief Pointer to the process tree of the current state
  PTreeNode *ptreeNode;

  /// @brief Ordered list of symbolics: used to generate test cases.
  //
  // FIXME: Move to a shared list structure (not critical).
  std::vector<std::pair<const MemoryObject *, const Array *> > symbolics;

  /// @brief Set of used array names for this state.  Used to avoid collisions.
  std::set<std::string> arrayNames;

  std::string getFnAlias(std::string fn);
  void addFnAlias(std::string old_fn, std::string new_fn);
  void removeFnAlias(std::string fn);

private:
  ExecutionState() : ptreeNode(0) {}

public:
  ExecutionState(KFunction *kf);

  // XXX total hack, just used to make a state so solver can
  // use on structure
  ExecutionState(const std::vector<ref<Expr> > &assumptions);

  ExecutionState(const ExecutionState &state);

  ~ExecutionState();

  ExecutionState *branch();

  void pushFrame(KInstIterator caller, KFunction *kf);
  void popFrame();

  void addSymbolic(const MemoryObject *mo, const Array *array);
  void addConstraint(ref<Expr> e) { constraints.addConstraint(e); }

  bool merge(const ExecutionState &b);
  void dumpStack(llvm::raw_ostream &out) const;
};
}

#endif
