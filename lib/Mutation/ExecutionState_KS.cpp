//===-- ExecutionState.cpp ------------------------------------------------===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ExecutionState_KS.h"

#include "klee/Internal/Module/Cell.h"
#include "klee/Internal/Module/InstructionInfoTable.h"
#include "klee/Internal/Module/KInstruction.h"
#include "klee/Internal/Module/KModule.h"

#include "klee/Expr.h"

// @KLEE-SEMu
#include "../Core/Memory.h"
//~KS

#if LLVM_VERSION_CODE >= LLVM_VERSION(3, 3)
#include "llvm/IR/Function.h"
#else
#include "llvm/Function.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <iomanip>
#include <sstream>
#include <cassert>
#include <map>
#include <set>
#include <stdarg.h>

using namespace llvm;
using namespace klee;

namespace { 
  cl::opt<bool>
  DebugLogStateMerge("debug-log-state-merge");
}

/***/

StackFrame::StackFrame(KInstIterator _caller, KFunction *_kf)
  : caller(_caller), kf(_kf), callPathNode(0), 
    minDistToUncoveredOnReturn(0), varargs(0) {
  locals = new Cell[kf->numRegisters];
}

StackFrame::StackFrame(const StackFrame &s) 
  : caller(s.caller),
    kf(s.kf),
    callPathNode(s.callPathNode),
    allocas(s.allocas),
    minDistToUncoveredOnReturn(s.minDistToUncoveredOnReturn),
    varargs(s.varargs) {
  locals = new Cell[s.kf->numRegisters];
  for (unsigned i=0; i<s.kf->numRegisters; i++)
    locals[i] = s.locals[i];
}

StackFrame::~StackFrame() { 
  delete[] locals; 
}

/***/

ExecutionState::ExecutionState(KFunction *kf) :
    pc(kf->instructions),
    prevPC(pc),

    queryCost(0.), 
    weight(1),
    depth(0),
    
    // @KLEE-SEMu
    ks_mutantID(0),
    ks_originalMutSisterStates(nullptr),
    ks_curBranchTreeNode(new KS_OrigBranchTreeNode(nullptr, this)),
    //~KS

    instsSinceCovNew(0),
    coveredNew(false),
    forkDisabled(false),
    ptreeNode(0) {
  pushFrame(0, kf);
}

ExecutionState::ExecutionState(const std::vector<ref<Expr> > &assumptions)
    : constraints(assumptions), queryCost(0.), ptreeNode(0), 
      // @KLEE-SEMu
      ks_mutantID(0), ks_originalMutSisterStates(nullptr), ks_curBranchTreeNode(nullptr) /*//~KS*/ {}

ExecutionState::~ExecutionState() {
  for (unsigned int i=0; i<symbolics.size(); i++)
  {
    const MemoryObject *mo = symbolics[i].first;
    assert(mo->refCount > 0);
    mo->refCount--;
    if (mo->refCount == 0)
      delete mo;
  }

  while (!stack.empty()) popFrame();
}

ExecutionState::ExecutionState(const ExecutionState& state):
    fnAliases(state.fnAliases),
    pc(state.pc),
    prevPC(state.prevPC),
    stack(state.stack),
    incomingBBIndex(state.incomingBBIndex),

    addressSpace(state.addressSpace),
    constraints(state.constraints),

    queryCost(state.queryCost),
    weight(state.weight),
    depth(state.depth),

    pathOS(state.pathOS),
    symPathOS(state.symPathOS),

    // @KLEE-SEMu
    ks_mutantID(state.ks_mutantID),
    ks_originalMutSisterStates(nullptr),
    ks_curBranchTreeNode(nullptr),
    //~KS
    
    instsSinceCovNew(state.instsSinceCovNew),
    coveredNew(state.coveredNew),
    forkDisabled(state.forkDisabled),
    coveredLines(state.coveredLines),
    ptreeNode(state.ptreeNode),
    symbolics(state.symbolics),
    arrayNames(state.arrayNames)
{
  for (unsigned int i=0; i<symbolics.size(); i++)
    symbolics[i].first->refCount++;
}

// @KLEE-SEMu
ExecutionState *ExecutionState::ks_branchMut() {
  depth++;

  ExecutionState *falseState = new ExecutionState(*this);
  falseState->coveredNew = false;
  falseState->coveredLines.clear();

  weight *= .5;
  falseState->weight -= weight;

  return falseState;
}

int ExecutionState::ks_compareStateWith (const ExecutionState &b, llvm::Value *MutantIDSelectDeclIns, std::vector<ref<Expr>> &inStateDiffExp, bool postExec, bool checkRegs/*=false*/) {
  //if (pc != b.pc)     //Commented beacause some states may terminate early but should still be considered(they may have different PC)
  //  return false;
  
  bool checkLocals = true;

  // The returned code
  int returnedCode = ksNO_DIFF;
  
  KInstIterator aPC, bPC;
  if (postExec) {
    aPC = prevPC;
    bPC = b.prevPC;
  } else {
    aPC = pc;
    bPC = b.pc;
  }

  //TODO>
  //> First: They should have same PC (The both executed the watch point)
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
    std::vector<StackFrame>::const_iterator itA = stack.begin();
    std::vector<StackFrame>::const_iterator itB = b.stack.begin();
    while (itA!=stack.end() && itB!=b.stack.end()) {
      // XXX vaargs?
      if (itA->caller != itB->caller || itA->kf != itB->kf) {
#ifdef ENABLE_KLEE_SEMU_DEBUG
        llvm::errs() << "--> itA->caller!=itB->caller || itA->kf!=itB->kf\n";
#endif
        llvm::errs() << "# A's Stack:";
        dumpStack(llvm::errs()); 
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
    if (itA!=stack.end() || itB!=b.stack.end()) {
#ifdef ENABLE_KLEE_SEMU_DEBUG
      llvm::errs() << "--> itA!=stack.end() || itB!=b.stack.end()\n";
#endif
      llvm::errs() << "# A's Stack:";
      dumpStack(llvm::errs()); 
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
      int noEntry_NegEntry0_1;
      if (stack.size() == 2)
          noEntry_NegEntry0_1 = (stack.at(1).kf->function->getName() == "__uClibc_main")? 1: -1;
      else
          noEntry_NegEntry0_1 = (! stack.back().caller)? 0: -1;
      
      if (noEntry_NegEntry0_1 >= 0) { //Entry point Function
        if (ri->getParent()->getParent()->getName() == mainFName[noEntry_NegEntry0_1]) { //Check only return values
          if (ks_lastReturnedVal.compare(b.ks_lastReturnedVal)) {
  #ifdef ENABLE_KLEE_SEMU_DEBUG
            llvm::errs() << "--> return Codes differ: main function.\n";
  #endif
            inStateDiffExp.push_back(NeExpr::create(ks_lastReturnedVal, b.ks_lastReturnedVal));
            returnedCode |= ksRETCODE_DIFF_MAINFUNC;
          }
          else {
            //return ksNO_DIFF;
          }
        } else {  //Check return val and globals
          if (ks_lastReturnedVal.compare(b.ks_lastReturnedVal)) {
  #ifdef ENABLE_KLEE_SEMU_DEBUG
            llvm::errs() << "--> return Codes differ: other entry point function.\n";
  #endif
            inStateDiffExp.push_back(NeExpr::create(ks_lastReturnedVal, b.ks_lastReturnedVal));
            returnedCode |= ksRETCODE_DIFF_ENTRYFUNC;
          }
          //checkLocals  = false;
        }
      } else {  //Check both returned val and Globals and locals
        if (ks_lastReturnedVal.compare(b.ks_lastReturnedVal)) {
  #ifdef ENABLE_KLEE_SEMU_DEBUG
          llvm::errs() << "--> return Codes differ: non entry point function.\n";
  #endif
          inStateDiffExp.push_back(NeExpr::create(ks_lastReturnedVal, b.ks_lastReturnedVal));
          returnedCode |= ksRETCODE_DIFF_OTHERFUNC;
        }
        //checkLocals  = true;
      }
    } 
  }

  // XXX is it even possible for these to differ? does it matter? probably
  // implies difference in object states?
  // check this only when we also compare local vars
  if (checkLocals && symbolics!=b.symbolics) {
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
    llvm::errs() << "A: " << addressSpace.objects << "\n";
    llvm::errs() << "B: " << b.addressSpace.objects << "\n";
  }
    
  std::set<const MemoryObject*> mutated;
  MemoryMap::iterator ai = addressSpace.objects.begin();
  MemoryMap::iterator bi = b.addressSpace.objects.begin();
  MemoryMap::iterator ae = addressSpace.objects.end();
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
      
      if (ai->second != bi->second && (*ai->second).read(0, ai->first->size*8).compare((*bi->second).read(0, bi->first->size*8)) != 0 && ai->first->allocSite != MutantIDSelectDeclIns) {
        if (DebugLogStateMerge)
          llvm::errs() << "\t\tmutated: " << ai->first->id << "\n";
        mutated.insert(ai->first);
        inStateDiffExp.push_back(NeExpr::create((*ai->second).read(0, ai->first->size*8), (*bi->second).read(0, bi->first->size*8)));
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
    std::vector<StackFrame>::iterator itA = stack.begin();
    std::vector<StackFrame>::const_iterator itB = b.stack.begin();
    for (; itA!=stack.end(); ++itA, ++itB) {
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
            inStateDiffExp.push_back(NeExpr::create(av, bv));
            returnedCode |= ksVARS_DIFF;
          }
        }
      }
    }
  }

  return returnedCode;
}
//~KS

ExecutionState *ExecutionState::branch() {
  depth++;

  ExecutionState *falseState = new ExecutionState(*this);
  falseState->coveredNew = false;
  falseState->coveredLines.clear();

  weight *= .5;
  falseState->weight -= weight;
  
  // @KLEE-SEMu
  if (ks_mutantID == 0) {
    assert(ks_curBranchTreeNode->lchild == 0 && ks_curBranchTreeNode->rchild == 0 && ks_curBranchTreeNode->exState == this && "Left child and right child should be NULL here");
    ks_curBranchTreeNode->lchild = new KS_OrigBranchTreeNode(ks_curBranchTreeNode, ks_curBranchTreeNode->exState);
    ks_curBranchTreeNode->rchild = new KS_OrigBranchTreeNode(ks_curBranchTreeNode, falseState);
    falseState->ks_curBranchTreeNode = ks_curBranchTreeNode->rchild;
    ks_curBranchTreeNode->exState = nullptr;
    ks_curBranchTreeNode = ks_curBranchTreeNode->lchild;
    
    falseState->ks_VisitedMutPointsSet = ks_VisitedMutPointsSet;    //when branching original, copy this so that children no that what was already mutated
  } else {
    ks_childrenStates.insert(falseState);
  }
  //~KS

  return falseState;
}

void ExecutionState::pushFrame(KInstIterator caller, KFunction *kf) {
  stack.push_back(StackFrame(caller,kf));
}

void ExecutionState::popFrame() {
  StackFrame &sf = stack.back();
  for (std::vector<const MemoryObject*>::iterator it = sf.allocas.begin(), 
         ie = sf.allocas.end(); it != ie; ++it)
    addressSpace.unbindObject(*it);
  stack.pop_back();
}

void ExecutionState::addSymbolic(const MemoryObject *mo, const Array *array) { 
  mo->refCount++;
  symbolics.push_back(std::make_pair(mo, array));
}
///

std::string ExecutionState::getFnAlias(std::string fn) {
  std::map < std::string, std::string >::iterator it = fnAliases.find(fn);
  if (it != fnAliases.end())
    return it->second;
  else return "";
}

void ExecutionState::addFnAlias(std::string old_fn, std::string new_fn) {
  fnAliases[old_fn] = new_fn;
}

void ExecutionState::removeFnAlias(std::string fn) {
  fnAliases.erase(fn);
}

/**/

llvm::raw_ostream &klee::operator<<(llvm::raw_ostream &os, const MemoryMap &mm) {
  os << "{";
  MemoryMap::iterator it = mm.begin();
  MemoryMap::iterator ie = mm.end();
  if (it!=ie) {
    os << "MO" << it->first->id << ":" << it->second;
    for (++it; it!=ie; ++it)
      os << ", MO" << it->first->id << ":" << it->second;
  }
  os << "}";
  return os;
}

bool ExecutionState::merge(const ExecutionState &b) {
  if (DebugLogStateMerge)
    llvm::errs() << "-- attempting merge of A:" << this << " with B:" << &b
                 << "--\n";
  if (pc != b.pc)
    return false;

  // XXX is it even possible for these to differ? does it matter? probably
  // implies difference in object states?
  if (symbolics!=b.symbolics)
    return false;

  {
    std::vector<StackFrame>::const_iterator itA = stack.begin();
    std::vector<StackFrame>::const_iterator itB = b.stack.begin();
    while (itA!=stack.end() && itB!=b.stack.end()) {
      // XXX vaargs?
      if (itA->caller!=itB->caller || itA->kf!=itB->kf)
        return false;
      ++itA;
      ++itB;
    }
    if (itA!=stack.end() || itB!=b.stack.end())
      return false;
  }

  std::set< ref<Expr> > aConstraints(constraints.begin(), constraints.end());
  std::set< ref<Expr> > bConstraints(b.constraints.begin(), 
                                     b.constraints.end());
  std::set< ref<Expr> > commonConstraints, aSuffix, bSuffix;
  std::set_intersection(aConstraints.begin(), aConstraints.end(),
                        bConstraints.begin(), bConstraints.end(),
                        std::inserter(commonConstraints, commonConstraints.begin()));
  std::set_difference(aConstraints.begin(), aConstraints.end(),
                      commonConstraints.begin(), commonConstraints.end(),
                      std::inserter(aSuffix, aSuffix.end()));
  std::set_difference(bConstraints.begin(), bConstraints.end(),
                      commonConstraints.begin(), commonConstraints.end(),
                      std::inserter(bSuffix, bSuffix.end()));
  if (DebugLogStateMerge) {
    llvm::errs() << "\tconstraint prefix: [";
    for (std::set<ref<Expr> >::iterator it = commonConstraints.begin(),
                                        ie = commonConstraints.end();
         it != ie; ++it)
      llvm::errs() << *it << ", ";
    llvm::errs() << "]\n";
    llvm::errs() << "\tA suffix: [";
    for (std::set<ref<Expr> >::iterator it = aSuffix.begin(),
                                        ie = aSuffix.end();
         it != ie; ++it)
      llvm::errs() << *it << ", ";
    llvm::errs() << "]\n";
    llvm::errs() << "\tB suffix: [";
    for (std::set<ref<Expr> >::iterator it = bSuffix.begin(),
                                        ie = bSuffix.end();
         it != ie; ++it)
      llvm::errs() << *it << ", ";
    llvm::errs() << "]\n";
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
    llvm::errs() << "A: " << addressSpace.objects << "\n";
    llvm::errs() << "B: " << b.addressSpace.objects << "\n";
  }
    
  std::set<const MemoryObject*> mutated;
  MemoryMap::iterator ai = addressSpace.objects.begin();
  MemoryMap::iterator bi = b.addressSpace.objects.begin();
  MemoryMap::iterator ae = addressSpace.objects.end();
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
      return false;
    }
    if (ai->second != bi->second) {
      if (DebugLogStateMerge)
        llvm::errs() << "\t\tmutated: " << ai->first->id << "\n";
      mutated.insert(ai->first);
    }
  }
  if (ai!=ae || bi!=be) {
    if (DebugLogStateMerge)
      llvm::errs() << "\t\tmappings differ\n";
    return false;
  }
  
  // merge stack

  ref<Expr> inA = ConstantExpr::alloc(1, Expr::Bool);
  ref<Expr> inB = ConstantExpr::alloc(1, Expr::Bool);
  for (std::set< ref<Expr> >::iterator it = aSuffix.begin(), 
         ie = aSuffix.end(); it != ie; ++it)
    inA = AndExpr::create(inA, *it);
  for (std::set< ref<Expr> >::iterator it = bSuffix.begin(), 
         ie = bSuffix.end(); it != ie; ++it)
    inB = AndExpr::create(inB, *it);

  // XXX should we have a preference as to which predicate to use?
  // it seems like it can make a difference, even though logically
  // they must contradict each other and so inA => !inB

  std::vector<StackFrame>::iterator itA = stack.begin();
  std::vector<StackFrame>::const_iterator itB = b.stack.begin();
  for (; itA!=stack.end(); ++itA, ++itB) {
    StackFrame &af = *itA;
    const StackFrame &bf = *itB;
    for (unsigned i=0; i<af.kf->numRegisters; i++) {
      ref<Expr> &av = af.locals[i].value;
      const ref<Expr> &bv = bf.locals[i].value;
      if (av.isNull() || bv.isNull()) {
        // if one is null then by implication (we are at same pc)
        // we cannot reuse this local, so just ignore
      } else {
        av = SelectExpr::create(inA, av, bv);
      }
    }
  }

  for (std::set<const MemoryObject*>::iterator it = mutated.begin(), 
         ie = mutated.end(); it != ie; ++it) {
    const MemoryObject *mo = *it;
    const ObjectState *os = addressSpace.findObject(mo);
    const ObjectState *otherOS = b.addressSpace.findObject(mo);
    assert(os && !os->readOnly && 
           "objects mutated but not writable in merging state");
    assert(otherOS);

    ObjectState *wos = addressSpace.getWriteable(mo, os);
    for (unsigned i=0; i<mo->size; i++) {
      ref<Expr> av = wos->read8(i);
      ref<Expr> bv = otherOS->read8(i);
      wos->write(i, SelectExpr::create(inA, av, bv));
    }
  }

  constraints = ConstraintManager();
  for (std::set< ref<Expr> >::iterator it = commonConstraints.begin(), 
         ie = commonConstraints.end(); it != ie; ++it)
    constraints.addConstraint(*it);
  constraints.addConstraint(OrExpr::create(inA, inB));

  return true;
}

void ExecutionState::dumpStack(llvm::raw_ostream &out) const {
  unsigned idx = 0;
  const KInstruction *target = prevPC;
  for (ExecutionState::stack_ty::const_reverse_iterator
         it = stack.rbegin(), ie = stack.rend();
       it != ie; ++it) {
    const StackFrame &sf = *it;
    Function *f = sf.kf->function;
    const InstructionInfo &ii = *target->info;
    out << "\t#" << idx++;
    std::stringstream AssStream;
    AssStream << std::setw(8) << std::setfill('0') << ii.assemblyLine;
    out << AssStream.str();
    out << " in " << f->getName().str() << " (";
    // Yawn, we could go up and print varargs if we wanted to.
    unsigned index = 0;
    for (Function::arg_iterator ai = f->arg_begin(), ae = f->arg_end();
         ai != ae; ++ai) {
      if (ai!=f->arg_begin()) out << ", ";

      out << ai->getName().str();
      // XXX should go through function
      ref<Expr> value = sf.locals[sf.kf->getArgRegister(index++)].value;
      if (value.get() && isa<ConstantExpr>(value))
        out << "=" << value;
    }
    out << ")";
    if (ii.file != "")
      out << " at " << ii.file << ":" << ii.line;
    out << "\n";
    target = sf.caller;
  }
}
