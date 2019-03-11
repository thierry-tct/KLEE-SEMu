//===-- OptionCategories.h --------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

/*
 * This header defines the option categories used in KLEE.
 */

#ifndef KLEE_OPTIONCATEGORIES_H
#define KLEE_OPTIONCATEGORIES_H

#include "llvm/Support/CommandLine.h"

namespace klee {
  extern llvm::cl::OptionCategory SolvingCat;
  extern llvm::cl::OptionCategory MergeCat;
}

#endif