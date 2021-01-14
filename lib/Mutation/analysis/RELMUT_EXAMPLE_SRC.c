#include <stdio.h>
#include <stdlib.h>


// Mutant ID selector global initialized to N+1 (N=4)
unsigned long  klee_semu_GenMu_Mutant_ID_Selector = 5;

// mutant ID selector function. SEMu forks mutant states 
// at calls to this
void klee_semu_GenMu_Mutant_ID_Selector_Func (unsigned long fromID, unsigned long toID);

// mutated code successor code. First code after mutant code.
// This is used by SEMu to do conservative pruning 
// of no infection
void klee_semu_GenMu_Post_Mutation_Point_Func (unsigned long fromID, unsigned long toID);

unsigned long long klee_change (unsigned long long x, unsigned long long y){return y;}

int main (int argc, char ** argv) {
    int tmp, y=-999999, x = atoi(argv[1]);
#ifdef CHANGE_BEFORE_MUTANT
    x = klee_change(x+1, x);
#endif
    if (x >= 0) {
        klee_semu_GenMu_Mutant_ID_Selector_Func(1,2);
        klee_semu_GenMu_Mutant_ID_Selector_Func(4,4);
        switch (klee_semu_GenMu_Mutant_ID_Selector) {
            case 1: y = x - 10; break;
            case 2: y = 100; break;
            case 4: break;
            default: y = x + 10;
        }
        klee_semu_GenMu_Post_Mutation_Point_Func(1,2);
        klee_semu_GenMu_Post_Mutation_Point_Func(4,4);
        printf ("Changed\n");
    }
    klee_semu_GenMu_Mutant_ID_Selector_Func(3,3);
    switch (klee_semu_GenMu_Mutant_ID_Selector) {
        case 3: x+=2;break;
        default: x+=1;
    }
    klee_semu_GenMu_Post_Mutation_Point_Func(3,3);

    printf ("DONE!\n");

#ifdef CHANGE_BEFORE_MUTANT
    tmp = y;
#else
    tmp = klee_change(y+1,y);
#endif

    return tmp;
}
