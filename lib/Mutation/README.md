

## Problems and Solutions
1. __Writing *ktest* and mutant killing test data fails with system failure.__
    This is due to the `ulimit` of open files (check with `ulimit -a` command). change the limit by setting a higher limit with the command:
    ```bash
    ulimit -n <new value>
    ```
2. __Solver Fork failure Using STP solver.__
    Use Z3 solver instead
3. __Resource limit reach with Z3 solver.__
    This is due to lack of resource with Z3 (I guess). apply the following patch to just consider it as a solver timeout and bypass it:
    ```diff
    diff --git a/lib/Solver/Z3Builder.cpp b/lib/Solver/Z3Builder.cpp
    index fc826c0..6b945e7 100644
    --- a/lib/Solver/Z3Builder.cpp
    +++ b/lib/Solver/Z3Builder.cpp
    @@ -39,7 +39,7 @@ void custom_z3_error_handler(Z3_context ctx, Z3_error_code ec) {
     #endif
       // FIXME: This is kind of a hack. The value comes from the enum
       // Z3_CANCELED_MSG but this isn't currently exposed by Z3's C API
    -  if (strcmp(errorMsg, "canceled") == 0) {
    +  if (strcmp(errorMsg, "canceled") == 0 || strcmp(errorMsg, "(resource limits reached)") == 0) {
          // Solver timeout is not a fatal error
          return;
       }
    diff --git a/lib/Solver/Z3Solver.cpp b/lib/Solver/Z3Solver.cpp
    index 1cbca56..ef59f84 100644
    --- a/lib/Solver/Z3Solver.cpp
    +++ b/lib/Solver/Z3Solver.cpp
    @@ -280,7 +280,7 @@ SolverImpl::SolverRunStatus Z3SolverImpl::handleSolverResponse(
       case Z3_L_UNDEF: {
         ::Z3_string reason =
             ::Z3_solver_get_reason_unknown(builder->ctx, theSolver);
    -    if (strcmp(reason, "timeout") == 0 || strcmp(reason, "canceled") == 0) {
    +    if (strcmp(reason, "timeout") == 0 || strcmp(reason, "canceled") == 0 || strcmp(reason, "(resource limits reached)") == 0) {
           return SolverImpl::SOLVER_RUN_STATUS_TIMEOUT;
         }
         if (strcmp(reason, "unknown") == 0) {
    ```

4. 