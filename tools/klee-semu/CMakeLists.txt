#===------------------------------------------------------------------------===#
#
#                     The KLEE Symbolic Virtual Machine
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
add_executable(klee-semu
  klee-semu.cpp
)

set(KLEE_LIBS
  kleeSemuCore
)

target_link_libraries(klee-semu ${KLEE_LIBS})

install(TARGETS klee-semu RUNTIME DESTINATION bin)

# The KLEE binary depends on the runtimes
add_dependencies(klee-semu BuildKLEERuntimes)
