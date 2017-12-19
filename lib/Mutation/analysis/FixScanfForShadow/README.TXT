USAGE:  Include 'FixScanfForShadow/scanf.h' into source containing the functions 'scanf', 'fscanf'.

WHY: Shadow do not generate input with normal scanf, the reason being that there is condition on the input when reading integer (see atob.h).

----
With SEMU, compile the my headers into a llvm .bc module and link with the program bc under evaluation, so that semu execution will consider my libs
