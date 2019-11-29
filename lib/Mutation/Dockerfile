### Run from the repo root dir
# docker build -t thierrytct/ks:llvm-3.4.2 -f lib/Mutation/Dockerfile .
# docker push thierrytct/ks:llvm-3.4.2

ARG mart_llvm_version=3.4.2

FROM thierrytct/mart:llvm-$mart_llvm_version

RUN apt-get -y update && apt-get -y install git \
  && apt-get -y install build-essential curl wget libcap-dev libncurses5-dev unzip libtcmalloc-minimal4 libgoogle-perftools-dev libsqlite3-dev doxygen \
  && pip3 install tabulate lit \
  && mkdir -p /home/klee/klee_src /home/klee/klee_build /home/klee/uclibc_solvers

# install uclibc, stp and z3
RUN git clone https://github.com/klee/klee-uclibc.git /home/klee/uclibc_solvers/klee-uclibc && cd /home/klee/uclibc_solvers/klee-uclibc && ./configure --make-llvm-lib && make -j2 \
  && apt-get -y install cmake bison flex libboost-all-dev perl minisat && git clone https://github.com/stp/stp /home/klee/uclibc_solvers/stp \
    && cd /home/klee/uclibc_solvers/stp && mkdir build && cd build && cmake .. && make && make install \
  && git clone https://github.com/Z3Prover/z3.git /home/klee/uclibc_solvers/z3 \
    && cd /home/klee/uclibc_solvers/z3 && mkdir build && cd build && cmake -G "Unix Makefiles" ../ && make -j4 && make install

COPY . /home/klee/klee_src

# build klee
RUN cd /home/klee/klee_build && cmake -DCMAKE_BUILD_TYPE=Release \
 -DENABLE_KLEE_ASSERTS=true \
 -DENABLE_KLEE_UCLIBC=true \
 -DENABLE_POSIX_RUNTIME=true \
 -DENABLE_SOLVER_METASMT=false \
 -DENABLE_SOLVER_STP=true \
 -DENABLE_SOLVER_Z3=true \
 -DZ3_INCLUDE_DIRS=/home/klee/uclibc_solvers/z3/install/include \
 -DZ3_LIBRARIES=/home/klee/uclibc_solvers/z3/install/lib/libz3.so \
 -DENABLE_TCMALLOC=true \
 -DENABLE_TESTS=true \
 -DENABLE_UNIT_TESTS=OFF \
 -DGTEST_SRC_DIR=../klee_src/unittests/ \
 -DKLEE_UCLIBC_PATH=/home/klee/uclibc_solvers/klee-uclibc \
 -DSTP_DIR=/home/klee/uclibc_solvers/stp/build \
 -DUSE_CXX11=true \
  ../klee_src

ENV PATH="/home/klee/klee_build/Release+Asserts/bin:${PATH}"


