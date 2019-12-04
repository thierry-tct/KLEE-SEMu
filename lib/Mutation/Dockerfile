### Run from the repo root dir
# docker build -t thierrytct/ks:llvm-3.4.2 -f lib/Mutation/Dockerfile .
# docker push thierrytct/ks:llvm-3.4.2

ARG mart_llvm_version=3.4.2

FROM thierrytct/mart:llvm-$mart_llvm_version

ARG mart_llvm_version=3.4.2

RUN apt-get -y update && apt-get -y install git vim \
  && apt-get -y install build-essential curl wget libcap-dev libncurses5-dev unzip libtcmalloc-minimal4 libgoogle-perftools-dev libsqlite3-dev doxygen \
  && pip3 install tabulate lit \
  && mkdir -p /home/klee-semu/klee_src /home/klee-semu/klee_build /home/klee-semu/uclibc_solvers

# install uclibc, stp and z3
RUN git clone https://github.com/klee/klee-uclibc.git /home/klee-semu/uclibc_solvers/klee-uclibc && cd /home/klee-semu/uclibc_solvers/klee-uclibc && ./configure --make-llvm-lib && make -j2 \
  && apt-get -y install cmake bison flex libboost-all-dev perl minisat && git clone https://github.com/stp/stp /home/klee-semu/uclibc_solvers/stp \
    && cd /home/klee-semu/uclibc_solvers/stp && mkdir build && cd build && cmake .. && make && make install \
  && git clone https://github.com/Z3Prover/z3.git /home/klee-semu/uclibc_solvers/z3 \
    && cd /home/klee-semu/uclibc_solvers/z3 && mkdir build && cd build && cmake -G "Unix Makefiles" ../ && make -j4 && make install

COPY . /home/klee-semu/klee_src

# build klee-semu
#RUN cd /home/klee-semu/klee_build \
#  && CXXFLAGS='-g' /home/klee-semu/klee_src/configure \
#  --with-llvmsrc=/home/LLVM/llvm-${mart_llvm_version}/src \
#  --with-llvmobj=/home/LLVM/llvm-${mart_llvm_version}/build_configure \
#  --with-llvmcc=/home/LLVM/llvm-${mart_llvm_version}/build_configure/Debug+Asserts/bin/clang \
#  --with-llvmcxx=/home/LLVM/llvm-${mart_llvm_version}/build_configure/Debug+Asserts/bin/clang++ \
#  --with-stp=/home/klee-semu/uclibc_solvers/stp \
#  --with-z3=/home/klee-semu/uclibc_solvers/z3/build \
#  --with-uclibc=/home/klee-semu/uclibc_solvers/klee-uclibc \
#  --enable-posix-runtime \
#  --enable-cxx11 \
#  && make -j2 \
#  && cp /home/klee-semu/uclibc_solvers/z3/build/libz3.so /home/klee-semu/klee_build/Debug+Asserts/lib
#ENV PATH="/home/klee-semu/klee_build/Debug+Asserts/bin:${PATH}"


RUN cd /home/klee-semu/klee_build \
  && CXXFLAGS="-fno-rtti -g" cmake -DCMAKE_BUILD_TYPE=Release \
           -DENABLE_KLEE_ASSERTS=true \
           -DENABLE_KLEE_UCLIBC=true \
           -DENABLE_POSIX_RUNTIME=true \
           -DENABLE_SOLVER_METASMT=false \
           -DENABLE_SOLVER_STP=true \
           -DENABLE_SOLVER_Z3=true \
           -DZ3_INCLUDE_DIRS=/home/klee-semu/uclibc_solvers/z3/install/include \
           -DZ3_LIBRARIES=/home/klee-semu/uclibc_solvers/z3/install/lib/libz3.so \
           -DENABLE_TCMALLOC=true \
           -DENABLE_TESTS=true \
           -DENABLE_UNIT_TESTS=OFF \
           -DGTEST_SRC_DIR=../src/unittests/ \
           -DKLEE_UCLIBC_PATH=/home/klee-semu/uclibc_solvers/klee-uclibc \
           -DSTP_DIR=/home/klee-semu/uclibc_solvers/stp/build \
           -DUSE_CXX11=true \
       ../klee_src \
  && CXXFLAGS='-fno-rtti -g' make

ENV PATH="/home/klee-semu/klee_build/Release+Asserts/bin:${PATH}"


