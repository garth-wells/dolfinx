# Dockerfile to build the FEniCS-X development libraries
#
# Authors:
# Jack S. Hale <jack.hale@uni.lu>
# Lizao Li <lzlarryli@gmail.com>
# Garth N. Wells <gnw20@cam.ac.uk>
# Jan Blechta <blechta@karlin.mff.cuni.cz>

FROM ubuntu:18.04
LABEL maintainer="fenics-project <fenics-support@googlegroups.org>"

WORKDIR /tmp

# Environment variables
ENV OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_VERBOSE=0

# Install dependencies available via apt-get
RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    cmake \
    doxygen \
    g++ \
    gfortran \
    git \
    gmsh \
    graphviz \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-iostreams-dev \
    libboost-math-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-timer-dev \
    libeigen3-dev \
    libhdf5-openmpi-dev \
    liblapack-dev \
    libopenmpi-dev \
    libopenblas-dev \
    openmpi-bin \
    pkg-config \
    python3-dev \
    valgrind \
    wget \
    bash-completion && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install setuptools
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    pip3 install --no-cache-dir setuptools && \
    rm -rf /tmp/*


# Install PETSc from source. PETSc build system needs Python 2 :(.
ARG PETSC_VERSION=3.9.1
RUN apt-get -qq update && \
    apt-get -y install bison flex python && \
    wget -nc --quiet https://bitbucket.org/petsc/petsc/get/v${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p petsc-src && tar -xf petsc-${PETSC_VERSION}.tar.gz -C petsc-src --strip-components 1 && \
    cd petsc-src && \
    ./configure \
    --COPTFLAGS="-O2 -g" \
    --CXXOPTFLAGS="-O2 -g" \
    --FOPTFLAGS="-O2 -g" \
    --with-debugging=yes \
    --with-fortran-bindings=no \
    --download-blacs \
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-spai \
    --download-suitesparse \
    --download-superlu \
    --prefix=/usr/local/petsc-32 && \
    make && \
    make install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install SLEPc from source
# NOTE: Issues building SLEPc from source tarball generated by
#       Bitbucket. Website tarballs work fine, however.
ARG SLEPC_VERSION=3.9.1
RUN wget -nc --quiet http://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p slepc-src && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C slepc-src --strip-components 1 && \
    export PETSC_DIR=/usr/local/petsc-32 && \
    cd slepc-src && \
    ./configure --prefix=/usr/local/slepc-32 && \
    make SLEPC_DIR=$(pwd) && \
    make install && \
    rm -rf /tmp/*

# By default use the 32-bit build of SLEPc and PETSc
ENV SLEPC_DIR=/usr/local/slepc-32 \
    PETSC_DIR=/usr/local/petsc-32

# Install Python packages (via pip)
ARG PETSC4PY_VERSION=3.9.1
ARG SLEPC4PY_VERSION=3.9.0
RUN pip3 install --no-cache-dir mpi4py numpy scipy numba && \
    pip3 install --no-cache-dir https://bitbucket.org/petsc/petsc4py/downloads/petsc4py-${PETSC4PY_VERSION}.tar.gz && \
    pip3 install --no-cache-dir https://bitbucket.org/slepc/slepc4py/downloads/slepc4py-${SLEPC4PY_VERSION}.tar.gz

# Install pybind11
ARG PYBIND11_VERSION=2.2.3
RUN wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz && \
    tar -xf v${PYBIND11_VERSION}.tar.gz && \
    cd pybind11-${PYBIND11_VERSION} && \
    mkdir build && \
    cd build && \
    cmake -DPYBIND11_TEST=False ../ && \
    make && \
    make install && \
    rm -rf /tmp/*

# Install FIAT, UFL, dijitso and ffcX (development versions, master branch)
RUN pip3 install --no-cache-dir git+https://bitbucket.org/fenics-project/fiat.git && \
    pip3 install --no-cache-dir git+https://bitbucket.org/fenics-project/ufl.git && \
    pip3 install --no-cache-dir git+https://bitbucket.org/fenics-project/dijitso.git && \
    pip3 install --no-cache-dir git+https://github.com/fenics/ffcX

# Install dolfinx
RUN git clone https://github.com/fenics/dolfinx.git && \
    cd dolfinx && \
    mkdir build && \
    cd build && \
    cmake ../cpp && \
    make && \
    make install && \
    rm -rf /tmp/*
