
# Submodules

    rlqp_train (this repo, not a fork)
    +- rlqp_benchmarks  (fork of https://github.com/osqp/osqp_benchmarks)
    
# Requirements:

The RLQP python solver must be separately installed.

    rlqp-python      (fork of https://github.com/osqp/osqp-python)
    +- rlqp_solver   (fork of https://github.com/osqp/osqp.git)
       +- qdldl      (https://github.com/osqp/qdldl, not forked)

# Setup instructions

    % git submodule update --init --recursive


# Virtualenv Instructions

For all these instructions, we assume the environment variable `RLQP` is set to the directory in which this README.md file is contained.

Create a virtual environment (1 time)

    % cd $RLQP
    % virtualenv venv/rlqp

Activate the virtual environment (run once in each new terminal):

    % . $RLQP/venv/rlqp/bin/activate

Compile and install the RLQP Python wrapper (1 time per git update)

    % pip uninstall -y rlqp
    % cd $RLQP/rlqp-python
    % rm -rf build extension/src/*.a osqp_sources/build
    % python setup.py install

Install rlqp_benchmarks to the venv:

    % cd $RLQP/rlqp_benchmarks
    % python setup.py develop

Place this project in development mode (1 time)

    % cd $RLQP
    % python setup.py develop
