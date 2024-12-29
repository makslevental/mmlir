# Minimal MLIR

This repo demos a slightly unconventional way to bootstrap an MLIR project:

1. It relies on https://makslevental.github.io/wheels for the upstream distribution of MLIR
2. It smashes all the include headers into a single header [include/MinimalDialect.h](include/MinimalDialect.h) and all the tablegen into a single [include/MinimalDialect.td](include/MinimalDialect.td) and *also emits all the tablegen into the source tree itself*;
   1. I think seeing the emitted tablegen is useful for demystifying how MLIR works
3. It smashes all the implementation into a single [src/MinimalDialect.cpp](src/MinimalDialect.cpp)
4. Python bindings @ [python/mmlir/dialects](python/mmlir/dialects) are arranged to have generated artifacts to be dumped in place.

It is primarily meant to be used as a learning aid (e.g., for understanding which parts of the upstream CMake are essential and which aren't) and not as a germ/seed/cookiecutter for a production quality project.

# Building and exercising

You can either use CMake to build and run lit tests or you can `pip install` and run [test/python/smoketest.py](test/python/smoketest.py).
Note, `pip install -r requirements.txt` is required either way and `pip download mlir -f https://makslevental.github.io/wheels` in order to get the `mlir` distribution package.

A minimal CMake might look like:

```shell 
pip install -r requirements.txt
pip download mlir -f https://makslevental.github.io/wheels
unzip mlir-*.whl

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$(which python) \
  -DCMAKE_PREFIX_PATH=$PWD/mlir \
  -DLLVM_EXTERNAL_LIT=$(which lit) \
  -B build \
  -S $PWD
```

and then just do `pushd build && ninja check-minimal && popd`.

Alternatively you can `pip install`, e.g., `pip install . -v --no-build-isolation` and then `pushd test/python && pthon smoketest.py && popd`.

If something isn't working you're probably missing `ninja` or `CMake` or you haven't done `pip install -r requirements.txt`.
My recommendation is to go to the tests/GitHub actions and see how they're run since they run consistently.
