```shell 
pip install -r requirements.txt

cmake \
  -DPython3_EXECUTABLE=$(which python)
  -DCMAKE_PREFIX_PATH=$(python -c "print(__import__('mlir').__path__[0])")
```