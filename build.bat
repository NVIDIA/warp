REM @echo off

REM pull packman dependencies
call "%~dp0repo" build --fetch-only %*

REM Use Packman python
SET PYTHON="%~dp0\_build\target-deps\python\python.exe"

REM Python dependencies
call %PYTHON% -m pip install numpy
call %PYTHON% -m pip install gitpython
call %PYTHON% -m pip install cmake
call %PYTHON% -m pip install ninja

REM Build
call %PYTHON% build_lib.py --msvc_path="_build/host-deps/msvc/VC/Tools/MSVC/14.29.30133" --sdk_path="_build/host-deps/winsdk" --cuda_path="_build/target-deps/cuda"
