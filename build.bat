REM @echo off

REM pull packman dependencies
call "%~dp0repo" build --fetch-only %*

REM Use Packman python
SET PYTHON="%~dp0\_build\target-deps\python\python.exe"

REM Python dependencies
call %PYTHON% -m pip install numpy

REM Build
call %PYTHON% build_lib.py --msvc_path="_build/host-deps/msvc/VC/Tools/MSVC/14.16.27023" --sdk_path="_build/host-deps/winsdk" --cuda_path="_build/target-deps/cuda"

REM Copy CUDA dependencies to bin dir
copy _build\target-deps\cuda\bin\nvrtc*.dll warp\bin
copy _build\target-deps\cuda\bin\cudart64*.dll warp\bin
