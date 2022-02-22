REM @echo off

call "%~dp0repo" build --fetch-only %*

SET PYTHON="%~dp0\_build\target-deps\python\python.exe"

REM Host dependencies
REM tools\packman\packman pull -p windows-x86_64 deps\host-deps.packman.xml
REM tools\packman\packman pull -p windows-x86_64 deps\target-deps.packman.xml

REM Python dependencies
call %PYTHON% -m pip install numpy

REM Build
call %PYTHON% build_lib.py

REM Copy CUDA dependencies to bin dir
copy _build\target-deps\cuda\bin\nvrtc*.dll warp\bin
copy _build\target-deps\cuda\bin\cudart64*.dll warp\bin
