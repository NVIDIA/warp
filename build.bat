REM @echo off
call "%~dp0repo" build --fetch-only %*

SET PYTHON="%~dp0\_build\target-deps\python\python.exe"

REM Dependencies
call %PYTHON% -m pip install numpy

call %PYTHON% build_lib.py