REM @echo off
call "%~dp0repo" build --fetch-only %*

SET PYTHON="%~dp0\_build\target-deps\python\python.exe"

dir
dir _build
dir _build\packages

cd _build\packages

echo "Installing Warp to Python"
call %PYTHON% -m pip install -e .

echo "Running tests"
call %PYTHON% tests\test_ctypes.py