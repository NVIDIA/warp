REM @echo off
call "%~dp0repo" build --fetch-only %*

SET PYTHON="%~dp0\_build\target-deps\python\python.exe"

echo "Installing Warp to Python"
call %PYTHON% -m pip install -e .

echo "Running tests"
call %PYTHON% tests\test_all.py