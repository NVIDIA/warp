REM @echo off
call "%~dp0repo" build --fetch-only %*

SET PYTHON="%~dp0\_build\target-deps\python\python.exe"

REM Dependencies
call %PYTHON% -m pip install numpy

REM WinSDK
SET WindowsSDKDir="%~dp0\_build\host-deps\winsdk"

call %PYTHON% build_lib.py