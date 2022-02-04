REM @echo off
call "%~dp0repo" build --fetch-only %*

SET PYTHON="%~dp0\_build\target-deps\python\python.exe"

REM Dependencies
call %PYTHON% -m pip install numpy
call %PYTHON% -m pip install pillow

REM WinSDK
call %PYTHON% build_lib.py

REM copy linux dependencies to bin dir
copy _build\target-deps\cuda\bin\nvrtc*.dll warp\bin
copy _build\target-deps\cuda\bin\cudart64*.dll warp\bin
