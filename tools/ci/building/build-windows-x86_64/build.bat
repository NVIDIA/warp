REM @echo off

REM pull packman dependencies
call "%~dp0..\..\..\..\repo.bat" build --fetch-only %*

REM Use Packman python
SET PYTHON="%~dp0..\..\..\..\_build\target-deps\python\python.exe"

REM Python dependencies
call %PYTHON% -m pip install --upgrade pip
call %PYTHON% -m pip install numpy
call %PYTHON% -m pip install gitpython
call %PYTHON% -m pip install cmake
call %PYTHON% -m pip install ninja

SET BUILD_MODE="release"
for %%i in (%*) do (
    if "%%i"=="--debug" set BUILD_MODE="debug"
)

REM Build
call %PYTHON% "%~dp0..\..\..\..\build_lib.py" ^
    --msvc_path="%~dp0..\..\..\..\_build\host-deps\msvc\VC\Tools\MSVC\14.29.30133" ^
    --sdk_path="%~dp0..\..\..\..\_build\host-deps\winsdk" ^
    --cuda_path="%~dp0..\..\..\..\_build\target-deps\cuda" ^
    --mode=%BUILD_MODE%
