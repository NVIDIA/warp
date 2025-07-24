@echo off
setlocal

:: Enable delayed expansion
SETLOCAL ENABLEDELAYEDEXPANSION

SET "BUILD_MODE=release"
SET "CUDA_MAJOR_VER=12"

:: Loop through arguments
SET "PREV_ARG="
for %%i in (%*) do (
    if "%%i"=="--debug" (
        SET "BUILD_MODE=debug"
    ) else if "!PREV_ARG!"=="--cuda" (
        SET "CUDA_MAJOR_VER=%%i"
    )
    SET "PREV_ARG=%%i"
)

:: pull packman dependencies
SET "PLATFORM=windows-x86_64"
call "%~dp0..\..\..\packman\packman" pull --platform %PLATFORM% "%~dp0..\..\..\..\deps/target-deps.packman.xml" --verbose
call "%~dp0..\..\..\packman\packman" pull --platform %PLATFORM% "%~dp0..\..\..\..\deps/cuda-toolkit-deps.packman.xml" --verbose --include-tag "cuda-%CUDA_MAJOR_VER%"
call "%~dp0..\..\..\packman\packman" install --verbose -l "%~dp0..\..\..\..\_build\host-deps\winsdk" winsdk 10.17763
call "%~dp0..\..\..\packman\packman" install --verbose -l "%~dp0..\..\..\..\_build\host-deps\msvc" msvc 2019-16.11.24

:: Use Packman python
SET PYTHON="%~dp0..\..\..\..\_build\target-deps\python\python.exe"

:: Python dependencies
call %PYTHON% -m pip install --upgrade pip
call %PYTHON% -m pip install numpy
call %PYTHON% -m pip install gitpython
call %PYTHON% -m pip install cmake
call %PYTHON% -m pip install ninja

:: Build
call %PYTHON% -u "%~dp0..\..\..\..\build_lib.py" ^
    --msvc_path="%~dp0..\..\..\..\_build\host-deps\msvc\VC\Tools\MSVC\14.29.30133" ^
    --sdk_path="%~dp0..\..\..\..\_build\host-deps\winsdk" ^
    --cuda_path="%~dp0..\..\..\..\_build\target-deps\cuda" ^
    --mode=%BUILD_MODE%

endlocal

if %ERRORLEVEL% neq 0 (
    exit /b %ERRORLEVEL%
)
