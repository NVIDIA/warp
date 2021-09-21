:: Reset errorlevel status (don't inherit from caller) [xxxxxxxxxxx]
@call :ECHO_AND_RESET_ERROR
:: You can remove the call below if you do your own manual configuration of the dev machines
call "%~dp0\bootstrap\configure.bat"

if %errorlevel% neq 0 ( exit /b %errorlevel% )
:: Everything below is mandatory
if not defined PM_PYTHON goto :PYTHON_ENV_ERROR
if not defined PM_MODULE goto :MODULE_ENV_ERROR

:: Generate temporary path for variable file
for /f "delims=" %%a in ('powershell -ExecutionPolicy ByPass -NoLogo -NoProfile ^
-File "%~dp0bootstrap\generate_temp_file_name.ps1"') do set PM_VAR_PATH=%%a

if %1.==. (
	set PM_VAR_PATH_ARG=
) else (
	set PM_VAR_PATH_ARG=--var-path="%PM_VAR_PATH%"
)

"%PM_PYTHON%" -S -s -u -E "%PM_MODULE%" %* %PM_VAR_PATH_ARG%
if %errorlevel% neq 0 ( exit /b %errorlevel% )

:: Marshall environment variables into the current environment if they have been generated and remove temporary file
if exist "%PM_VAR_PATH%" (
	for /F "usebackq tokens=*" %%A in ("%PM_VAR_PATH%") do set "%%A"
)
if %errorlevel% neq 0 ( goto :VAR_ERROR )

if exist "%PM_VAR_PATH%" (
	del /F "%PM_VAR_PATH%"
)
if %errorlevel% neq 0 ( goto :VAR_ERROR )

set PM_VAR_PATH=
goto :eof

:: Subroutines below
:PYTHON_ENV_ERROR
@echo User environment variable PM_PYTHON is not set! Please configure machine for packman or call configure.bat.
exit /b 1

:MODULE_ENV_ERROR
@echo User environment variable PM_MODULE is not set! Please configure machine for packman or call configure.bat.
exit /b 1

:VAR_ERROR
@echo Error while processing and setting environment variables!
exit /b 1

:ECHO_AND_RESET_ERROR
@echo off
if /I "%PM_VERBOSITY%"=="debug" (
	@echo on
)
exit /b 0
