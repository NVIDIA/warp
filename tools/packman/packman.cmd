:: RUN_PM_MODULE must always be at the same spot for packman update to work (batch reloads file during update!) 
:: [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]
:: Reset errorlevel status (don't inherit from caller) 
@call :ECHO_AND_RESET_ERROR
:: You can remove the call below if you do your own manual configuration of the dev machines
call "%~dp0\bootstrap\configure.bat"

if %errorlevel% neq 0 ( exit /b %errorlevel% )
:: Everything below is mandatory
if not defined PM_PYTHON goto :PYTHON_ENV_ERROR
if not defined PM_MODULE goto :MODULE_ENV_ERROR

set PM_VAR_PATH_ARG=

if "%1"=="pull" goto :SET_VAR_PATH
if "%1"=="install" goto :SET_VAR_PATH

:RUN_PM_MODULE
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

:SET_VAR_PATH
:: Generate temporary path for variable file
for /f "delims=" %%a in ('%PM_PYTHON% -S -s -u -E -c "import tempfile;file = tempfile.NamedTemporaryFile(mode='w+t', delete=False);print(file.name)"') do (set PM_VAR_PATH=%%a)
set PM_VAR_PATH_ARG=--var-path="%PM_VAR_PATH%"
goto :RUN_PM_MODULE
