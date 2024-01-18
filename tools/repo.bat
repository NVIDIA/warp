@echo off

call "%~dp0\packman\python.bat" "%~dp0\repoman\repoman.py" %*
if %errorlevel% neq 0 ( goto Error )

:Success
exit /b 0

:Error
exit /b %errorlevel%
