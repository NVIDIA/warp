@echo off

call "%~dp0..\..\..\..\repo.bat" build_number
if %errorlevel% neq 0 ( exit /b %errorlevel% )


