@echo off

call "%~dp0..\..\publish.bat"
if %errorlevel% neq 0 ( exit /b %errorlevel% )
