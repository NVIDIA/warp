@echo off

:: tests
call "%~dp0..\..\..\..\repo.bat" test --config release --from-package
if %errorlevel% neq 0 ( exit /b %errorlevel% )

