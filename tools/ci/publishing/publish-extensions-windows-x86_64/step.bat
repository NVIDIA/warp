@echo off
setlocal

call "%~dp0..\..\..\..\repo.bat" build --fetch-only
if %errorlevel% neq 0 ( exit /b %errorlevel% )

call "%~dp0..\..\..\..\repo.bat" publish_exts -c release --from-package %*
if %errorlevel% neq 0 ( exit /b %errorlevel% )

call "%~dp0..\..\..\..\repo.bat" publish_exts -c debug --from-package %*
if %errorlevel% neq 0 ( exit /b %errorlevel% )
