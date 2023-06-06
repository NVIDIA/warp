@echo off
setlocal

call "%~dp0..\..\..\..\repo.bat" docs --config release --stage publish --edition s3web --publish-as-latest
if %errorlevel% neq 0 ( exit /b %errorlevel% )
