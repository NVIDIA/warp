@echo off
call "%~dp0repo" build --fetch-only %*

call python build_lib.py