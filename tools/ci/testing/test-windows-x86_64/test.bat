REM @echo off
call "%~dp0..\..\..\..\repo.bat" build --fetch-only %*

SET PYTHON="%~dp0..\..\..\..\_build\target-deps\python\python.exe"

echo "Installing test dependencies"
call %PYTHON% -m pip install matplotlib
call %PYTHON% -m pip install usd-core
call %PYTHON% -m pip install coverage
call %PYTHON% -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu115
:: According to Jax docs, the pip packages don't work on Windows and may fail silently
::call %PYTHON% -m pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "Installing Warp to Python"
call %PYTHON% -m pip install -e "%~dp0..\..\..\..\."

echo "Running tests"
call %PYTHON% -m warp.tests
