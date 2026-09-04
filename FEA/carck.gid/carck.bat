@echo off
REM carck solver launcher for GiD
REM Called as: carck.bat <basename> <n> <problem_dir>
set BASENAME=%1
set PROBDIR=%3
set SOLVER_DIR=D:\test\Crack\FEA
cd %2 
mkdir a

REM Normalise separators
set PROBDIR=%PROBDIR:/=\%
if "%PROBDIR:~-1%"=="\" set PROBDIR=%PROBDIR:~0,-1%

set DAT_FILE=%PROBDIR%\%BASENAME%.dat

if not exist "%DAT_FILE%" (
    echo ERROR: DAT file not found: %DAT_FILE%
    exit /b 1
)

echo Running carck solver on %DAT_FILE% ...
python "%SOLVER_DIR%\run_fea.py" "%DAT_FILE%" --output "%PROBDIR%"

if errorlevel 1 (
    echo ERROR: carck solver failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Solver finished successfully.
