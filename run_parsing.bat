@echo off

REM List of values to pass to the script
set values=0 1 2 3 8 9 15 16 17 24 25 32 33 40 41 48 49 56 57 60 61 62 63
set file_name="pim_trace_1027x256.out"
set input_dim=256
set output_dim=1027

REM Loop through each value and run the script in the background
for %%a in (%values%) do (
    start /B python parse_animate_faster.py %%a %file_name% %input_dim% %output_dim%
)

REM Wait for all background processes to finish
:waitloop
tasklist | find /I "python.exe" >nul 2>&1
if not errorlevel 1 (
    timeout /T 1 /NOBREAK >nul
    goto waitloop
)

echo All processes have completed.