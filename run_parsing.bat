@echo off

REM List of values to pass to the script
set values=0 
set file_name="pim_trace_1ch_gemv_64x256.out"
set input_dim=256
set output_dim=64

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