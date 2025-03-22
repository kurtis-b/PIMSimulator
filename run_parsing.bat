@echo off
setlocal enabledelayedexpansion

REM List of values to pass to the script
set values=0
set input_dim=256 512 1024 2048
set output_dim=64 128 256

REM Loop through each value and run the script in the background
for %%a in (%values%) do (
    for %%b in (%input_dim%) do (
        for %%c in (%output_dim%) do (
            set "filename=pim_trace_1ch_gemv_%%cx%%b.out"
            start /B python parse_output_channels.py %%a !filename! %%b %%c
        )
    )
)

REM Wait for all background processes to finish
:waitloop
tasklist | find /I "parse_output_channels.py" >nul 2>&1
if not errorlevel 1 (
    timeout /T 1 /NOBREAK >nul
    goto waitloop
)

echo All processes have completed.