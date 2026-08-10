@echo off
REM Starts the Streamlit frontend on http://localhost:8501
REM Place this file in the project root (same folder as requirements.txt).

setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM ---- load .env into the environment (skips blank lines and #comments) ----
if exist .env (
    for /f "eol=# tokens=1,* delims==" %%A in (.env) do (
        set "%%A=%%B"
    )
)

cd frontend
streamlit run app.py

if errorlevel 1 (
    echo.
    echo Frontend exited with an error - see above.
    pause
)

endlocal
