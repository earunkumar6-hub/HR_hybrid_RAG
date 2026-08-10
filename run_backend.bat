@echo off
REM Starts the FastAPI backend on http://localhost:8000 (docs at /docs).
REM Place this file in the project root (same folder as requirements.txt).

setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM ---- load .env into the environment (skips blank lines and #comments) ----
if exist .env (
    for /f "eol=# tokens=1,* delims==" %%A in (.env) do (
        set "%%A=%%B"
    )
)

if not defined BACKEND_HOST set "BACKEND_HOST=0.0.0.0"
if not defined BACKEND_PORT set "BACKEND_PORT=8000"

cd backend
uvicorn app:app --host %BACKEND_HOST% --port %BACKEND_PORT% --reload

if errorlevel 1 (
    echo.
    echo Backend exited with an error - see above.
    pause
)

endlocal
