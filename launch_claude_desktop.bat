@echo off
setlocal

:: Set the environment variables for routing through the proxy
set "ANTHROPIC_BASE_URL=http://localhost:8082"
set "ANTHROPIC_AUTH_TOKEN=freecc"

echo Starting Claude Desktop with Antigravity proxy...

:: Dynamically get the Windows Store App installation path using PowerShell
for /f "delims=" %%I in ('powershell.exe -NoProfile -NonInteractive -Command "(Get-AppxPackage -Name 'Claude').InstallLocation"') do set "CLAUDE_DIR=%%I"

if "%CLAUDE_DIR%"=="" (
    echo [ERROR] Claude Desktop could not be found via Get-AppxPackage.
    echo Make sure you have the official Claude app installed from the web or store.
    pause
    exit /b 1
)

set "CLAUDE_EXE=%CLAUDE_DIR%\app\Claude.exe"

if exist "%CLAUDE_EXE%" (
    start "" "%CLAUDE_EXE%"
) else (
    echo [ERROR] Claude executable not found at: %CLAUDE_EXE%
    pause
    exit /b 1
)
