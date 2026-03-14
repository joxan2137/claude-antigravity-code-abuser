@echo off
setlocal EnableDelayedExpansion

echo ===================================================
echo    Free Claude Code (Antigravity) Auto-Setup
echo ===================================================
echo.

:: 1. Check if uv is installed
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] 'uv' package manager not found. Installing...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%LOCALAPPDATA%\uv;%PATH%"
) else (
    echo [OK] 'uv' package manager is already installed.
)

:: 2. Setup .env file
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [OK] Created .env file from .env.example.
    ) else (
        echo [WARNING] .env.example not found! Cannot create .env cleanly.
    )
) else (
    echo [OK] .env file already exists.
)

:: 3. Configure Claude Code Desktop Settings (~/.claude/settings.json)
set "CLAUDE_DIR=%USERPROFILE%\.claude"
set "SETTINGS_FILE=%CLAUDE_DIR%\settings.json"

if not exist "%CLAUDE_DIR%" (
    mkdir "%CLAUDE_DIR%"
)

:: Use a temporary PowerShell script to manipulate the JSON robustly
set "PS_SCRIPT=%TEMP%\claude_setup_json.ps1"
echo $SettingsPath = "%SETTINGS_FILE%" > "%PS_SCRIPT%"
echo if (Test-Path $SettingsPath) { >> "%PS_SCRIPT%"
echo     $json = Get-Content $SettingsPath -Raw ^| ConvertFrom-Json -ErrorAction SilentlyContinue >> "%PS_SCRIPT%"
echo     if ($null -eq $json) { $json = @{} } >> "%PS_SCRIPT%"
echo } else { >> "%PS_SCRIPT%"
echo     $json = @{} >> "%PS_SCRIPT%"
echo } >> "%PS_SCRIPT%"
echo if ($null -eq $json.env) { >> "%PS_SCRIPT%"
echo     Add-Member -InputObject $json -NotePropertyName "env" -NotePropertyValue @{} >> "%PS_SCRIPT%"
echo } >> "%PS_SCRIPT%"
echo $json.env.ANTHROPIC_BASE_URL = "http://localhost:8082" >> "%PS_SCRIPT%"
echo $json.env.ANTHROPIC_AUTH_TOKEN = "freecc" >> "%PS_SCRIPT%"
echo $json ^| ConvertTo-Json -Depth 10 ^| Set-Content $SettingsPath >> "%PS_SCRIPT%"

powershell -ExecutionPolicy ByPass -File "%PS_SCRIPT%"
del "%PS_SCRIPT%"
echo [OK] Configured Claude Desktop settings (~/.claude/settings.json) to route through proxy.

echo.
echo ===================================================
echo Setup complete! Now we need to add your Google accounts.
echo A browser window will open for you to log in.
echo ===================================================
pause

:: 4. Run the Account Manager to add an account
echo [INFO] Launching Account Manager...
call uv run manage_accounts.py

echo.
echo ===================================================
echo ALL DONE! 
echo.
echo To start the proxy server, run this command:
echo    uv run uvicorn server:app --host 0.0.0.0 --port 8082
echo.
echo Once the proxy is running, any new Claude Code or Desktop App 
echo process will automatically route through the proxy!
echo ===================================================
pause
