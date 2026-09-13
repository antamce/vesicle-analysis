@echo off
setlocal
set "PROJECT_DIR=%~dp0"
if "%~1"=="" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%launcher\windows-launcher.ps1"
) else if /I "%~1"=="-SelfTest" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%launcher\windows-launcher.ps1" -SelfTest
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%launcher\windows-launcher.ps1" -AppArguments "%~1"
)
if errorlevel 1 (
    echo.
    echo Vesicle Analysis could not be started. Review the message above.
    pause
)
endlocal
