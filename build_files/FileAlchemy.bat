@echo off
echo ===================================
echo    FileAlchemy Conversion Suite
echo ===================================
echo.
echo Starting FileAlchemy Server...
echo.
echo The web interface will open automatically in your default browser.
echo.
echo NOTE: Keep this window open while using FileAlchemy.
echo       Close this window to shut down the server when you're done.
echo.
echo Opening browser...
timeout /t 2 > nul
start "" http://localhost:8001
echo.
echo Server is running...
echo.
FileAlchemy.exe
echo.
echo Server has been shut down.
echo Thank you for using FileAlchemy!
pause 