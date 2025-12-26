@echo off
echo Starting Ambulance Detection Server
echo ===================================

:: Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Running setup...
    python setup.py
    if errorlevel 1 (
        echo Setup failed. Please check the errors above.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if model file exists
if not exist "models\best.pt" (
    echo.
    echo ❌ Model file not found!
    echo Please place your 'best.pt' file in the 'models' folder
    echo.
    pause
    exit /b 1
)

:: Start the server
echo Starting server...
python model_server.py

pause
