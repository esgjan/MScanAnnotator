@echo off
REM Windows launcher for MScanAnnotator.
REM Update these paths before sharing with another user:
REM 1) Change the project folder below if this repository lives somewhere else.
REM 2) Make sure the virtual environment exists at .venv\Scripts\activate.bat.
REM 3) Replace the final folder argument with that user's default scan directory.
REM After that, the user can double-click this .bat file to start the app.
cd /d "C:\Users\ZOJESSIG\git\MScanAnnotator"
call ".venv\Scripts\activate.bat"
oct-annotate "D:\iiOCT_data\npy"