@echo off
rem Camera license plate recognition GUI (MVVM, pythonw, no console window)
rem   run.bat                         -> last used camera and resolution, recognition off (press I)
rem   run.bat --camera 1 --size 640x480
rem   run.bat --video D:\videos\road.mp4   -> start with a video file
cd /d "%~dp0"
start "" "C:\Users\USER\anaconda3\envs\physical_ai\pythonw.exe" main.py %*
