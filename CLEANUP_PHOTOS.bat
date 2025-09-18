@echo off
title Intelligent Photo Cleanup Tool
color 0E

echo.
echo ===============================================
echo    INTELLIGENT PHOTO CLEANUP TOOL
echo ===============================================
echo.
echo This tool will help you:
echo.
echo  🔍 Review similarity groups interactively
echo  🎯 Choose which photos to keep/delete  
echo  🧹 Remove unwanted duplicates from system
echo  💾 Create clean organized photo collection
echo  📊 Save storage space
echo.
echo WARNING: This will DELETE unwanted photos from your system!
echo Make sure you have backups if needed.
echo.
pause

echo.
echo Starting interactive cleanup...
echo.

python interactive_cleanup.py

echo.
echo ===============================================
echo    CLEANUP PROCESS COMPLETE!
echo ===============================================
echo.
echo Your clean photo collection is now ready!
echo Check the CLEAN_PHOTOS folder for your final collection.
echo.
pause
