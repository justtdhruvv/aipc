@echo off
title AI-Powered Intelligent Photo Organizer
color 0A

echo.
echo ===============================================
echo    AI-POWERED INTELLIGENT PHOTO ORGANIZER
echo ===============================================
echo.
echo Advanced AI-powered photo organization using:
echo.
echo  🧠 CLIP SEMANTIC      - Understands photo content/meaning
echo  👁️  RESNET VISUAL      - Deep visual pattern recognition  
echo  👤 FACE RECOGNITION   - Groups photos by people
echo  🎯 OBJECT DETECTION   - Categorizes by content (landscapes, objects)
echo  🔍 HYBRID SIMILARITY  - Combines AI + traditional algorithms
echo.
echo AI CATEGORIES:
echo  1. FACES_PEOPLE       - Photos grouped by people detected
echo  2. LANDSCAPES_NATURE  - Natural scenes and landscapes
echo  3. OBJECTS_ITEMS      - Items, vehicles, and objects
echo  4. AI_SIMILAR_SEMANTIC - Semantically similar content
echo  5. AI_SIMILAR_VISUAL   - Visually similar appearance
echo  6. MIXED_GROUPS        - Multi-category similarities
echo  7. UNIQUE_AI          - Unique photos with no matches
echo.
echo INSTRUCTIONS:
echo  1. Put your photos in the 'photos_to_scan' folder
echo  2. Press any key to start AI organization
echo  3. Review AI results in 'organized_photos' folder
echo.
pause

echo.
echo 🚀 Starting AI-Powered Photo Organization...
echo Loading advanced AI models (CLIP, ResNet50, YOLO, Face Recognition)...
echo This may take a moment to initialize AI models on first run.
echo.

python ai_photo_organizer.py

echo.
echo ===============================================
echo    AI ORGANIZATION COMPLETE!
echo ===============================================
echo.
echo Check the 'organized_photos' folder for AI-organized results:
echo.
echo  1_FACES_PEOPLE       - Photos grouped by faces detected
echo  2_LANDSCAPES_NATURE  - Natural scenes and outdoor photos
echo  3_OBJECTS_ITEMS      - Items, vehicles, and object photos
echo  4_AI_SIMILAR_SEMANTIC - Photos with similar meaning/content  
echo  5_AI_SIMILAR_VISUAL   - Photos with similar visual appearance
echo  6_TRADITIONAL_SIMILAR - Traditional hash-based similarities
echo  7_MIXED_GROUPS        - Cross-category similar photos
echo  8_UNIQUE_AI          - Unique photos with no AI matches
echo.
echo NOW DETECTS PHOTOS WITH MINOR DIFFERENCES!
echo.
pause
