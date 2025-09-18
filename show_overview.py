#!/usr/bin/env python3
"""
Quick Photo Organization Overview
=================================
Shows you exactly what's in your organized folders
"""

from pathlib import Path

def show_overview():
    base_dir = Path(__file__).parent
    organized_dir = base_dir / "organized_photos"
    
    print("📁 ORGANIZED PHOTOS OVERVIEW")
    print("=" * 50)
    
    total_photos = 0
    
    # Check each AI folder
    folders = [
        ("1_FACES_PEOPLE", "Photos grouped by people detected"),
        ("2_LANDSCAPES_NATURE", "Natural scenes and landscapes"),
        ("3_OBJECTS_ITEMS", "Items, vehicles, and objects"),
        ("4_AI_SIMILAR_SEMANTIC", "Semantically similar content"),
        ("5_AI_SIMILAR_VISUAL", "Visually similar appearance"),
        ("6_TRADITIONAL_SIMILAR", "Traditional hash-based similarities"),
        ("7_MIXED_GROUPS", "Cross-category similar photos"),
        ("8_UNIQUE_AI", "Unique photos (no AI similarities)")
    ]
    
    for folder_name, description in folders:
        folder_path = organized_dir / folder_name
        if folder_path.exists():
            if folder_name == "8_UNIQUE_AI":
                # Count files directly
                photos = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpeg"))
                count = len(photos)
                total_photos += count
                print(f"\n📂 {folder_name}/")
                print(f"   {description}")
                print(f"   📸 {count} photos")
                if count > 0:
                    for photo in photos[:3]:  # Show first 3
                        print(f"      • {photo.name}")
                    if count > 3:
                        print(f"      • ... and {count-3} more")
            else:
                # Count groups and photos in subfolders
                groups = [d for d in folder_path.iterdir() if d.is_dir()]
                group_count = len(groups)
                photo_count = 0
                
                for group in groups:
                    group_photos = list(group.glob("*.jpg")) + list(group.glob("*.png")) + list(group.glob("*.jpeg"))
                    photo_count += len(group_photos)
                
                total_photos += photo_count
                print(f"\n📂 {folder_name}/")
                print(f"   {description}")
                print(f"   📁 {group_count} groups, 📸 {photo_count} photos")
                
                # Show group details
                for group in groups[:3]:  # Show first 3 groups
                    group_photos = list(group.glob("*.jpg")) + list(group.glob("*.png")) + list(group.glob("*.jpeg"))
                    print(f"   📁 {group.name}: {len(group_photos)} photos")
                    for photo in group_photos[:2]:  # Show first 2 photos
                        print(f"      • {photo.name}")
                    if len(group_photos) > 2:
                        print(f"      • ... and {len(group_photos)-2} more")
                
                if group_count > 3:
                    print(f"   📁 ... and {group_count-3} more groups")
    
    print(f"\n📊 TOTAL: {total_photos} photos organized")
    print("\n💡 Run CLEANUP_PHOTOS.bat to interactively clean up your collection!")

if __name__ == "__main__":
    show_overview()
