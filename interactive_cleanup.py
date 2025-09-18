#!/usr/bin/env python3
"""
INTELLIGENT PHOTO CLEANUP TOOL
==============================

This tool helps you systematically clean up your photo collection by:
1. Showing you what's in each similarity group
2. Letting you choose which photos to keep/delete
3. Moving the final clean collection to a "CLEAN_PHOTOS" folder
4. Removing all unwanted duplicates and similar photos
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

class PhotoCleanupTool:
    """Interactive photo cleanup tool"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.organized_dir = self.base_dir / "organized_photos"
        self.clean_photos_dir = self.base_dir / "CLEAN_PHOTOS"
        
        # Create clean photos directory
        if self.clean_photos_dir.exists():
            shutil.rmtree(self.clean_photos_dir)
        self.clean_photos_dir.mkdir(exist_ok=True)
        
        self.stats = {
            'total_original': 0,
            'total_kept': 0,
            'total_deleted': 0,
            'groups_processed': 0
        }

    def count_all_photos(self):
        """Count all photos in organized folders"""
        total = 0
        for folder in self.organized_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith('7_REPORTS'):
                if folder.name == '6_UNIQUE_PHOTOS':
                    total += len(list(folder.glob('*.png')))
                else:
                    for subfolder in folder.iterdir():
                        if subfolder.is_dir():
                            total += len(list(subfolder.glob('*.png')))
        return total

    def show_group_photos(self, group_path):
        """Show photos in a group"""
        photos = list(group_path.glob('*.png'))
        print(f"\n📁 Group: {group_path.name}")
        print(f"📸 Contains {len(photos)} photos:")
        for i, photo in enumerate(photos, 1):
            size_mb = photo.stat().st_size / (1024 * 1024)
            print(f"  {i}. {photo.name} ({size_mb:.2f} MB)")
        return photos

    def get_user_choice(self, photos):
        """Get user choice for which photos to keep"""
        print(f"\n💡 OPTIONS:")
        print(f"  1. Keep FIRST photo only")
        print(f"  2. Keep LAST photo only") 
        print(f"  3. Choose specific photos to keep")
        print(f"  4. Keep ALL photos")
        print(f"  5. DELETE all photos")
        
        while True:
            try:
                choice = input(f"\n🔤 Your choice (1-5): ").strip()
                
                if choice == '1':
                    return [photos[0]]
                elif choice == '2':
                    return [photos[-1]]
                elif choice == '3':
                    print(f"\n📋 Enter photo numbers to KEEP (e.g., 1,3,5):")
                    numbers = input("📝 Numbers: ").strip()
                    if numbers:
                        keep_indices = [int(n.strip()) - 1 for n in numbers.split(',')]
                        return [photos[i] for i in keep_indices if 0 <= i < len(photos)]
                    return []
                elif choice == '4':
                    return photos
                elif choice == '5':
                    return []
                else:
                    print("❌ Please enter 1, 2, 3, 4, or 5")
            except (ValueError, IndexError):
                print("❌ Invalid input. Please try again.")

    def process_similarity_groups(self):
        """Process all similarity groups interactively"""
        print("🔍 PROCESSING SIMILARITY GROUPS")
        print("=" * 50)
        
        # Process Very Similar (80-95%)
        very_similar_dir = self.organized_dir / "3_VERY_SIMILAR_80"
        if very_similar_dir.exists():
            groups = list(very_similar_dir.iterdir())
            if groups:
                print(f"\n🔥 VERY SIMILAR GROUPS (80-95% match) - {len(groups)} groups")
                print("These photos are almost identical - keep only the best quality!")
                
                for group_path in groups:
                    if group_path.is_dir():
                        photos = self.show_group_photos(group_path)
                        keep_photos = self.get_user_choice(photos)
                        self.move_selected_photos(keep_photos, photos)
                        self.stats['groups_processed'] += 1

        # Process Somewhat Similar (50-65%)
        somewhat_similar_dir = self.organized_dir / "5_SOMEWHAT_SIMILAR_50"
        if somewhat_similar_dir.exists():
            groups = list(somewhat_similar_dir.iterdir())
            if groups:
                print(f"\n📎 SOMEWHAT SIMILAR GROUPS (50-65% match) - {len(groups)} groups")
                print("These photos have visual similarities with minor differences")
                
                for group_path in groups:
                    if group_path.is_dir():
                        photos = self.show_group_photos(group_path)
                        keep_photos = self.get_user_choice(photos)
                        self.move_selected_photos(keep_photos, photos)
                        self.stats['groups_processed'] += 1

    def move_selected_photos(self, keep_photos, all_photos):
        """Move selected photos to clean folder and track stats"""
        # Copy kept photos to clean folder
        for photo in keep_photos:
            try:
                dst_path = self.clean_photos_dir / photo.name
                # Handle name conflicts by adding number
                counter = 1
                while dst_path.exists():
                    name_parts = photo.stem, counter, photo.suffix
                    dst_path = self.clean_photos_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                    counter += 1
                
                shutil.copy2(photo, dst_path)
                self.stats['total_kept'] += 1
                print(f"  ✅ Kept: {photo.name}")
            except Exception as e:
                print(f"  ⚠️ Error copying {photo.name}: {e}")
        
        # Count deleted photos
        deleted_count = len(all_photos) - len(keep_photos)
        self.stats['total_deleted'] += deleted_count
        if deleted_count > 0:
            print(f"  🗑️ Will delete: {deleted_count} photos")

    def process_unique_photos(self):
        """Process unique photos"""
        unique_dir = self.organized_dir / "6_UNIQUE_PHOTOS"
        if unique_dir.exists():
            photos = list(unique_dir.glob('*.png'))
            if photos:
                print(f"\n🌟 UNIQUE PHOTOS - {len(photos)} photos")
                print("These photos have no duplicates or similarities")
                
                print(f"\n💡 OPTIONS:")
                print(f"  1. Keep ALL unique photos (recommended)")
                print(f"  2. Review and select specific photos")
                print(f"  3. Skip unique photos")
                
                while True:
                    choice = input(f"\n🔤 Your choice (1-3): ").strip()
                    if choice == '1':
                        # Keep all unique photos
                        for photo in photos:
                            try:
                                dst_path = self.clean_photos_dir / photo.name
                                shutil.copy2(photo, dst_path)
                                self.stats['total_kept'] += 1
                            except Exception as e:
                                print(f"  ⚠️ Error copying {photo.name}: {e}")
                        print(f"  ✅ Kept all {len(photos)} unique photos")
                        break
                    elif choice == '2':
                        for i, photo in enumerate(photos, 1):
                            size_mb = photo.stat().st_size / (1024 * 1024)
                            print(f"  {i}. {photo.name} ({size_mb:.2f} MB)")
                        
                        numbers = input(f"\n📝 Enter photo numbers to KEEP (e.g., 1,3,5) or 'all': ").strip()
                        if numbers.lower() == 'all':
                            keep_photos = photos
                        elif numbers:
                            try:
                                keep_indices = [int(n.strip()) - 1 for n in numbers.split(',')]
                                keep_photos = [photos[i] for i in keep_indices if 0 <= i < len(photos)]
                            except (ValueError, IndexError):
                                print("❌ Invalid input. Keeping all photos.")
                                keep_photos = photos
                        else:
                            keep_photos = []
                        
                        for photo in keep_photos:
                            try:
                                dst_path = self.clean_photos_dir / photo.name
                                shutil.copy2(photo, dst_path)
                                self.stats['total_kept'] += 1
                            except Exception as e:
                                print(f"  ⚠️ Error copying {photo.name}: {e}")
                        
                        deleted_count = len(photos) - len(keep_photos)
                        self.stats['total_deleted'] += deleted_count
                        print(f"  ✅ Kept {len(keep_photos)} photos")
                        if deleted_count > 0:
                            print(f"  🗑️ Will delete: {deleted_count} photos")
                        break
                    elif choice == '3':
                        print("  ⏭️ Skipped unique photos")
                        break
                    else:
                        print("❌ Please enter 1, 2, or 3")

    def cleanup_unwanted_files(self):
        """Remove all unwanted files from the system"""
        print(f"\n🗑️ CLEANING UP UNWANTED FILES")
        print("=" * 40)
        
        # Get list of kept photos
        kept_photos = set(photo.name for photo in self.clean_photos_dir.glob('*.png'))
        
        # Remove unwanted files from original photos_to_scan folder
        photos_to_scan = self.base_dir / "photos_to_scan"
        if photos_to_scan.exists():
            original_photos = list(photos_to_scan.glob('*.png'))
            deleted_count = 0
            
            print(f"🔍 Checking {len(original_photos)} original photos...")
            
            for photo in original_photos:
                # Check if this photo was kept (by name)
                if photo.name not in kept_photos:
                    try:
                        photo.unlink()  # Delete the file
                        deleted_count += 1
                        print(f"  🗑️ Deleted: {photo.name}")
                    except Exception as e:
                        print(f"  ⚠️ Error deleting {photo.name}: {e}")
                else:
                    print(f"  ✅ Kept: {photo.name}")
            
            print(f"\n🎯 Cleanup Summary:")
            print(f"  📁 Original photos: {len(original_photos)}")
            print(f"  ✅ Photos kept: {len(original_photos) - deleted_count}")
            print(f"  🗑️ Photos deleted: {deleted_count}")

    def generate_cleanup_report(self):
        """Generate cleanup report"""
        report_file = self.base_dir / f"CLEANUP_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("INTELLIGENT PHOTO CLEANUP REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Cleanup Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Clean Photos Folder: {self.clean_photos_dir}\n\n")
            
            f.write("CLEANUP STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total Original Photos: {self.stats['total_original']}\n")
            f.write(f"Photos Kept: {self.stats['total_kept']}\n")
            f.write(f"Photos Deleted: {self.stats['total_deleted']}\n")
            f.write(f"Groups Processed: {self.stats['groups_processed']}\n")
            
            space_saved_mb = self.stats['total_deleted'] * 0.5  # Estimate 0.5MB per screenshot
            f.write(f"Estimated Space Saved: {space_saved_mb:.1f} MB\n\n")
            
            f.write("RESULTS:\n")
            f.write("-" * 12 + "\n")
            f.write(f"✅ Your clean photo collection is in: {self.clean_photos_dir}\n")
            f.write(f"🗑️ Unwanted duplicates and similar photos have been removed\n")
            f.write(f"💾 You have saved storage space and organized your collection\n\n")
            
            f.write("NEXT STEPS:\n")
            f.write("-" * 15 + "\n")
            f.write("1. Review your clean photos in the CLEAN_PHOTOS folder\n")
            f.write("2. Back up your clean collection\n")
            f.write("3. The system has been cleaned of unwanted duplicates\n")
        
        print(f"📄 Cleanup report generated: {report_file}")

    def run_interactive_cleanup(self):
        """Run the interactive cleanup process"""
        print("🧹 INTELLIGENT PHOTO CLEANUP TOOL")
        print("=" * 60)
        print("This tool will help you clean up your photo collection by:")
        print("• Removing unwanted duplicates and similar photos")
        print("• Creating a clean collection of your chosen photos")
        print("• Freeing up storage space")
        print()
        
        # Count original photos
        self.stats['total_original'] = self.count_all_photos()
        print(f"📊 Found {self.stats['total_original']} organized photos to review")
        
        input("\n🚀 Press Enter to start the interactive cleanup process...")
        
        # Process similarity groups
        self.process_similarity_groups()
        
        # Process unique photos
        self.process_unique_photos()
        
        # Show summary before cleanup
        print(f"\n📋 CLEANUP SUMMARY:")
        print(f"  ✅ Photos to keep: {self.stats['total_kept']}")
        print(f"  🗑️ Photos to delete: {self.stats['total_deleted']}")
        print(f"  📁 Clean photos location: {self.clean_photos_dir}")
        
        # Confirm cleanup
        confirm = input(f"\n⚠️ DELETE {self.stats['total_deleted']} unwanted photos from your system? (yes/no): ").strip().lower()
        
        if confirm in ['yes', 'y']:
            self.cleanup_unwanted_files()
            self.generate_cleanup_report()
            
            print(f"\n🎉 CLEANUP COMPLETE!")
            print(f"=" * 30)
            print(f"✅ Kept {self.stats['total_kept']} photos in: {self.clean_photos_dir}")
            print(f"🗑️ Deleted {self.stats['total_deleted']} unwanted photos")
            print(f"💾 Your system is now cleaned and organized!")
            print(f"\n💡 Your clean photo collection is ready to use!")
        else:
            print(f"\n⏭️ Cleanup cancelled. Your photos remain unchanged.")
            print(f"💡 Your selected photos are available in: {self.clean_photos_dir}")


def main():
    """Main entry point"""
    try:
        cleanup_tool = PhotoCleanupTool()
        cleanup_tool.run_interactive_cleanup()
    except KeyboardInterrupt:
        print("\n⚠️ Cleanup interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during cleanup: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
