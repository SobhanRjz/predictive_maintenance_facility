"""Script to convert Excel files to CSV with proper naming and column translation."""
from src.utils.file_renamer import convert_excel_to_csv, FileRenamer


def main():
    """Convert Excel files to CSV."""
    print("Converting Excel files to CSV...")
    print("="*50)
    
    # Initialize file renamer for translation
    
    # Convert Excel files
    converted_files = convert_excel_to_csv(
        source_dir='datasets',
        target_dir='datasets_converted',
    )
    
    print("="*50)
    print(f"Conversion complete! {len(converted_files)} files converted:")
    for file_path in converted_files:
        print(f"  - {file_path}")
    
    print("\nConverted files are saved in 'datasets_converted/' directory")
    print("You can now use these CSV files for the main preprocessing pipeline.")


if __name__ == '__main__':
    main()
