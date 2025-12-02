"""
Remove Duplicates from Dataset
Removes duplicate URLs from the 4labels_dataset.csv file while preserving the original label.
"""

import csv
import os
from collections import OrderedDict
from typing import Dict, List


def remove_duplicates(input_file: str, output_file: str = None) -> Dict:
    """
    Remove duplicate URLs from CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: overwrites input file)
        
    Returns:
        Dictionary with statistics about the operation
    """
    if output_file is None:
        output_file = input_file
    
    print(f"Removing duplicates from: {input_file}")
    print("="*60)
    
    # Read the data
    seen_urls = OrderedDict()  # Preserves insertion order
    duplicate_count = 0
    total_rows = 0
    label_stats_before = {}
    label_stats_after = {}
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            
            # Check if required columns exist
            if not reader.fieldnames:
                raise ValueError("CSV file is empty or has no header")
            
            # Detect column names (handle variations)
            url_col = None
            label_col = None
            
            for col in reader.fieldnames:
                col_lower = col.lower().strip()
                if col_lower in ['url', 'urls']:
                    url_col = col
                elif col_lower in ['label', 'type', 'category']:
                    label_col = col
            
            if not url_col or not label_col:
                raise ValueError(f"CSV must contain URL and label columns. Found: {reader.fieldnames}")
            
            print(f"Detected columns: URL='{url_col}', Label='{label_col}'")
            print()
            
            # Process rows
            for row in reader:
                total_rows += 1
                url = row[url_col].strip()
                label = row[label_col].strip()
                
                # Count labels before deduplication
                label_stats_before[label] = label_stats_before.get(label, 0) + 1
                
                # Check for duplicates
                if url in seen_urls:
                    duplicate_count += 1
                    # Keep the first occurrence (could also implement logic to prefer certain labels)
                else:
                    seen_urls[url] = label
                    label_stats_after[label] = label_stats_after.get(label, 0) + 1
        
        print(f"Original rows: {total_rows:,}")
        print(f"Unique URLs: {len(seen_urls):,}")
        print(f"Duplicates found: {duplicate_count:,}")
        print()
        
        # Write deduplicated data
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            fieldnames = ['url', 'label']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for url, label in seen_urls.items():
                writer.writerow({'url': url, 'label': label})
        
        print(f"✓ Successfully removed duplicates")
        print(f"✓ Output saved to: {output_file}")
        print()
        
        # Print statistics
        print("Label Distribution BEFORE deduplication:")
        print("-" * 40)
        for label, count in sorted(label_stats_before.items()):
            percentage = (count / total_rows * 100) if total_rows > 0 else 0
            print(f"  {label:15s}: {count:8,} ({percentage:5.2f}%)")
        
        print()
        print("Label Distribution AFTER deduplication:")
        print("-" * 40)
        unique_count = len(seen_urls)
        for label, count in sorted(label_stats_after.items()):
            percentage = (count / unique_count * 100) if unique_count > 0 else 0
            print(f"  {label:15s}: {count:8,} ({percentage:5.2f}%)")
        
        print()
        print(f"Reduction: {duplicate_count:,} rows ({(duplicate_count/total_rows*100):.2f}%)")
        
        return {
            'original_rows': total_rows,
            'unique_rows': len(seen_urls),
            'duplicates_removed': duplicate_count,
            'labels_before': label_stats_before,
            'labels_after': label_stats_after
        }
        
    except FileNotFoundError:
        print(f"✗ Error: File not found: {input_file}")
        raise
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


def main():
    """Main function to remove duplicates from 4labels_dataset.csv."""
    print("Dataset Duplicate Remover")
    print("="*60)
    print()
    
    input_file = "data/4labels_dataset.csv"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"✗ Error: {input_file} not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        return 1
    
    # Get file size before
    original_size = os.path.getsize(input_file)
    print(f"Original file size: {original_size:,} bytes")
    print()
    
    # Remove duplicates
    try:
        stats = remove_duplicates(input_file)
        
        # Get file size after
        new_size = os.path.getsize(input_file)
        size_reduction = original_size - new_size
        reduction_percent = (size_reduction / original_size) * 100 if original_size > 0 else 0
        
        print()
        print("="*60)
        print("File size after deduplication:")
        print(f"  Before: {original_size:,} bytes")
        print(f"  After:  {new_size:,} bytes")
        print(f"  Saved:  {size_reduction:,} bytes ({reduction_percent:.1f}%)")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Failed to remove duplicates: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
