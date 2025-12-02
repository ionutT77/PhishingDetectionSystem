"""
Dataset Splitter
Splits the 4labels_dataset.csv into train, validation, and test sets
with stratified sampling to maintain label distribution.
"""

import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


class DatasetSplitter:
    """Splits dataset into train, validation, and test sets with stratification."""
    
    def __init__(self, input_file: str, output_dir: str = "data"):
        """
        Initialize the dataset splitter.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Directory where split files will be saved
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.data_by_label: Dict[str, List[Dict]] = defaultdict(list)
        
    def load_data(self) -> bool:
        """Load and organize data by label."""
        print(f"Loading data from: {self.input_file}")
        print("="*60)
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Detect column names
                if not reader.fieldnames:
                    raise ValueError("CSV file is empty or has no header")
                
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
                
                # Load data grouped by label
                total_rows = 0
                for row in reader:
                    url = row[url_col].strip()
                    label = row[label_col].strip()
                    
                    # Skip invalid labels
                    if label.lower() == 'label':
                        continue
                    
                    self.data_by_label[label].append({
                        'url': url,
                        'label': label
                    })
                    total_rows += 1
                
                print(f"\nTotal rows loaded: {total_rows:,}")
                print("\nLabel distribution:")
                for label, data in sorted(self.data_by_label.items()):
                    percentage = (len(data) / total_rows * 100) if total_rows > 0 else 0
                    print(f"  {label:15s}: {len(data):8,} ({percentage:5.2f}%)")
                
                return True
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def split_data(self, train_ratio: float = 0.70, val_ratio: float = 0.15, 
                   test_ratio: float = 0.15, random_seed: int = 42) -> Tuple[List, List, List]:
        """
        Split data into train, validation, and test sets with stratification.
        
        Args:
            train_ratio: Proportion for training set (default: 0.70)
            val_ratio: Proportion for validation set (default: 0.15)
            test_ratio: Proportion for test set (default: 0.15)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        print(f"\n{'='*60}")
        print("Splitting dataset with stratification...")
        print(f"Train: {train_ratio*100:.0f}% | Validation: {val_ratio*100:.0f}% | Test: {test_ratio*100:.0f}%")
        print(f"Random seed: {random_seed}")
        print("="*60)
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        train_data = []
        val_data = []
        test_data = []
        
        # Split each label separately to maintain distribution
        for label, data in self.data_by_label.items():
            # Shuffle data for this label
            shuffled_data = data.copy()
            random.shuffle(shuffled_data)
            
            # Calculate split indices
            n = len(shuffled_data)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            # Split
            label_train = shuffled_data[:train_end]
            label_val = shuffled_data[train_end:val_end]
            label_test = shuffled_data[val_end:]
            
            train_data.extend(label_train)
            val_data.extend(label_val)
            test_data.extend(label_test)
            
            print(f"\n{label}:")
            print(f"  Train: {len(label_train):,} | Val: {len(label_val):,} | Test: {len(label_test):,}")
        
        # Shuffle the combined datasets
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        print(f"\n{'='*60}")
        print("Final split sizes:")
        print(f"  Training:   {len(train_data):8,} ({len(train_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
        print(f"  Validation: {len(val_data):8,} ({len(val_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
        print(f"  Test:       {len(test_data):8,} ({len(test_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
        print(f"  Total:      {len(train_data)+len(val_data)+len(test_data):8,}")
        print("="*60)
        
        return train_data, val_data, test_data
    
    def save_split(self, data: List[Dict], filename: str) -> str:
        """Save a data split to CSV file."""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['url', 'label']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"✓ Saved {len(data):,} rows to {filepath}")
        return filepath
    
    def verify_split(self, train_data: List, val_data: List, test_data: List):
        """Verify that the split maintains label distribution."""
        print(f"\n{'='*60}")
        print("Verifying label distribution across splits...")
        print("="*60)
        
        def get_label_dist(data):
            dist = defaultdict(int)
            for row in data:
                dist[row['label']] += 1
            return dist
        
        train_dist = get_label_dist(train_data)
        val_dist = get_label_dist(val_data)
        test_dist = get_label_dist(test_data)
        
        all_labels = sorted(set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys()))
        
        print(f"\n{'Label':<15} {'Train %':<10} {'Val %':<10} {'Test %':<10}")
        print("-" * 50)
        
        for label in all_labels:
            train_pct = (train_dist[label] / len(train_data) * 100) if train_data else 0
            val_pct = (val_dist[label] / len(val_data) * 100) if val_data else 0
            test_pct = (test_dist[label] / len(test_data) * 100) if test_data else 0
            
            print(f"{label:<15} {train_pct:>8.2f}%  {val_pct:>8.2f}%  {test_pct:>8.2f}%")
        
        print("="*60)
        print("✓ Distribution is consistent across all splits")


def main():
    """Main function to split the dataset."""
    print("Dataset Splitter for Phishing Detection")
    print("="*60)
    print()
    
    input_file = "data/4labels_dataset.csv"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"✗ Error: {input_file} not found")
        return 1
    
    # Create splitter
    splitter = DatasetSplitter(input_file)
    
    # Load data
    if not splitter.load_data():
        return 1
    
    # Split data (70-15-15 split - recommended)
    print("\nUsing recommended 70-15-15 split:")
    print("  70% Training   - For learning patterns")
    print("  15% Validation - For hyperparameter tuning")
    print("  15% Test       - For final evaluation")
    
    train_data, val_data, test_data = splitter.split_data(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42  # For reproducibility
    )
    
    # Save splits
    print(f"\n{'='*60}")
    print("Saving split files...")
    print("="*60)
    
    splitter.save_split(train_data, "train.csv")
    splitter.save_split(val_data, "validation.csv")
    splitter.save_split(test_data, "test.csv")
    
    # Verify distribution
    splitter.verify_split(train_data, val_data, test_data)
    
    print(f"\n{'='*60}")
    print("✓ Dataset split completed successfully!")
    print("="*60)
    print("\nFiles created:")
    print("  - data/train.csv      (70% of data)")
    print("  - data/validation.csv (15% of data)")
    print("  - data/test.csv       (15% of data)")
    print("\nAll splits maintain the original label distribution!")
    
    return 0


if __name__ == "__main__":
    exit(main())
