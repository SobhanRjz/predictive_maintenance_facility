"""Data export implementations."""
from pathlib import Path
import pandas as pd
from src.core.interfaces import IDataExporter


class CSVExporter(IDataExporter):
    """Exports data to CSV files."""
    
    def export(self, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_path / 'train.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)

