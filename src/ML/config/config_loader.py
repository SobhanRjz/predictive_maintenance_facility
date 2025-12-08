"""YAML model configuration loader with validation."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import yaml


@dataclass
class TargetConfig:
    """Target variable configuration."""
    column: str
    classes: list[str]
    type: str


@dataclass
class DataFilterConfig:
    """Data filtering configuration for hierarchical models."""
    column: str
    operator: str
    value: str


@dataclass
class FeaturesConfig:
    """Feature extraction configuration."""
    window_size: str
    extract_features: bool
    feature_types: list[str]
    timestamp_col: str = 'timestamp'
    exclude_cols: list[str] = None
    
    def __post_init__(self):
        """Set default exclude_cols if None."""
        if self.exclude_cols is None:
            self.exclude_cols = ['run_id']


@dataclass
class DatasetConfig:
    """Dataset splitting and preprocessing configuration."""
    datasets_dir: str
    output_dir: str
    use_timeseries_split: bool
    use_stratification: bool
    test_size: float
    validation_size: Optional[float]
    random_state: int
    resample_freq: str
    resample_method: str
    add_hierarchical_labels: bool
    normal_files: list[str]
    failure_files: list[str]
    warning_files: list[str]
    
    def get_normal_paths(self) -> list[str]:
        """Get full paths to normal condition files."""
        from pathlib import Path
        if self.normal_files != None and len(self.normal_files) > 0:
            return [str(Path(self.datasets_dir) / f) for f in self.normal_files]
        else:
            return []
    
    def get_abnormal_paths(self) -> list[str]:
        """Get full paths to failure and warning files."""
        from pathlib import Path
        if self.warning_files == None:
            self.warning_files = []
        if self.failure_files == None:
            self.failure_files = []
       
        all_files = self.failure_files + self.warning_files
        return [str(Path(self.datasets_dir) / f) for f in all_files]



@dataclass
class OutputConfig:
    """Model output paths configuration."""
    model_path: str
    metrics_path: str
    predictions_path: str


@dataclass
class ModelConfig:
    """Complete model configuration from YAML."""
    name: str
    type: str
    task: str
    description: str
    hyperparameters: dict[str, Any]
    target: TargetConfig
    data_filter: Optional[DataFilterConfig]
    features: FeaturesConfig
    dataset: DatasetConfig
    output: OutputConfig


class ConfigLoader:
    """Loads and validates YAML model configurations."""
    
    @staticmethod
    def load(config_path: str | Path) -> ModelConfig:
        """
        Load model configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Parsed ModelConfig object
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        model_data = raw_config['model']
        
        # Parse nested configurations
        target_config = TargetConfig(**model_data['target'])
        
        data_filter_config = None
        if model_data['data_filter'] is not None:
            data_filter_config = DataFilterConfig(**model_data['data_filter'])
        
        features_data = model_data['features']
        features_config = FeaturesConfig(
            window_size=features_data['window_size'],
            extract_features=features_data['extract_features'],
            feature_types=features_data['feature_types'],
            timestamp_col=features_data.get('timestamp_col', 'timestamp'),
            exclude_cols=features_data.get('exclude_cols', ['run_id'])
        )
        
        dataset_data = model_data['dataset']
        dataset_config = DatasetConfig(
            datasets_dir=dataset_data.get('datasets_dir', 'datasets\\datasets_renamed'),
            output_dir=dataset_data.get('output_dir', 'datasets\\processed_data'),
            use_timeseries_split=dataset_data['use_timeseries_split'],
            use_stratification=dataset_data['use_stratification'],
            test_size=dataset_data['test_size'],
            validation_size=dataset_data['validation_size'],
            random_state=dataset_data['random_state'],
            resample_freq=dataset_data['resample_freq'],
            resample_method=dataset_data['resample_method'],
            add_hierarchical_labels=dataset_data.get('add_hierarchical_labels', True),
            normal_files=dataset_data.get('normal_files', []),
            failure_files=dataset_data.get('failure_files', []),
            warning_files=dataset_data.get('warning_files', [])
        )
        
        output_config = OutputConfig(**model_data['output'])
        
        return ModelConfig(
            name=model_data['name'],
            type=model_data['type'],
            task=model_data['task'],
            description=model_data['description'],
            hyperparameters=model_data['hyperparameters'],
            target=target_config,
            data_filter=data_filter_config,
            features=features_config,
            dataset=dataset_config,
            output=output_config
        )
