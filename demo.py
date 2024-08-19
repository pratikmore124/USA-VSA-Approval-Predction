from us_visa.pipeline.training_pipeline import TrainPipeline
from us_visa.exception import USvisaException
import sys

from us_visa.utils.main_utils import read_yaml_file
from us_visa.constants import SCHEMA_FILE_PATH
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
import pandas as pd
'''
df = pd.read_csv(DataIngestionArtifact.trained_file_path)
print(len(df))
a = read_yaml_file(file_path=SCHEMA_FILE_PATH)
print(len(a["columns"]))
'''
pipeline = TrainPipeline()
pipeline.run_pipeline()

