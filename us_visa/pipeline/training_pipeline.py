import sys

from us_visa.components.data_ingestion import DataIngestion
from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging


class TrainPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self)-> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """

        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongoDB")

            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Got the train set and test set from mongoDB")
            logging.info("Exited the start_data_ingestion method of TrainPipelin class")

            return data_ingestion_artifact

        except Exception as e:
            raise USvisaException(e,sys)

    def run_pipeline(self)-> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """

        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise USvisaException(e,sys)