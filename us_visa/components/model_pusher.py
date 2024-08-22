import sys

from us_visa.cloud_storage.aws_storage import SimpleStorageService
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.artifact_entity import ModelPusherArtifact,ModelEvaluationArtifact
from us_visa.entity.config_entity import ModelEvaluationConfig,ModelPusherConfig
from us_visa.entity.s3_estimator import USvisaEstimator

class ModelPusher:
    def __init__(self,model_evaluation_artifact:ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

        self.s3 = SimpleStorageService()
        self.usvisa_estimator = USvisaEstimator(bucket_name=model_pusher_config.bucket_name,
                                                model_path=model_pusher_config.s3_model_key_path)
        
    def initiate_model_pusher(self)->ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model pusher artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        logging.info("Starting Model pusher method from ModelPusher class")

        try:
            logging.info("Uploading artifact to the S3 bucket")
            
            self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.usvisa_estimator.bucket_name,
                                                      s3_model_path=self.model_pusher_config.s3_model_key_path)
            
            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
        except Exception as e:
            raise USvisaException(e,sys)