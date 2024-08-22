from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact,DataValidationArtifact,ModelEvaluationArtifact,ModelTrainerArtifact
from sklearn.metrics import f1_score

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.constants import TARGET_COLUMN,CURRENT_YEAR

import sys
import pandas as pd
from typing import Optional

from us_visa.entity.s3_estimator import USvisaEstimator
from dataclasses import dataclass
from us_visa.entity.estimator import USvisaModel
from us_visa.entity.estimator import TargetValueMapping

@dataclass 
class EvaluateModelResponse:
    trained_model_f1_score:float
    base_model_f1_score:float
    is_model_accepted:bool
    difference: float

class ModelEvaluation:

    def __init__(self,model_eval_config:ModelEvaluationConfig,data_ingestion_artifact:DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise USvisaException(e,sys)
        

    def get_best_model(self)->Optional[USvisaEstimator]:
        """
        Method name : get_best_model
        Description : This function is used to get model in production. If there is any model in production it will return that model else None 

        Output      : Return model object if available in S3 bucket
        On Failure  : Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            usvisa_estimator = USvisaEstimator(bucket_name=bucket_name,model_path=model_path)

            if(usvisa_estimator.is_model_present(model_path=model_path)):
                return usvisa_estimator
            
            return None

        except Exception as e:
            raise USvisaException(e,sys) from e
        
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name : evaluate_model
        Description : This method is used to evaluate trained model with production model and choose best model

        Output      : Return bool value based on validation result
        On Failure  : Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df["company_age"] = CURRENT_YEAR-test_df["yr_of_estab"]

            x,y =  test_df.drop(TARGET_COLUMN,axis=1),test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict())

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            base_model_f1_score = None
            base_model = self.get_best_model()

            if(base_model is not None):
                y_hat_base_model = base_model.predict(x)     # get predicted value of test data for S3 bucket model
                base_model_f1_score = f1_score(y,y_hat_base_model)    # get f1 score for s3 bucket model
            
            # Storing the F1 score from base model if present from S3 bucket
            tmp_best_model_score = 0 if base_model_f1_score is None else base_model_f1_score

            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           base_model_f1_score=base_model_f1_score,
                                           is_model_accepted=trained_model_f1_score>tmp_best_model_score,
                                           difference=trained_model_f1_score-tmp_best_model_score)

            return result
        except Exception as e:
            raise Exception(e,sys) from e
        

    def initiate_model_evaluation(self) ->ModelEvaluationArtifact:
        """
        Method Name : Initiate_model_valuation
        Description : This function will initiate all steps of model evaluation

        Output      : Return Model evaluation artifact
        On Failure  : Write exception log then raise an exception 
        """
        logging.info("Starting model evaluation method")

        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                changed_accuracy=evaluate_model_response.difference,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )

            logging.info(f"Model evaluation artifact : {model_evaluation_artifact}")

            return model_evaluation_artifact
        
        except Exception as e:
            raise USvisaException(e,sys) from e