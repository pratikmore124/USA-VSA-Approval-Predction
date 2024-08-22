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
    best_model_f1_score:float
    is_model_accepted:bool
    differnce: float

class ModelEvaluation:

    def __init__(self,model_eval_config:ModelEvaluationConfig,data_ingestion_artifact:DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise USvisaException(e,sys)
        

    def get_best_model(Self)->Optional[USvisaEstimator]:
        pass

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name : evaluate_model
        Description : This method is used to evaluate trained model with production model and choose best model

        Output      : Return bool value based on validation result
        On Failure  : Write an exception log and then raise an exception
        """
        try:
            pass

        except Exception as e:
            raise Exception(e,sys)
    def initiate_model_evaluation(self) ->ModelEvaluationArtifact:
        """
        Method Name : Initiate_model_valuation
        Description : This function will initiate all steps of model evaluation

        Output      : Return Model evaluation artifact
        On Failure  : Write exception log then raise an exception 
        """

        try:
            pass
        except Exception as e:
            raise USvisaException(e,sys)