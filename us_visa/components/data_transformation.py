import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from us_visa.exception import USvisaException
from us_visa.logger import logging

from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN,SCHEMA_FILE_PATH,CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from us_visa.utils.main_utils import save_object,save_numpy_array_data,read_yaml_file,drop_columns
from us_visa.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,
                data_validation_artifact: DataValidationArtifact,
                data_transformation_config: DataTransformationConfig,
                ):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            logging.info("Data validation artifact in Transformatin init:"+str(self.data_validation_artifact))
        except Exception as e:
            raise USvisaException(e,sys)
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise USvisaException(e,sys)

    def  get_data_tranformer_object(self):

        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered get_data_tranformer_object method of DataTransformation class")
        
        numberical_transform = StandardScaler()
        oh_transformer = OneHotEncoder()
        or_transformer = OrdinalEncoder()

        logging.info("Initialize Standard Scalar, One hot encoder, Ordinal encoder")

        oh_columns = self._schema_config["oh_columns"]
        or_columns = self._schema_config["or_columns"]
        num_features = self._schema_config["num_features"]
        transform_columns = self._schema_config["transform_columns"]

        logging.info("Initilaizing Powertransformer")

        """
        We perform Power Transform to make data normally distributed. This can improve convergence and performance of ML algorithm.
        """

        transform_pipe = Pipeline(steps=[
            ("transformer",PowerTransformer(method="yeo-johnson"))
        ])

        preprocessing = ColumnTransformer(
            [
                ("OneHotEncoder",oh_transformer,oh_columns),
                ("OrdinalEncoder",or_transformer,oh_columns),
                ("Transformer",transform_pipe,transform_columns),
                ("StandarScaler",numberical_transform,num_features)
            ]
        )

        logging.info("Creating preprocessing object from ColumnTransformer")

        logging.info("Exited get_data_tranformer_object method from ")
        return preprocessing


    def initiate_data_transformation(self)->DataTransformationArtifact:
        """
        Method Name : initiate_data_transformation
        Description : This method initiate the data transformation component for the piplelin

        output      : Data transformation steps are performed and preprocessing object is created
        on Failure  : Write an exception log and then raise an exception
        """

        try:
            
            if(self.data_validation_artifact.validation_status):
                logging.info("Starting Data Transformation ")
                preprocessing = self.get_data_tranformer_object()
                logging.info("Got preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                # split data to independent feature and Target feature for train df
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got train fetaure and test feature of training dataset")

                input_feature_train_df["company_age"] = CURRENT_YEAR-input_feature_train_df["yr_of_estab"]

                logging.info("Adding company age column in input_feature_train_df")

                drop_cols = self._schema_config["drop_columns"] 
                
                logging.info(f"Drop columns in drop_cols:{drop_cols} of training dataset")

                input_feature_train_df = drop_columns(df = input_feature_train_df,cols=drop_cols)


                # split data to independent feature and Target feature for test df
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info("Got train fetaure and test feature of training dataset")

                input_feature_test_df["company_age"] = CURRENT_YEAR-input_feature_test_df["yr_of_estab"]

                logging.info("Adding company age column in input_feature_train_df")

                drop_cols = self._schema_config["drop_columns"] 
                
                logging.info(f"Drop columns in drop_cols:{drop_cols} of training dataset")
                logging.info("df columns:"+str(input_feature_test_df.columns))
                input_feature_test_df = drop_columns(df = input_feature_test_df,cols=drop_cols)

                logging.info("Drop the column in drop_cols of Test Dataset")

                target_feature_test_df = target_feature_test_df.replace(TargetValueMapping()._asdict())

                logging.info("Got train and test feature for  testing dataset")

                logging.info("Applying preprocessing object on training dataframe and testing dataframe")

                input_feature_train_arr = preprocessing.fit_transform(input_feature_train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessing.fit_transform(input_feature_train_df)

                logging.info("Used the preprocessor object to transform the test features")

                logging.info("Applying SMOTEENN on Training dataset")

                """
                SMOTEENN is a technique used to handle class imbalance, which is a common issue when one class is significantly underrepresented compared to another
                """
                smt = SMOTEENN(sampling_strategy="minority")

                input_feature_train_final,target_feature_train_final = smt.fit_resample(input_feature_train_arr,target_feature_train_df)

                logging.info("applied SMOTEENN on training dataset")

                logging.info("Applying SMOTEENN on testing dataset")

                input_feature_test_final,target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr,target_feature_test_df
                )

                logging.info("Applied SMOTEENN on test dataset")

                logging.info("Creating train array and test array")
                
                # np.c_ is short hand for numpy concatination
                train_arr = np.c_[input_feature_train_final,np.array(target_feature_train_final)]
                test_arr = np.c_[input_feature_test_final,np.array(target_feature_test_final)]

                save_object(self.data_transformation_config.transformed_object_file_path,preprocessing),
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)

                logging.info("Existing initiate_data_transformation method of Data_Transformatio class")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path
                )

                logging.info(f"Data Transformation completed. Transformation artifact : {data_transformation_artifact}")
                return data_transformation_artifact
            
            else:
                raise Exception(self.data_validation_artifact.message)
    

        except Exception as e:
            raise USvisaException(e,sys) from e