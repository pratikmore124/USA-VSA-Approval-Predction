from us_visa.configuration.mongo_db_connection import MongoDBClient
from us_visa.constants import DATABASE_NAME
from us_visa.exception import USvisaException
from us_visa.logger import logging
import pandas as pd 
import numpy as np
from typing import Optional

class USvisaData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name = DATABASE_NAME)

        except Exception as e:
            raise USvisaException(e,sys) from e
        
    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None->pd.DataFrame):
        """
            export entire collectin as dataframe:
            return pd.DataFrame of collection
        """
        
        try:
            if database_name in None:
                collection = self.mongo_client.database[collection_name]

            else:
                collection = self.mongo_client[database_name][collection_name]

            
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list:
                df = df.drop(columns=["_id"],axis=1)

            df.replace({"na":np.nan},inplace=True})
            return df
        
        except Exception as e:
            raise USvisaException(e,sys) from e
        
