import numpy as np
import pandas as pd 
from sklearn.pipeline import Pipeline
from us_visa.logger import logging
from us_visa.exception import USvisaException

class TargetValueMapping:
    def __init__(self):
        self.Certified:int=0
        self.Denied:int=1

    def _asdict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_reponse =  self._asdict()
        return dict(zip(mapping_reponse.values(),mapping_reponse.keys()))
        