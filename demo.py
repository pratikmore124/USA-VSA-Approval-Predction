from us_visa.pipeline.training_pipeline import TrainPipeline
from us_visa.exception import USvisaException
import sys
#pipeline = TrainPipeline()
#pipeline.run_pipeline()

try:
    print(1/0)
except Exception as e:
    raise USvisaException(e,sys) from sys