import os
from pathlib import Path

# This code is used to create a folder structure

# root folder name
Project_name = "USA-VISA-Approval"

# list of folder we need to create
list_of_files = [
    f"{Project_name}/__init__.py", # Root folder with a constructor,
    
    f"{Project_name}/components/__init__.py", # folder for all cicd pipleline code,
    f"{Project_name}/components/data_ingestion.py",   # data ingestion folder
    f"{Project_name}/components/data_validation.py",   # data validation folder
    f"{Project_name}/components/data_transformation.py",   # data transformation folder
    f"{Project_name}/components/model_trainer.py",   # data model trainer folder
    f"{Project_name}/components/model_evaluation.py",   # data model evaluation folder
    f"{Project_name}/components/model_pusher.py",   # data model pusher folder
    
    f"{Project_name}/configuration/__init__.py",   # configuration file
    
    f"{Project_name}/constant/__init__.py",          # for declaring constant values
    
    f"{Project_name}/entity/__init__.py",
    f"{Project_name}/entity/config_entity.py",
    f"{Project_name}/entity/artifact_entity.py",

    f"{Project_name}/exception/__init__.py",

    f"{Project_name}/logger/__init__.py",

    f"{Project_name}/pipeline/__init__.py",
    f"{Project_name}/pipeline/predication_pipeline.py",
    f"{Project_name}/pipeline/training_pipeline.py",

    f"{Project_name}/utils/__init__.py",
    f"{Project_name}/utils/main_utils.py",

    "app.py",
    "requirements.py",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml"
]

# code to create all the file listed in "list_of_files"

for file_path in list_of_files:
    filePath = Path(file_path)
    filedir, filename = os.path.split(filePath)
    if(filedir != ""):
        os.makedirs(filedir,exist_ok=True)
    if(not os.path.exists(file_path)) or os.path.getsize(file_path) == 0:
        with open(file_path,"w") as f:
            pass
    else:
        print(f"File is already prsent at:{file_path}")

print("Required file have been created")