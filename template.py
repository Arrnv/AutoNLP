import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project = 'Mobile_Device_Usage'

list_of_file =[
    # ".github/workflow/.gitkeep",
    f"src/{project}/__inti__.py",
    f"src/{project}/components/__init__.py",
    f"src/{project}/components/data_ingestion.py",
    f"src/{project}/components/data_transformation.py",
    f"src/{project}/components/model_trainier.py",
    f"src/{project}/components/model_monitering.py",
    f"src/{project}/pipeline/__init__.py",
    f"src/{project}/pipeline/traning_pipeline.py",
    f"src/{project}/pipeline/prediction_pipeline.py",
    f"src/{project}/exception.py",
    f"src/{project}/logging.py",
    f"src/{project}/utils.py",
    "app.py",
    "Dockerfile",
    "setup.py",
    "requirements.txt"
]


for filepath in list_of_file:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directories: {filedir} for the file {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating emptyfile {filepath}")
            
    else:
        logging.info(f"{filename} already exist")