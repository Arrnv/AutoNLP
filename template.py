import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project = 'AutoNLP'

list_of_file =[
    # ".github/workflow/.gitkeep",
    f"src/{project}/__inti__.py",
    f"src/{project}/components/__init__.py",
    f"src/{project}/components/Preprocessing.py",
    f"src/{project}/components/eda.py",
    f"src/{project}/components/model_trainier.py",
    f"src/{project}/pipeline/__init__.py",
    f"src/{project}/pipeline/pipeline.py",
    f"src/{project}/pipeline/visualization.py",
    f"src/{project}/report/reports.py",
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