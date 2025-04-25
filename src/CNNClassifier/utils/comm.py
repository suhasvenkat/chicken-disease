import os
from box.exceptions import BoxValueError
import yaml
from CNNClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import pandas as pd

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns args:path_to_yaml(Str):path like input
    returns: ConfigBox :confibox type"""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations     
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data
    """     
    data = json.load(open(path))
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(data)

@ensure_annotations
def get_size(path: Path) -> str:    
    """get size of the file
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations 
def decodeImage(imgstring, filename):
    imgdata = base64.b64decode(imgstring)
    with open(filename, "wb") as f:
        f.write(imgdata)
    logger.info(f"Decoded image from base64 to {filename}") 

@ensure_annotations
def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as image_file:
        return base64.b64encode(image_file.read())
    
@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file
    """
    data = pickle.dumps(data)
    with open(path, "wb") as f:
        f.write(data)   
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary file
    """ 
    data = pickle.loads(path.read_bytes())
    logger.info(f"binary file loaded successfully from: {path}")
    return data 

@ensure_annotations
def get_df(path: Path) -> pd.DataFrame:
    """get dataframe
    """
    df  = pd.read_csv(path)
    logger.info(f"csv file loaded successfully from: {path}")
    return df


