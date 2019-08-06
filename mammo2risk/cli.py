# -*- coding: utf-8 -*-

"""Console script for mammo2risk."""
import sys
import click
from mammo2risk.facade import MammoRiskManager
from glob import glob
import os 
from os.path import expanduser
from pathlib import Path

@click.command()
@click.option('--d', help='dicom directory')
@click.option('--f', help='file directory')
@click.option('--o', default=".", help="output directory")
@click.option('--w', help="weight directory")
def main(d, f, o, w):
    """Console script for mammo2risk."""
    if w is None:
      w = expanduser("~/mammo2risk/weights")
      
    model_config  = MammoRiskManager.get_config(w)
    mammo_manager = MammoRiskManager(**model_config)
    
    if d:
      files = get_dicom_files(d)
    elif f :
      files = [f]
    else: 
      d = os.path.abspath('') # current path
      files = get_dicom_files(d)
    
    result = mammo_manager.mammo2risk(files)
    result.to_csv(o+"/result.csv", index=False)
    return 0

def get_dicom_files(d): 
    files = Path(d).glob('**/*.dcm')
    if len(files) == 0: 
      print("No dicom files")
    return files
  
if __name__ == "__main__":
    sys.exit(main())
