# -*- coding: utf-8 -*-

"""Console script for mammo2risk."""
import sys
import click
sys.path.append(".")
from mammo2risk.facade import MammoRiskManager
from mammo2risk.preprocessing import Preprocessor
from glob import glob
import os 
from os.path import expanduser
from pathlib import Path
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@click.command()
@click.option('--d', help='dicom directory')
@click.option('--f', help='file directory')
@click.option('--o', default=".", help="output directory")
@click.option('--w', help="weight directory")
@click.option('--r', help="recursive", is_flag=True)
@click.option('--save', help="save image", is_flag=True)
@click.option('--g', help="GPU id")

def main(d, f, o, w, r, save, g):
    """Console script for mammo2risk."""
    
    if is_valid(d, f, o, w, r, g) == False:
      print("Mammo2risk stopped.")
      print("Arguments are not valid.")
      return 0
    
    if w is None:
      w = expanduser("~/mammo2risk/weights/config_v1.0.json")

    # directory 
    if d:
      files = get_dicom_files(d, r)

    # file
    elif f :
      files = [f]

    # default
    else: 
      d = os.path.abspath('') # current path
      files = get_dicom_files(d, r)

    if(len(files) == 0):
        print("No files were detected.")
        return 0
     
    # gpu number
    if g: 
      os.environ["CUDA_VISIBLE_DEVICES"] = g

    model_config  = MammoRiskManager.get_config(w)
    mammo_manager = MammoRiskManager(**model_config)
    result = mammo_manager.mammo2risk(files, img_save=save, img_save_path=o)
    print("Mammo2risk has run successfully.")
    
    # Save result csv files
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    output_file_name = "mammo2risk"+current_time+".csv"
    output_path = o+"/"+output_file_name
    output_path = os.path.abspath(output_path)
    print(output_path)
    result.to_csv(output_path, index=False)
    
    return 0

def is_valid(d, f, o, w, r, g) : 
    result = True 
    if (d == None) & (f == None) : 
      print("Either --d or --f option is mandatory.")
      result = False 
    elif (d != None) & (f != None): 
      print("Please choose one of --d or --f option. Both should not be given.")
      result = False
      
    return result 

def get_dicom_files(d, r): 
    result = []
    if r: 
      print("Recursively getting dicom files...")
      files = [str(x) for x in list(Path(d).glob('**/*.dcm'))]
      print(f"{len(files)} files were detected.")
      
    else: 
      files = glob(d+"/*.dcm")
      print(f"{len(files)} files were detected.")
      
    if len(files) == 0: 
      print("No dicom files")
    return files
  
if __name__ == "__main__":
    main()
    os._exit(0)
