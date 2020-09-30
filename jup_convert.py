#!/home/ch215616/miniconda2/envs/tch1/bin/python

"""
A script for converting jupyter notebook files into .html and .py format, such that they could be committed on git. 

The script will convert all .ipynb files in a given directory. 

Args: 
    directory (str): command line argument specifying the directory in which the .ipynb files are to be searched 

"""

import sys 
import glob 
import subprocess 
import os 

# wrap the first argument in the command line as directory with the .ipynb files 
args = sys.argv
if len(args)<2: 
    print("Please specify input directory.")
    sys.exit()
elif args[1].startswith('.'): 
    args = args[1]+"/"
elif not args[1].endswith('/'): 
    args = args[1]+"/"
else:
    args = args[1]

for file in glob.glob(args+"*.ipynb"): 
    
    #define and create output dirs 
    output_dir,_ = os.path.split(file)
    html_dir = output_dir+"/ipynb_html"
    py_dir = output_dir+"/ipynb_py"
    print(f"writing to: {html_dir} and {py_dir}")
    os.makedirs(html_dir,exist_ok=True)
    os.makedirs(py_dir,exist_ok=True)    
    
    # convert to html 
    cmd=["jupyter", "nbconvert",file,"--to","html","--output-dir",html_dir,"--output", file.replace('.ipynb','.html')]
    print(' '.join(cmd))
    subprocess.call(cmd)
    # convert to python 
    cmd=["jupyter", "nbconvert",file,"--to","python","--output-dir",py_dir,"--output", file.replace('.ipynb','.py')]
    subprocess.call(cmd)
