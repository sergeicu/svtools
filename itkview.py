#!/usr/bin/env python

"""
Show images in ITKsnap with specific input parameters. Helper python function to view images quickly 

"""


import argparse
import subprocess
import nrrd 

import svtools as sv 


def load_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='root directory with files')    
    parser.add_argument('--test_dirs', type=str,nargs='+', required=True, help='names of experiment directories to access ')        
    parser.add_argument('--suffix', type=str, help='suffix to append after experiment directory names')
    parser.add_argument('--image_types', nargs='+', required=True, type=str, help='list names of files to show')        
    parser.add_argument('--ext', default='.nrrd', help='change extension type')            
    parser.add_argument('--verbose', action='store_true', help='prints the string being printed')            
    args = parser.parse_args()    
    return args    

if __name__ == '__main__': 
    
    opt = load_args()
    
    # add a slash to path if missing 
    opt.root = opt.root + "/" if not opt.root.endswith("/") else opt.root 
    # avoid error if suffix is empty 
    if opt.suffix is None: 
        opt.suffix = ""
    
    # if image types is part of the enum - add them to the list of images to be displayed 
    images = []
    for image in opt.image_types: 
        if image=='S0':
            images.append('meanB0_1'+opt.ext) 
        elif image=='D':
            images.append('meanADC_1'+opt.ext) 
        elif image=='P':
            images.append('meanPER_1'+opt.ext) 
        elif image=='Pf' or image=='PF':
            images.append('meanPER_FRAC_1'+opt.ext) 
        else: 
            images.append(image) 

    cmd = ["itksnap","-g"]
    i = 0 
    # construct string to pass to subprocess 
    for test in opt.test_dirs:
        for image in images: 

            # append to command string 
            if i==1: 
                cmd.append("-o")
                cmd.append(opt.root+test+"/"+opt.suffix+"/"+image)
            else: 
                cmd.append(opt.root+test+"/"+opt.suffix+"/"+image)
            i=i+1 

    if opt.verbose: 
        print(" ".join(cmd))    
    # call bash subprocess 
    subprocess.call(cmd)
   

