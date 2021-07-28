""" Convert a 4D diffusion mosaic into individual 3D files with the correct ordering of b-values, and the correct name convention*. 

    4D diffusion mosaic images is an output of DCM2NIIX tool conversion. 

    While the information on b-values that are output with this transaction IS correct, the order of files is not. We are therefore reading .bvec file output to get the ordering correct. 

    *Correct name convention according to CRL / QUIN DWI IVIM processing pipeline, which requires that files are to be named in the following manner: b0_#1.nii.gz, b0_#2.nii.gz...b400_#5.nii.gz
    where b0 refers to b-value and #2 refers to diffusion direction. 
    
    
    Usage: 
        python fix_dwi_bval_ordering.py --filenames <file1> <file2> ... <dir3> --directions 6 
        python fix_dwi_bval_ordering.py --dirname <dir> --directions 6 
        
    Args: 
        --filenames - path to a .nii file (or files) that contains 4D diffusion mosaic (path can be relative if this script is in the same folder as the .nii file, else it must be absolute full path). 
                      IMPORTANT: .nii file must have a corresponding .bval file (output of DCM2NIIX conversion tool)
        --dirname - alternatively to using 'filenames', provide a directory and all files in the directory will be processed
        --directors - number of directions expected to be found (default = 6)
    Output: 
        3D .nii.gz files are saved into a separate directory in the same folder as the .nii file. Each new directory has the same name as each .nii file (this is done to prevent overwriting files when multiple .nii files exist in the same directory). 
    
"""



import os 
import sys 
import glob 
import argparse
import numpy as np
import nibabel as nb

from collections import Counter 

def main():
    
    # load input args
    args = load_args()

    files = []
    
    # load filenames
    if args.filenames is not None:         
        if isinstance(args.filenames, str):
            files.append(args.filenames) #add single filename
        else:
            files.extend(args.filenames) # add list of filenames

        # check that paths exist
        assert all([os.path.exists(f) for f in files])

    else:
        assert args.dirname is not None, "Please specify --filenames or --dirname at input"
        
        # find files 
        files = glob.glob(args.dirname + "/*.nii")
        assert files, f"no .nii files found in {args.dirname}"
        
        
    # process files 
    fix_dwi_bval_ordering(args,files)

    

def fix_dwi_bval_ordering(args,files):
    
    """
    Args:
        files(list): list of filepaths to .nii files
    """
    
    for f in files:
        print(f"\n\n\n\n\nProcessing: {f}")
        
        # get bval file 
        bval_path = f.replace(".nii", ".bval")
        if not os.path.exists(bval_path):
            # skip this file as no .bval file was found for it 
            print(f"No .bval file found for {f}, skipping this file")
            continue
            
        # get bvalues 
        bvals = get_bvector(bval_path)

        # convert nifti file 
        convert_4D_to_3D(f, bvals, args.directions)    
    
    return 
        
def load_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenames',type=str, nargs='+', default=None, help='path to a .nii file (or files) that contains 4D diffusion mosaic. Relative or absolute path')
    parser.add_argument('--dirname',type=str,default=None, help='path to a folder that contains .nii files with 4D diffusion mosaic format. Any non diffusion mosaic .nii files will be ignored. Relative or absolute path')
    parser.add_argument('--directions',type=int,default=6, help='number of diffusion directions expected from the file')
    
    args = parser.parse_args()
    
    return args 
        
        
def get_bvector(bval_path):
    """
    Obtain a list of bvecs from a .bvec file produced as output of dcm2niix conversion
    
    """
    
    with open(bval_path) as f:
        line = f.readline()

    bvals = line.split(' ')
    bvals[-1] = bvals[-1][:-1]
    bvals = [int(i) for i in bvals]

    return bvals
    
def convert_4D_to_3D(impath, original_bvals, directions):
    
    """Convert a 4D diffusion mosaic into individual 3D files. 
    
    4D diffusion mosaic is an output of DCM2NIIX conversion process. 
        
    We submit a correct set of b-values that were given to the scanner at scan time. 
    
    
    WARNING: this process assumes that the ordering of b-values is correct. To check this - please read the following output file that is produced by DCM2NIIX process: <outputname>.bval. You may also want to verify that <outputname>.bvec corresponds to the original bvector file given to the scanner at scan time
    
    Args: 
        imagepath (str): full path to .nii file produced by the DCM2NIIX process 
        original_bvalues (list): list of integers denoting the FULL list of bvalues that correspond to the list of bvectors given to the scanner. E.g. if there were 8 b-values and 6 directions for each, this will be a list of length 8*6 (assuming that the first bvalue, such as b0, was acquired 6 times)
        
    
    
    """
    assert os.path.exists(impath)
    imo = nb.load(impath)
    im = imo.get_fdata()
    
    dirname = os.path.splitext(impath)[0] + "/"
    os.makedirs(dirname,exist_ok=True)

    original_vector = original_bvals
        
    assert len(original_vector) == imo.shape[-1], "length of the original vector must be the same as the image produced by the dcm2nii converter"
    
    # initiate counter to keep track of how many times a particular b-value has been seen (so that we can increment directions)
    c = Counter()

    # build new header (require to decrease the number of directions in t) - this will be the same for all files 
    header = imo.header 
    header['dim'][4] = 1 

    # cycle through each individual file
    for i in range(0,len(original_vector)):

        # get individual image that represents single bvalues and single direction
        im_singleBval_singleDir = im[:,:,:,i]

        # extract bvalue number 
        bvalnum = original_vector[i]    

        # extract the direction number (direction is equal to the number of times that a particular bvalue has already been seen - starting from zero)
        directionnum = c[bvalnum]
        c[bvalnum] += 1  # increment counter 


        # save this image into separate file  in the following format: b<bvalnum>#_<directionnum>.nii.gz 
        savename = dirname + 'b'+str(bvalnum)+"#_"+str(directionnum)+ ".nii.gz"

        # make a nifti image and save
        imnewo = nb.Nifti1Image(im_singleBval_singleDir,affine=imo.affine, header=header)
        nb.save(imnewo, savename)
        # print progress
        print(savename)        
    
    
    
if __name__ == '__main__':
    main()
