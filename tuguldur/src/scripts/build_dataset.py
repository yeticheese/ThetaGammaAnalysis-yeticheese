import re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io

DATASET_DIR     = "/home/miranjo/common-datasets/LFP_CBD/"
OUTPUT_DIR      = '/home/miranjo/common-datasets/LFP_CBD/rem/'
POSTTRIAL_EXPR  = "*posttrial*.mat"
OVERVIEW_PATH   = '/home/miranjo/common-datasets/LFP_CBD/overview.csv'

"""
This script is based on the Genzel Lab's CBD dataset.
The expected data structure:
    -rat3
        -SD1
        -SD2
        ...
    -rat4
        -SD1
        -SD2
        ...
    ...
    overview.csv

For each study day there are 5 post-trials, and a pre-sleep trial directories. In each post-sleep 
directory there is a LFP file for the hippocampus and states.mat file which contains timestamps for 
REM states. An overview document (.csv format) of study days is used for extracting the metadata 
(condition, treatment).
"""


def find_matching_files(input_directory, expression):
    """
    Find all files in the given directory that match the specified expression.

    Parameters:
        input_directory (Path): The directory to search for matching files.
        expression (str): The pattern to match against file names.

    Returns:
        List of Path objects that match the pattern.
    """
    matching_files = list(input_directory.rglob(expression))
    return matching_files

def map_HPC_files(posttrial_states):
    """
    Map the corresponding HPC files to a list of posttrial state file paths.

    Parameters:
        posttrial_states (list of pathlib.Path): List of file paths to posttrial states.

    Returns:
        dict: A dictionary mapping posttrial state file paths to the corresponding HPC file paths.
    """
    files_connected = {}

    for filepath in posttrial_states:
        dir_path = filepath.parent
        HPC_file = next(dir_path.glob("*HPC*"))
        files_connected[filepath] = HPC_file

    return files_connected

def rem_extract(lfp, sleep_trans):
    """
    Extract REM sleep data from a LFP using sleep transition times.

    Parameters:
        lfp (numpy.ndarray): A NumPy array.
        sleep_trans (numpy.ndarray): A NumPy array containing pairs of sleep transition times.

    Returns:
        list of numpy.ndarray: A list of NumPy arrays, each representing a segment of REM sleep data.
    """
    rems = []

    for rem in sleep_trans:
        t1 = int(rem[0])
        t2 = int(rem[1])
        rems.append(lfp[t1:t2])

    return rems

def extract_REM(HPC_files):
    """
    Extract and save REM sleep data from HPC files.

    Parameters:
        HPC_files (dictionary of pathlib.Path): A dictionary where the keys are Paths to states
            and values are Paths to the LFP recordings.
    """
    #extracting the treatment value from the Study days overview document
    overview_df = pd.read_csv(OVERVIEW_PATH, comment='#')

    for state_file in HPC_files.keys():
        HPC_file = HPC_files[state_file]
        name = create_name(str(HPC_file), overview_df)
        lfp = scipy.io.loadmat(HPC_file)['HPC']
        lfp = np.transpose(lfp)
        sleep = scipy.io.loadmat(state_file)
        transitions = sleep['transitions']

        if np.any(transitions[:, 0] == 5):
            sleep_transitions = transitions[transitions[:, 0] == 5][:, -2:]
            lfp = lfp[0]
            sleep_trans = np.floor(sleep_transitions * 2500)
            REM_file = OUTPUT_DIR + name
            print(f'Saving REM data to: {REM_file}')
            REM = rem_extract(lfp, sleep_trans)
            np.savez(REM_file, *REM)

def create_name(file, overview_df):
    #pattern for matching the information on the rat
    pattern = r'Rat(\d+)_.*_SD(\d+)_([A-Z]+).*posttrial(\d+)'

    # extract the information from the file path
    match = re.search(pattern, file)
    rat_num = int(match.group(1))
    sd_num = int(match.group(2))
    condition = str(match.group(3))
    posttrial_num = int(match.group(4))
    
    if condition == 'HC':
        condition_full = 'HomeCage'
    else:
        condition_full = 'ObjectSpace'

    mask = (overview_df['Rat no.'] == rat_num) & (overview_df['Study Day'] == sd_num) & (overview_df['Condition'] == condition)

    # use boolean indexing to extract the Treatment value
    treatment_value = overview_df.loc[mask, 'Treatment'].values[0]
    
    # Extract the value from the "treatment" column of the matching row
    if treatment_value == 0:
        treatment = 'TreatmentNegative'
    else:
        treatment = 'TreatmentPositive'
       
    title_name = 'Rat' + str(rat_num) +'_' + 'SD' + str(sd_num) + '_' + condition +'_' + condition_full + '_' + treatment + '_' + 'posttrial' + str(posttrial_num)
    #RatID,StudyDay,condition,conditionfull, treatment, treatmentfull, posstrial number

    return title_name

if __name__ == '__main__':

    # Search recursively for *posttrial*.mat files in the given dataset directory
    posttrial_states = find_matching_files(input_directory=Path(DATASET_DIR), expression=POSTTRIAL_EXPR)
    
    # Map the posttrial state files with corresponding HPC files
    HPC_files = map_HPC_files(posttrial_states)
    
    # Extract and save REM states
    extract_REM(HPC_files)