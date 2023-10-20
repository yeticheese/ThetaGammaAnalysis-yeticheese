import re
import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path

INPUT_DIR = "REM_Phasic-Tonic/data"
OVERVIEW_PATH = '/home/miranjo/common-datasets/LFP_CBD/overview.csv'
OUTPUT_FILE = '/home/miranjo/common-datasets/LFP_CBD/rem/'

# Utility Functions
def find_matching_files(input_directory, expression):
    """
    Find all files in the given directory that match the specified expression.

    Args:
        input_directory (str or Path): The directory to search for matching files.
        expression (str): The pattern to match against file names.

    Returns:
        List of Path objects that match the pattern.
    """
    input_directory = Path.home() / Path(input_directory)
    matching_files = list(input_directory.rglob(expression))
    return matching_files

def find_HPC_files(posttrial_states):
    """
    Find corresponding HPC files for a list of posttrial state file paths.

    Args:
        posttrial_states (list of pathlib.Path): List of file paths to posttrial states.

    Returns:
        dict: A dictionary mapping posttrial state file paths to corresponding HPC file paths.
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

    Args:
        lfp (numpy.ndarray): A NumPy array.
        sleep_trans (numpy.ndarray): A NumPy array containing pairs of sleep transition times.

    Returns:
        list of numpy.ndarray: A list of NumPy arrays, each representing a segment of REM sleep data.
    """
    REM = []

    for rem in sleep_trans:
        t1 = int(rem[0])
        t2 = int(rem[1])
        REM.append(lfp[t1:t2])

    return REM

def extract_REM(HPC_files):
    """
    Extract and save REM sleep data from HPC files.

    Args:
        HPC_files (dictionary of pathlib.Path): A dictionary where the keys are Paths to states
            and values are Paths to the LFP recordings.
    """
    for state_file in HPC_files.keys():
        HPC_file = HPC_files[state_file]
        name = create_name(str(HPC_file))
        lfp = scipy.io.loadmat(HPC_file)['HPC']
        lfp = np.transpose(lfp)
        sleep = scipy.io.loadmat(str(state_file))
        transitions = sleep['transitions']

        if np.any(transitions[:, 0] == 5):
            sleep_transitions = transitions[transitions[:, 0] == 5][:, -2:]
            lfp = lfp[0]
            sleep_trans = np.floor(sleep_transitions * 2500)
            REM_file = OUTPUT_FILE + name
            print(f'Saving REM data to: {REM_file}')
            REM = rem_extract(lfp, sleep_trans)
            np.savez(REM_file, *REM)

def create_name(file):
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

    #extracting the treatment value from the Study days overview document
    # Read the Excel file
    df = pd.read_csv(OVERVIEW_PATH, comment='#')

    mask = (df['Rat no.'] == rat_num) & (df['Study Day'] == sd_num) & (df['Condition'] == condition)

    # use boolean indexing to extract the Treatment value
    treatment_value = df.loc[mask, 'Treatment'].values[0]
    
    # Extract the value from the "treatment" column of the matching row
    if treatment_value == 0:
        treatment = 'TreatmentNegative'
    else:
        treatment = 'TreatmentPositive'
       
    title_name = 'Rat' + str(rat_num) +'_' + 'SD' + str(sd_num) + '_' + condition +'_' + condition_full + '_' + treatment + '_' + 'posttrial' + str(posttrial_num)
    #RatID,StudyDay,condition,conditionfull, treatment, treatmentfull, posstrial number
    return title_name
