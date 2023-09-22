import re
import scipy
from pathlib import Path
import pandas as pd
import numpy as np

from neurodsp.filt import filter_signal
from neurodsp.plts import plot_time_series
from neurodsp.sim import sim_combined
# Import utilities for loading and plotting data
from neurodsp.utils import create_times
from neurodsp.utils.download import load_ndsp_data
from neurodsp.plts.time_series import plot_time_series, plot_instantaneous_measure
# Import time-frequency functions
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time

from bycycle.features import compute_features
from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.cyclepoints.zerox import find_flank_zerox
from bycycle.plts import plot_burst_detect_summary, plot_cyclepoints_array
from bycycle.utils.download import load_bycycle_data
from bycycle.utils import get_extrema_df



INPUT_DIR = "REM_Phasic-Tonic/data"
POSTTRIAL = "*posttrial*.mat"
HPC = "*posttrial*/*HPC*.continuous.mat"
VEH = "*VEH*posttrial*/REM_data.mat"
REM = "*REM_data.npz"

fs = 2500

# Filter settings
f_theta = (4, 12)
f_lowpass = 20
n_seconds_filter = .5  # could be changed to 0.1
n_seconds_theta = .75

plot_display = True

threshold_kwargs = {'amp_fraction_threshold': 0.8,
                        'amp_consistency_threshold': 0,
                        'period_consistency_threshold': 0,
                        'monotonicity_threshold': 0,
                        'min_n_cycles': 8}



def find_matching_files(input_directory, expression):

    """
    Find all files in the given directory that match the specified expression.

    Args:
        input_directory (str or Path): The directory to search for matching files.
        expression (str): The pattern to match against file names.

    Returns:
        List of Path objects that match the pattern.
    """
    # Convert input_directory to a Path object if it's a string
    input_directory = Path.home() / Path(input_directory)

    # Search the directory for files matching the pattern
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
    files_connected = {}  # Initialize an empty dictionary to store the mapping

    for filepath in posttrial_states:
        dir_path = filepath.parent  # Get the directory path of the file

        HPC_file = next(dir_path.glob("*HPC*"))  # Find the corresponding HPC files in the same folder as states
        
        # Store the mapping in the dictionary
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
    REM = []  # Initialize a list to store REM sleep segments

    for rem in sleep_trans:
        t1 = int(rem[0])  # Start time index of REM segment
        t2 = int(rem[1])  # End time index of REM segment

        # Extract REM sleep segment from the continuous signal and add to REM list
        REM.append(lfp[t1:t2])

    return REM



def find_REM(HPC_files):
    """
    Extract and save REM sleep data from HPC files.

    Args:
        HPC_files (dictionary of pathlib.Path): A dictionary where the keys are Paths to states
            and values are Paths to the LFP recordings.

    Returns:
        list of str: List of file paths to the saved REM sleep data files.
    """
    REM_files = []  # Initialize a list to store paths to saved REM data files

    for state_file in HPC_files.keys():
        # Load HPC data
        HPC_file = HPC_files[state_file]
        lfp = scipy.io.loadmat(HPC_file)['HPC']
        lfp = np.transpose(lfp)

        # Load sleep data
        sleep = scipy.io.loadmat(str(state_file))
        transitions = sleep['transitions']

        # Check if the file contains REM sleep (state code 5)
        if np.any(transitions[:, 0] == 5):
            # Extract REM sleep transitions
            sleep_transitions = transitions[transitions[:, 0] == 5][:, -2:]
            lfp = lfp[0]
            sleep_trans = np.floor(sleep_transitions * 2500)

            # Define the file path for saving REM data
            REM_file = HPC_file.parent / 'REM_data.npz'
            print(f'Saving REM data to: {REM_file}')

            # Extract REM data using the function rem_extract
            REM = rem_extract(lfp, sleep_trans)

            # Save REM data in the same folder as the original data
            np.savez(REM_file, *REM)

            # Add the file path to the list
            REM_files.append(REM_file)

    return REM_files

def load_data(file):
    container = np.load(file)
    dataREMs = [container[key] for key in container]
    return dataREMs

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
    df = pd.read_excel('CBD_Chronic_treatment_conditions_study days overview.xlsx', skiprows=2)

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

def find_REM():
    return find_matching_files(INPUT_DIR, REM)


def main():
    REM_files = find_matching_files(input_directory=INPUT_DIR, expression=REM)
    for file in REM_files:
        print(file)

if __name__ == '__main__':
    main()