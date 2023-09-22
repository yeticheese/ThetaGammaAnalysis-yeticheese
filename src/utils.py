import yaml
import os
import re
import itertools
import scipy.io as sio
import inspect
import h5py
import numpy as np
from .logger import *
from pandas import read_csv


def load_config(config_path):
    """Load configuration settings from a YAML file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_files_dict(file_path):
    """
    Recursively navigates through a directory tree and generates a nested dictionary
    representing the file structure maintained at the Genzel Lab Databases,
    along with extracted information from the file paths.

    Args:
        file_path (str): The path to the root directory.

    Returns:
        dict: A nested dictionary representing the file structure and extracted information.
            The keys are directory names, and the values are nested dictionaries or file names
            and extracted information parsed with RegEx from file paths.

    Example:
        file_dict = get_files_dict('/path/to/directory')
        :param file_path:
    """
    file_dict = {}

    if isinstance(file_path, tuple):  # If it's a tuple of paths
        all_iterators = (os.walk(path, topdown=True) for path in file_path)
        walker = itertools.chain(*all_iterators)
    else:  # If it's a single path
        walker = os.walk(file_path, topdown=True)

    for root, dirs, files in walker:
        rat_conditions = re.findall('Rat\d|HC|OR|OD|presleep|SD\d\d|SD\d|posttrial\d', root)
        if len(rat_conditions) < 4:
            continue

        file_dir_parts = root.split(os.sep)[7:]
        for file in files:
            state = bool(re.search('states.*.mat', file))
            hpc = bool(re.search(r"\bHPC.*.mat", file))
            if state or hpc:
                file_dir_parts.append(file)

        sub_dict = file_dict
        for part in file_dir_parts[:3]:
            sub_dict = sub_dict.setdefault(part, {})
            # print(sub_dict)

        sub_dict['rat'] = int(re.findall('\d', rat_conditions[0])[0])
        sub_dict['study_day'] = int(re.findall('\d\d|\d', rat_conditions[1])[0])
        sub_dict['condition'] = rat_conditions[2]
        sub_dict['trial'] = rat_conditions[3]
        sub_dict['HPC'] = file_dir_parts[-1]
        sub_dict['states'] = file_dir_parts[-2]
    return file_dict


def process_functions(*functions, **kwargs):
    """
       Process a series of functions sequentially, managing inputs and outputs dynamically.

       Parameters:
       *functions: Variable-length positional argument. A series of functions to process sequentially.
       **kwargs: Keyword argument dictionary with configuration options.
           - 'output_data' (optional): Data passed between functions. Initially None.

       Returns:
       The result of the last function call in the chain.
       """
    kwargs.setdefault('output_data', None)
    for func in functions:
        # Get the function signature
        signature = inspect.signature(func)
        # Get the function parameters
        params = signature.parameters
        if kwargs['output_data'] is None:
            kwargs = kwargs
        else:
            kwargs[str(next(iter(params)))] = kwargs['output_data']
        # Get the remaining keyword arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        # # Call the function with the positional arguments and filtered kwargs
        kwargs['output_data'] = func(**filtered_kwargs)


def write_dict_to_hdf5(file, data, parent_key=''):
    """
    Recursively write a dictionary structure to an HDF5 file.

    Parameters:
    file (h5py.File or h5py.Group): The HDF5 file or group where data will be written.
    data (dict): The dictionary to be written to the HDF5 file.
    parent_key (str, optional): The parent key under which the data will be stored. Default is an empty string.

    Returns:
    None

    Notes:
    - This function recursively iterates through the dictionary structure and writes it to the HDF5 file.
    - If a dictionary value is encountered, it creates a new group in the HDF5 file.
    - If a NumPy array is encountered, it is written as a dataset.
    - If a scalar or list is encountered, it is converted to a dataset and stored.
    """
    for key, value in data.items():
        current_key = f'{parent_key}/{key}' if parent_key else key

        if isinstance(value, dict):
            # If the value is a dictionary, create a new group in the HDF5 file
            group = file.create_group(current_key)
            print(group.name)
            write_dict_to_hdf5(file, value, parent_key=current_key)
        else:
            # If the value is not a dictionary, write it as a dataset in the HDF5 file
            if isinstance(value, np.ndarray):
                # If the value is a NumPy array, convert it to a dataset
                file.create_dataset(current_key, data=value)
            else:
                # If the value is a scalar or list, convert it to a dataset
                file[current_key] = value


def read_hdf5_to_dict(read_filename):
    """
    Recursively read data from an HDF5 file into a dictionary structure.

    Parameters:
    read_filename (str): The name of the HDF5 file to be read.

    Returns:
    dict: A dictionary containing the data read from the HDF5 file.

    Notes:
    - This function recursively traverses the HDF5 file and reads data into a dictionary structure.
    - Groups in the HDF5 file are represented as nested dictionaries.
    - Datasets in the HDF5 file are represented as keys with associated data in the dictionary.
    - NumPy arrays, integers, and string values are appropriately converted and stored in the dictionary.
    """
    def traverse_hdf5_group(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = traverse_hdf5_group(item)
            elif isinstance(item, h5py.Dataset):
                if isinstance(item[()], np.ndarray):
                    result[key] = np.array(item[()])
                elif isinstance(item[()], np.int32):
                    result[key] = int(item[()])
                else:
                    result[key] = item[()].decode()
        return result

    with h5py.File(read_filename, 'r') as file:
        return traverse_hdf5_group(file)


def dict_walk(file_dict, folder_root, read_suffix='', write_suffix='', *functions, **kwargs):
    """
        Walk through a nested dictionary structure and process data using a series of functions.

        Parameters:
        file_dict (dict): A nested dictionary structure containing file information.
        folder_root (str): The root folder where data files are located.
        read_suffix (str, optional): Suffix to be added to read filenames. Default is an empty string.
        write_suffix (str, optional): Suffix to be added to write filenames. Default is an empty string.
        *functions: Variable-length positional argument. A series of functions to process data.
        **kwargs: Keyword argument dictionary with initial and additional configuration options.

        Returns:
        None

        Notes:
        - This function walks through a nested dictionary structure and processes data using a series of functions.
        - It constructs filenames, loads data, and passes it to the specified functions.
        - If an error occurs during processing, it logs the error and continues processing other items.
        """
    logger_file = r'{folder_root}\log.txt'
    logger_file_path = logger_file.format(folder_root=folder_root)
    logger = setup_logger(f'{logger_file_path}')
    CBD_file = r'{folder_root}\CBD_log.csv'
    CBD_logger_file_path = CBD_file.format(folder_root=folder_root)
    cbd_df = read_csv(f'{CBD_logger_file_path}')
    for rat, day_trial_dict in zip(file_dict.keys(), file_dict.values()):
        # Get the last sub-dictionary in the current sub-dictionary
        for day_trial_type, sleep_type_dict in zip(day_trial_dict.keys(), day_trial_dict.values()):
            for sleep_type, last_dict in zip(sleep_type_dict.keys(), sleep_type_dict.values()):
                file_path = r'{folder_root}\{rat}\{day_trial_type}\{sleep_type}\{state}'
                filename_struct = r'{folder_root}\{rat}\{day_trial_type}\{sleep_type}\Rat{rat}_SD{study_day}_{' \
                                  r'condition}_{treatment}_{trial}{suffix}'
                query_string = f"rat == {last_dict['rat']} and study_day == {last_dict['study_day']} and condition ==" \
                               f"'{last_dict['condition']}'"
                cbd_bool = cbd_df.query(query_string)['treatment']
                if bool(cbd_bool.iloc[0]):
                    treatment = 'CBD'
                else:
                    treatment = 'VEH'
                states_file_path = file_path.format(folder_root=folder_root,
                                                    rat=rat,
                                                    day_trial_type=day_trial_type,
                                                    sleep_type=sleep_type,
                                                    state=last_dict['states'])
                HPC_file_path = file_path.format(folder_root=folder_root,
                                                 rat=rat,
                                                 day_trial_type=day_trial_type,
                                                 sleep_type=sleep_type,
                                                 state=last_dict['HPC'])
                states = sio.loadmat(f'{states_file_path}')
                HPC = sio.loadmat(f'{HPC_file_path}')
                read_filename = filename_struct.format(folder_root=folder_root,
                                                       rat=rat,
                                                       day_trial_type=day_trial_type,
                                                       sleep_type=sleep_type,
                                                       study_day=last_dict['study_day'],
                                                       condition=last_dict['condition'],
                                                       treatment=treatment,
                                                       trial=last_dict['trial'],
                                                       suffix=read_suffix)
                write_filename = filename_struct.format(folder_root=folder_root,
                                                        rat=rat,
                                                        day_trial_type=day_trial_type,
                                                        sleep_type=sleep_type,
                                                        study_day=last_dict['study_day'],
                                                        condition=last_dict['condition'],
                                                        treatment=treatment,
                                                        trial=last_dict['trial'],
                                                        suffix=write_suffix)
                if os.path.exists(write_filename):
                    print("Filename already exists")
                    continue
                else:
                    header_dict = {'header_info': {'rat': int(last_dict['rat']),
                                                   'study_day': int(last_dict['study_day']),
                                                   'condition': str(last_dict['condition']),
                                                   'trial': str(last_dict['trial']),
                                                   'treatment': treatment}}
                print(header_dict)
                kwargs.update({'x': HPC['HPC'],
                               'rem_states': states['states'],
                               'header_dict': header_dict,
                               'write_filename': write_filename,
                               'read_filename': read_filename})
                error_file_loc = file_path.format(folder_root=folder_root,
                                                  rat=rat,
                                                  day_trial_type=day_trial_type,
                                                  sleep_type=sleep_type,
                                                  state=last_dict['states'])

                # process_functions(*functions, **kwargs)
                try:
                    process_functions(*functions, **kwargs)
                except Exception as e:
                    # Log the error
                    logger.error(f'Error processing item: {error_file_loc}. {e}')

                    continue
