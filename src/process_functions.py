from .utils import write_dict_to_hdf5
import h5py
import numpy as np
import pandas as pd


def dict_write(rem_dict, header_dict, write_filename):
    """

    :param rem_dict:
    :param header_dict:
    :param write_filename:
    :return:
    """
    rem_record = header_dict
    rem_record.update(rem_dict)
    file = h5py.File(write_filename, 'w')
    write_dict_to_hdf5(file, rem_record)
    file.close()


def cog_dataset_loader(rem_dict):
    cog = np.empty((0, 2))
    for k, sub_dict in rem_dict.items():
        if k == 'header_info':
            continue
        else:
            for key, value in sub_dict.items():
                if key == 'CoG':
                    cog = np.vstack((cog, value))
                else:
                    continue
    return cog


def dataset_to_csv(cog, data_filename):
    cog_df = pd.DataFrame(cog)
    cog_df.to_csv(data_filename, mode='a', header=False, index=False)


def dataset_stacker(rem_dict, desired_key):
    stack = np.array([]).reshape(-1,120)
    for k, sub_dict in rem_dict.items():
        if k == 'header_info':
            continue
        else:
            print(k)
            for key, value in sub_dict.items():
                if key == desired_key:
                    stack = np.vstack((value.mean(axis=2), stack))
    print(stack.shape)
    return stack


def dataset_to_np_csv(stack, data_filename):
    with open(data_filename, 'ab') as file:
        stack.tofile(file)
