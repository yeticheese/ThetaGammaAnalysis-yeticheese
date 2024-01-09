#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 14:32:12 2023

@author: eskor
"""
import re
import numpy as np
import pandas as pd
import scipy.io
from scipy import ndimage
import os
import csv
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import glob
import fnmatch
import scipy.stats as stats

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
pd.options.display.max_columns = 10

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

#file = '/Users/eskor/cbd/5/Rat_OS_Ephys_cbd_chronic_Rat5_411358_SD17_HC_20210804/2021-08-04_14-05-15_posttrial5/REM_data.mat'
#file = '/Users/eskor/cbd/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD3_OD_20210714/2021-07-14_16-07-39_posttrial5/REM_data.mat'
def find_paths_state():
    expr = '*posttrial*states*.mat*'
    input_dir = Path.home() /"cbd" #name of the folder where all the data is stored
    posttrial_states = list(input_dir.rglob(expr))
    return posttrial_states


def find_paths_VEH_REM():
    expr = '*VEH*posttrial*/REM_data.mat'
    VEH_REMs = [file_path for file_path in glob.iglob('**/*', recursive=True) if fnmatch.fnmatch(file_path, expr)]
    return VEH_REMs    
     

def find_HPC_files(posttrial_states):
    files_connected = {}
    for filepath in posttrial_states:
        dir_path = filepath.parent  # Get the directory path of the file
        #print(dir_path)
        HPC_file = list(dir_path.glob("*HPC*"))  # find HPC files in the same folders as states
        HPC_path = Path(HPC_file[0])  # define the file path as a PosixPath object
        HPC_dir_name = HPC_path.name  # get the directory name
        HPC_file = os.path.join(str(HPC_path.parent),
                                HPC_dir_name) # construct the new file path with only the directory name
        files_connected[filepath] = HPC_file  # make a dictionary with states and HPC files from the same folder
    return files_connected


def rem_extract(lfp, sleep_trans):
    remsleep = sleep_trans
    REM = []
    for i, rem in enumerate(remsleep):
        t1 = int(rem[0])
        t2 = int(rem[1])
        REM[i:] = [lfp[t1:t2]]
    REM = np.array(REM)
    return REM


def find_REM_in_HPC(files_connected):
    REM_files = []
    for state_file in files_connected.keys():
        HPC_file = files_connected[state_file]
        lfp = scipy.io.loadmat(HPC_file)['HPC']
        lfp = np.transpose(lfp)
        sleep = scipy.io.loadmat(str(state_file))
        states = sleep['states']
        transitions = sleep['transitions']
        if np.any(transitions[:, 0] == 5): # only if the file contains REM sleep
            sleep_transitions = transitions[transitions[:, 0] == 5][:, -2:]
            lfp = lfp[0]
            sleep_trans = np.floor(sleep_transitions * 2500)
            print(str(state_file.parent) + '/REM_data.mat')
            REM = rem_extract(lfp, sleep_trans)
            REM_file = str(state_file.parent) + '/REM_data.mat'
            scipy.io.savemat((REM_file), {'REM': REM}) #saving the file in the same folder
                                                                                        #as the original data
            REM_files.append(REM_file) #adding the filepaths to the list
    return REM_files


def load_data(file):
    dataREM = scipy.io.loadmat(file)
    #print(file)
   #epochs = dataREM['REM'].shape[1]
    return dataREM


def get_lfpREM(dataREM, epoch):
    if epoch == -1:
        lfpREM = np.array(dataREM['REM'][0])
        lfpREM = lfpREM.flatten()
        return lfpREM
    else:
        lfpREM = np.array(dataREM['REM'][0][epoch])
        lfpREM = lfpREM.flatten()
        return lfpREM


def get_sig_low(lfpREM):
    return filter_signal(lfpREM, fs, 'lowpass', f_lowpass,
                            n_seconds=n_seconds_filter, remove_edges=False)


def plot_signal(lfpREM, sig_low, times, title, epoch):
    xlim = (4, 10)
    tidx = np.logical_and(times >= xlim[0], times < xlim[1])

    plot_time_series(times[tidx], [lfpREM[tidx], sig_low[tidx]], colors=['k', 'k'], alpha=[.5, 1], lw=2)
    plt.title(title + ' timeseries ' + str(epoch +1))
    plt.show()
    


def get_peaks_troughs(sig_low):
    return find_extrema(sig_low, fs, f_theta,
                    filter_kwargs={'n_seconds':n_seconds_theta})


def plot_cyclepoints(sig_low, peaks, troughs, title, epoch):
    plot_cyclepoints_array(sig_low, fs, peaks=peaks, troughs=troughs, xlim=(4, 10))
    plt.title(title + ' cycle points ' + str(epoch + 1))
    
        
def localize_rises_decays(sig_low, peaks, troughs, title, epoch):
    rises, decays = find_zerox(sig_low, peaks, troughs)

    # plot_cyclepoints_array(sig_low, fs, xlim=(13, 14), peaks=peaks, troughs=troughs,
    #                        rises=rises, decays=decays)
    # plt.plot([13,14],[0,0],'b--')
    # plt.title(title + ' rises and decays ' + str(epoch + 1))
    
    
def separate_phasic_tonic(df_features):
    df_theta_phasic = df_features[df_features['is_burst']]
    df_theta_tonic  = df_features[df_features['is_burst' ] == False]
    return df_theta_phasic, df_theta_tonic


def get_cycles_phasic_tonic(df_theta_phasic, df_theta_tonic):
    cycles_phasic = df_theta_phasic['volt_amp']
    cycles_tonic  = df_theta_tonic['volt_amp']
    return cycles_phasic, cycles_tonic


def plot_cycles_amplitude(df_theta_phasic, df_theta_tonic, cycles_phasic, cycles_tonic, title, epoch):
    plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size': 12})
    if not df_theta_phasic.empty:
        plt.hist(cycles_phasic, weights=[1/len(df_theta_phasic)]*len(df_theta_phasic),
             bins=np.linspace(100, 1800,25), color='k', alpha=.8, label='Phasic REM')
    if not df_theta_tonic.empty:
        plt.hist(cycles_tonic , weights=[1/len(df_theta_tonic)]*len(df_theta_tonic),
             bins=np.linspace(100, 1800,25), color='r', alpha=.6, label='Tonic REM')
    #plt.xticks(np.arange(7))
    plt.legend(fontsize=15)
    #plt.xlim((0,6))
    plt.xlabel('Cycle amplitude ($\mu$V)')
    plt.ylabel('fraction of cycles')
    plt.title(title + ' cycle amplitudes ' + str(epoch + 1))
    

def plot_cycles_wo_weightning(cycles_phasic, cycles_tonic, title, epoch):
    plt.figure(figsize=(7,7))
    plt.hist(cycles_phasic, density = True,
             bins=20, color='k', alpha=.5, label='Phasic REM')
    plt.hist(cycles_tonic , density = True,
             bins=20, color='r', alpha=.5, label='Tonic REM')
    #plt.xticks(np.arange(7))
    plt.legend(fontsize=15)
    #plt.xlim((0,6))
    plt.xlabel('Cycle amplitude (a.u.)')
    plt.ylabel('fraction of cycles')
    plt.title(title + ' cycle amplitudes w/o weightning ' + str(epoch + 1))
    

def plot_cycle_period(cycles_phasic, cycles_tonic, title, epoch):
    plt.figure(figsize=(7,7))
    plt.rcParams.update({'font.size': 15})
    plt.hist(cycles_phasic, density = True,
             bins=25, color='k', alpha=.8, label='Phasic REM')
    plt.hist(cycles_tonic , density = True,
             bins=25, color='r', alpha=.6, label='Tonic REM')
    #plt.xticks(np.arange(7))
    plt.legend(fontsize=15)
    #plt.xlim((0,6))
    plt.xlabel('Cycle amplitude ($\mu$V)')
    plt.ylabel('fraction of cycles')
    plt.title(title + ' cycle period ' + str(epoch + 1))
    
def determine_bursting_parts(df_features, lfpREM):
    _, side_e = get_extrema_df(df_features)
    is_osc = np.zeros(len(lfpREM), dtype=bool)
    df_osc = df_features.loc[df_features['is_burst']]
    start = 0

    for _, cyc in df_osc.iterrows():
        samp_start_burst = int(cyc['sample_last_' + side_e]) - int(fs * start)
        samp_end_burst = int(cyc['sample_next_' + side_e] + 1) - int(fs * start)
        is_osc[samp_start_burst:samp_end_burst] = True
    return is_osc

def compute_instanteneous_freq(lfpREM, is_osc): # title, epoch):
    lfpPhasic   = lfpREM[is_osc == True]
    lfpTonic    = lfpREM[is_osc == False]
    
    if len(lfpPhasic) == 0 or len(lfpPhasic) < 1875:
        return
    
    i_f_p = freq_by_time(lfpPhasic, fs, f_theta)
    i_f_t = freq_by_time(lfpTonic, fs, f_theta)
    # run median filter on the data because of the transitions
    i_f_p_med = ndimage.median_filter(i_f_p, size= 20)
    i_f_t_med = ndimage.median_filter(i_f_t, size= 20)
    # plt.figure(figsize=(6,6))
    # plt.rcParams.update({'font.size': 15})
    # plt.hist(i_f_p_med, density=True, bins=np.arange(4,12,0.2), color='k', alpha=0.8, label='Phasic REM')
    # plt.hist(i_f_t_med, density=True, bins=np.arange(4,12,0.2), color='r', alpha=0.6, label='Tonic REM')
    # plt.legend(fontsize=15)
    # plt.xlabel('Theta Frequency [Hz]')
    # plt.ylabel('Probability') 
    # plt.title(title + ' instanteneous frequencies ' + str(epoch + 1))
    return i_f_p_med, i_f_t_med
    
    
def intervals_extract(iterable):
      
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
    lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]


def get_times_phasic(df_theta_phasic):
    bursting_int_phasic = df_theta_phasic.index
    bursting_list_phasic = list(intervals_extract(bursting_int_phasic))
    return bursting_list_phasic


def get_times_tonic(df_theta_tonic):
    bursting_int_tonic = df_theta_tonic.index
    bursting_list_tonic = list(intervals_extract(bursting_int_tonic))
    return bursting_list_tonic


def compute_interpeaks_phasic(bursting_list_phasic, df_features, times):
    phasicIPI = []
    
    for indBint in range(len(bursting_list_phasic)):
        indexBurst  = bursting_list_phasic[indBint]
        peaksInt    = df_features['sample_peak'][indexBurst[0]:indexBurst[1]].values
        peakTS      = times[peaksInt]
        IPI         = np.diff(peakTS)
        phasicIPI.extend(IPI)
    phasicIPI = np.array(phasicIPI)
    return phasicIPI
    

def compute_interpeaks_tonic(bursting_list_tonic, df_features, times):
    tonicIPI = []
    
    for indBint in range(len(bursting_list_tonic)):
        indexBurst  = bursting_list_tonic[indBint]
        peaksInt    = df_features['sample_peak'][indexBurst[0]:indexBurst[1]].values
        peakTS      = times[peaksInt]
        IPI         = np.diff(peakTS)
        tonicIPI.extend(IPI)
    tonicIPI = np.array(tonicIPI)
    return tonicIPI


def compute_length_oscillations(bursting_list, df_features):
    durations = []
    for list in bursting_list:
        start = df_features['sample_last_trough'][list[0]]
        end = df_features['sample_next_trough'][list[1]]
        duration = (end - start)/fs
        durations.append(duration)
    return durations


class PeriodType:
            def __init__(self, length, oscillation):
                self.length = length
                self.oscillation = oscillation
                
                
def create_period_type(durations, df_features, bursting_list):
    counter = 0
    mini_epochs = []
    for duration in durations:
        length = duration
        df_row = bursting_list[counter][0]
        if df_features['is_burst'][df_row] == True:
            oscillation = 'phasic'
        else:
            oscillation = 'tonic'
            
        mini_epoch = PeriodType(length, oscillation)
        mini_epochs.append(mini_epoch)
        counter = counter + 1
    return mini_epochs


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
    df = pd.read_excel('CBD_Chronic_treatment_conditions_study days overview.xlsx')

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


def analyse_epoch(dataREM, epoch, output_file, file):
    lfpREM = get_lfpREM(dataREM, epoch)
    if lfpREM.shape[0] < 25000:
        return
    
    title_name = create_name(file)
    sig_low = get_sig_low(lfpREM)
    times = np.arange(0, len(lfpREM)/fs, 1/fs)
    peaks, troughs = get_peaks_troughs(sig_low)
    localize_rises_decays(sig_low, peaks, troughs, title_name, epoch)
    df_features = compute_features(lfpREM, fs, f_theta,threshold_kwargs=threshold_kwargs, center_extrema='peak')
    df_theta_phasic, df_theta_tonic = separate_phasic_tonic(df_features)
    cycles_phasic, cycles_tonic = get_cycles_phasic_tonic(df_theta_phasic, df_theta_tonic)
    
    
    plot_signal(lfpREM, sig_low, times, title_name, epoch)
    plot_cyclepoints(sig_low, peaks, troughs, title_name, epoch)    
    sig_filt = filter_signal(lfpREM, fs, 'bandpass', (4, 10), n_seconds=.75, plot_properties=True)
    plot_burst_detect_summary(df_features, lfpREM, fs, threshold_kwargs,figsize=(16, 3), plot_only_result=True)
    plot_cycles_amplitude(df_theta_phasic, df_theta_tonic, cycles_phasic, cycles_tonic, title_name, epoch)
    plot_cycles_wo_weightning(cycles_phasic, cycles_tonic, title_name, epoch)
    
    cycles_phasic_period = df_theta_phasic['period']/ fs * 2500
    cycles_tonic_period  = df_theta_tonic['period']/ fs * 2500
    plot_cycle_period(cycles_phasic_period, cycles_tonic_period, title_name, epoch)
    idx_peak = df_theta_phasic['sample_peak'].values
    time_peak_phas = times[idx_peak]
    is_osc = determine_bursting_parts(df_features, lfpREM)
    compute_instanteneous_freq(lfpREM, is_osc, title_name, epoch)
    
    bursting_list_phasic = get_times_phasic(df_theta_phasic)
    bursting_list_tonic = get_times_tonic(df_theta_tonic)
    bursting_list_all = sorted(bursting_list_phasic + bursting_list_tonic)
    phasicIPI = compute_interpeaks_phasic(bursting_list_phasic, df_features, times)
    tonicIPI = compute_interpeaks_tonic(bursting_list_tonic, df_features, times)
    
    # durations_phasic = compute_length_oscillations(bursting_list_phasic, df_features)
    # durations_tonic = compute_length_oscillations(bursting_list_tonic, df_features)
    # durations_all = compute_length_oscillations(bursting_list_all, df_features)
    # mini_epochs = create_period_type(durations_all, df_features, bursting_list_all)
    # write_to_csv(output_file, epoch, mini_epochs)
    
def get_amplitudes(amplitudes, features):
    amplitudes_values = features['amp_fraction'].values
    amplitudes.extend(amplitudes_values)
    

def compute_amplitudes(amplitudes, features):
    for index,row in features.iterrows():
        amplitude_values = (row['volt_rise'] + row['volt_decay'])/2
        amplitudes.append(amplitude_values)
    return amplitudes


def compute_durations(durations, features):
    for index, row in features.iterrows():
        durations_cycles = (row['time_rise'] + row['time_decay'])/fs
        durations.append(durations_cycles)
    return durations
       

def compute_rise_decay_symmetry(rds_list, features):
    for index, row in features.iterrows():
        time_rise = row['time_rise']
        time_decay = row['time_decay']
        rds = time_rise/(time_rise + time_decay)
        rds_list.append(rds)
    return rds_list


def compute_peak_trough_symmetry(pts_list, features):
    for index, row in features.iterrows():
        time_peak = row['time_peak']
        pts = time_peak/(time_peak + row['sample_zerox_rise'] - row['sample_last_zerox_decay'])
        pts_list.append(pts)
    return pts_list


def get_instanteneous_freq(lfpREM, is_osc, inst_freq_phasic, inst_freq_tonic):
    returned_phasic, returned_tonic = compute_instanteneous_freq(lfpREM, is_osc)
    returned_phasic = returned_phasic.tolist()
    returned_tonic = returned_tonic.tolist()
    return inst_freq_phasic.append(returned_phasic), inst_freq_tonic.append(returned_tonic)

def find_VEH_csv():
    expr = '*TreatmentNegative*posttrial*.csv'
    input_dir = Path.home() /"cbd" #name of the folder where all the data is stored
    VEH_csvs = list(input_dir.rglob(expr))
    return VEH_csvs

def get_oscillations(csv_file, oscillations_phasic_HC, oscillations_phasic_OS,
                     oscillations_tonic_HC, oscillations_tonic_OS):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Initialize counters
    phasic_count = 0
    tonic_count = 0
    
    # Iterate over the rows in the DataFrame
    for _, row in df.iterrows():
        oscillation = row['Oscillation']
        if oscillation == 'phasic':
            phasic_count += 1
        elif oscillation == 'tonic':
            tonic_count += 1
            
    if 'HC' in csv_file.stem:
        oscillations_phasic_HC.append(phasic_count)
        oscillations_tonic_HC.append(tonic_count)
    else:
        oscillations_phasic_OS.append(phasic_count)
        oscillations_tonic_OS.append(tonic_count)
        
    return oscillations_phasic_HC, oscillations_phasic_OS, oscillations_tonic_HC, oscillations_tonic_OS
    
    
def get_lengths(csv_file, lengths_phasic_HC, lengths_phasic_OS,
                lengths_tonic_HC, lengths_tonic_OS):
    df = pd.read_csv(csv_file)
    
    if 'HC' in csv_file.stem:
        for _, row in df.iterrows():
            oscillation = row['Oscillation']
            length = row['Length (s)']
            if oscillation == 'phasic':
                lengths_phasic_HC.append(length)
            elif oscillation == 'tonic':
                lengths_tonic_HC.append(length)
    else:
        for _, row in df.iterrows():
            oscillation = row['Oscillation']
            length = row['Length (s)']
            if oscillation == 'phasic':
                lengths_phasic_OS.append(length)
            elif oscillation == 'tonic':
                lengths_tonic_OS.append(length)
    
    return lengths_phasic_HC, lengths_phasic_OS, lengths_tonic_HC, lengths_tonic_OS


def is_a_single_epoch_file(dataREM):
    if dataREM['REM'].shape[0] > 0:
        return not isinstance(dataREM['REM'][0][0], np.ndarray)
    else:
        return None

    
def analyse_cycles(file, dataREM, epoch, amplitudes_phasic, amplitudes_tonic,
               phasic_durations, tonic_durations, rds_phasic, rds_tonic, 
               pts_phasic, pts_tonic, inst_freq_phasic, inst_freq_tonic, is_a_single_epoch):
    #print(file, '\n', epoch)
    if is_a_single_epoch:
        lfpREM = get_lfpREM(dataREM, -1)
    else:
        lfpREM = get_lfpREM(dataREM, epoch)
    
    if lfpREM.shape[0] < 25000:
        return
    
    title_name = create_name(file)
    sig_low = get_sig_low(lfpREM)
    times = np.arange(0, len(lfpREM)/fs, 1/fs)
    peaks, troughs = get_peaks_troughs(sig_low)
    localize_rises_decays(sig_low, peaks, troughs, title_name, epoch)
    df_features = compute_features(lfpREM, fs, f_theta,threshold_kwargs=threshold_kwargs, center_extrema='peak')
    df_theta_phasic, df_theta_tonic = separate_phasic_tonic(df_features)
    
    if not df_theta_phasic.empty:
        #get_amplitudes(amplitudes_phasic, df_theta_phasic)
        #compute_amplitudes(amplitudes_phasic, df_theta_phasic)
        phasic_durations = compute_durations(phasic_durations, df_theta_phasic)
        rds_phasic = compute_rise_decay_symmetry(rds_phasic, df_features)
        pts_phasic = compute_peak_trough_symmetry(pts_phasic, df_features)
        #inst_freq_phasic, inst_freq_tonic = get_instanteneous_freq(lfpREM, determine_bursting_parts(df_features, lfpREM), inst_freq_phasic, inst_freq_tonic)
        
        
        
    #get_amplitudes(amplitudes_tonic, df_theta_tonic)
    #compute_amplitudes(amplitudes_tonic, df_theta_tonic)
    tonic_durations = compute_durations(tonic_durations, df_theta_tonic)
    rds_tonic = compute_rise_decay_symmetry(rds_tonic, df_features)
    pts_tonic = compute_peak_trough_symmetry(pts_tonic, df_features)
    

def plot_hist(group_1, group_2):
    plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size': 15})
    plt.hist(group_1, density=True, bins=np.arange(4,12,0.2), color='k', alpha=0.8, label='Phasic REM')
    plt.hist(group_2, density=True, bins=np.arange(4,12,0.2), color='r', alpha=0.6, label='Tonic REM')
    
    
def write_to_csv(output_file, epoch, mini_epochs):
        
    epoch_num = epoch + 1
    
    for mini_epoch in mini_epochs:
        duration = mini_epoch.length
        formatted_duration = "{:.4f}".format(duration)  # format to 4 decimal places
        oscillation = mini_epoch.oscillation
        
        # Open the CSV file in write mode
        with open(output_file, 'a', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)
            # Loop over the values to write
            csvwriter.writerow([epoch_num, oscillation, formatted_duration])  

    
def create_csv(file, title_name):
   input_file = file
   output_file = title_name + '.csv'
   
   with open(output_file, 'w', newline = '') as csvfile:
       csvwriter = csv.writer(csvfile)
       csvwriter.writerow([ '#epoch', 'Oscillation', 'Length (s)'])
    
   return output_file


def Shapiro_Wilk_test(data):
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print("Shapiro-Wilk Test:")
    print(f"Test statistic: {shapiro_stat:.4f}")
    print(f"P-value:", shapiro_p)
    return shapiro_p
    
def t_test(group1, group2, plot_results, name, y_label):
    normality_stat1 = Shapiro_Wilk_test(group1)
    normality_stat2 = Shapiro_Wilk_test(group2)
    
    if normality_stat1 > 0.05 and normality_stat2 > 0.05:
        t_statistic, p_value = stats.ttest_ind(group1, group2)
        
        # Calculate degrees of freedom
        n1 = len(group1)
        n2 = len(group2)
        df = n1 + n2 - 2
        
        # Calculate standard error of the mean
        se = np.sqrt(np.var(group1) / n1 + np.var(group2) / n2)
        
        print("T-statistic:", t_statistic)
        print("P-value:", p_value)
        print("Degrees of Freedom:", df)
        print("Standard Error of the Mean:", se)
    
        if plot_results:
            plt.boxplot([group1, group2])
            plt.xticks([1, 2], ['Phasic', 'Tonic'])
            plt.ylabel(y_label)
            plt.title(name)
            
            # Add text with t-test results
            plt.text(1.5, np.max(group1 + group2), f"t-statistic: {t_statistic:.2f}\np-value: {p_value:.6f}", ha='center', va='top')
            
            # Calculate SEM
            sem_group1 = np.std(group1) / np.sqrt(len(group1))
            sem_group2 = np.std(group2) / np.sqrt(len(group2))
            
            # Plot SEM error bars
            plt.errorbar([1, 2], [np.mean(group1), np.mean(group2)], yerr=[sem_group1, sem_group2], fmt='o', color='red')
            
            # Show the plot
            plt.show()

        
def U_test(group1, group2, name, y_label):
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    # Calculate degrees of freedom
    n1 = len(group1)
    n2 = len(group2)
    df = n1 + n2 - 2
    
    se = np.sqrt(np.var(group1) / n1 + np.var(group2) / n2)
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    print("Mann-Whitney U Test:")
    print(f"Test statistic: {u_stat:.4f}")
    print(f"P-value:", p_value)
    print("Degrees of Freedom:", df)
    print("Standard Error of the Mean:", se)
    print("Mean1: ", mean1)
    print("Mean2: ", mean2)
    
    
    plt.boxplot([group1, group2])
    plt.xticks([1, 2], ['Phasic', 'Tonic'])
    plt.ylabel(y_label)
    plt.title(name)
    
    # Add text with t-test results
    plt.text(1.5, np.max(group1 + group2), f"u-statistic: {u_stat:.2f}\np-value: {p_value:.6f}", ha='center', va='top')
    
    # Calculate SEM
    sem_group1 = np.std(group1) / np.sqrt(len(group1))
    sem_group2 = np.std(group2) / np.sqrt(len(group2))
    
    # Plot SEM error bars
    plt.errorbar([1, 2], [np.mean(group1), np.mean(group2)], yerr=[sem_group1, sem_group2], fmt='o', color='red')
    
    # Show the plot
    plt.show()

   
def main():
    
    #REM_files = find_REM_in_HPC(find_HPC_files(find_paths_state()))
    #print(REM_files)
    
    VEH_csvs = find_VEH_csv()
    amplitudes_phasic = []
    amplitudes_tonic = []
    phasic_durations = []
    tonic_durations = []
    rds_phasic = []
    rds_tonic = []
    pts_phasic = []
    pts_tonic = []
    inst_freq_phasic = []
    inst_freq_tonic = []
    oscillations_phasic_HC = []
    oscillations_phasic_OS = []
    oscillations_tonic_HC = []
    oscillations_tonic_OS = []
    lengths_phasic_HC = []
    lengths_phasic_OS = []
    lengths_tonic_HC = []
    lengths_tonic_OS = []
    
    
    VEH_files = find_paths_VEH_REM()
    
   
    for file in VEH_files:
        # if 'Rat6' in file:
        #     continue
        
        print(file)
        dataREM = load_data(file)
        is_a_single_epoch = is_a_single_epoch_file(dataREM)
        
        if is_a_single_epoch:
            analyse_cycles(file, dataREM, -1, amplitudes_phasic, amplitudes_tonic,
                            phasic_durations, tonic_durations, rds_phasic, rds_tonic, 
                            pts_phasic, pts_tonic, inst_freq_phasic, inst_freq_tonic, is_a_single_epoch)
        else:
            for epoch in range(dataREM['REM'].shape[1]):
                analyse_cycles(file, dataREM, epoch, amplitudes_phasic, amplitudes_tonic,
                                phasic_durations, tonic_durations, rds_phasic, rds_tonic, 
                                pts_phasic, pts_tonic, inst_freq_phasic, inst_freq_tonic, is_a_single_epoch)
    
    
    for csv_file in VEH_csvs:
        # if 'Rat6' in csv_file.stem:
        #     continue
        
        oscillations_phasic_HC, oscillations_phasic_OS, oscillations_tonic_HC, oscillations_tonic_OS = get_oscillations(csv_file, oscillations_phasic_HC, oscillations_phasic_OS, oscillations_tonic_HC, oscillations_tonic_OS)
        lengths_phasic_HC, lengths_phasic_OS, lengths_tonic_HC, lengths_tonic_OS = get_lengths(csv_file, lengths_phasic_HC, lengths_phasic_OS, lengths_tonic_HC, lengths_tonic_OS)
    
    
    #running t-tests for different parameters

    print('Comparison of Amplitudes')
    #t_test(amplitudes_phasic, amplitudes_tonic, True, 'amplitude', 'Voltage (mV)')
    U_test(amplitudes_phasic, amplitudes_tonic, 'amplitude', 'Voltage (mV)')
    
    
    print('\n','Comparisom of Cycle Durations')
    #t_test(phasic_durations, tonic_durations, True, 'Cycle durations', 'seconds')
    U_test(phasic_durations, tonic_durations,'Cycle durations', 'seconds')

    print('\n','Comparison of Rise-Decay Symmetries')
    #t_test(rds_phasic, rds_tonic, True, 'Rise-Decay symmetry', 'fraction of period in rise')
    U_test(rds_phasic, rds_tonic, 'Rise-Decay symmetry', 'fraction of period in rise')

    print('\n','Comparison of Peak-Trough Symmetries')
    #t_test(pts_phasic, pts_tonic, True, 'Peak-Trough symmetry', 'fraction of period in peak')
    U_test(pts_phasic, pts_tonic, 'Peak-Trough symmetry', 'fraction of period in peak')
    
    print('\n','Comparison of Phasic Oscillation Occurences')
    #t_test(oscillations_phasic_HC, oscillations_phasic_OS, True, 'Oscillation Occurences', '')
    U_test(oscillations_phasic_HC, oscillations_phasic_OS, 'Oscillation Occurences', '')
    
    print('\n','Comparison of Tonic Oscillation Occurences')
    #t_test(oscillations_tonic_HC, oscillations_tonic_OS, True, 'Oscillation Occurences', '')
    U_test(oscillations_tonic_HC, oscillations_tonic_OS, 'Oscillation Occurences', '')

    print('\n','Comparison of Phasic Period Durations')
    #t_test(lengths_phasic_HC, lengths_phasic_OS, True, 'Period Durations', 'seconds')
    U_test(lengths_phasic_HC, lengths_phasic_OS, 'Period Durations', 'seconds')
    
    print('\n','Comparison of Tonic Period Durations')
    #t_test(lengths_tonic_HC, lengths_tonic_OS, True, 'Period Durations', 'seconds')
    U_test(lengths_tonic_HC, lengths_tonic_OS, 'Period Durations', 'seconds')

       

        
    
if __name__ == '__main__':
    main()
