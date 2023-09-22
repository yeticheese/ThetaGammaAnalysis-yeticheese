import pandas as pd
import numpy as np
from scipy import ndimage
import itertools


from utils import find_matching_files, create_name, load_data

from matplotlib import pyplot as plt

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

def localize_rises_decays(sig_low, peaks, troughs, title):
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

def plot_signal(lfpREM, sig_low, times, title):
    xlim = (4, 10)
    tidx = np.logical_and(times >= xlim[0], times < xlim[1])

    plot_time_series(times[tidx], [lfpREM[tidx], sig_low[tidx]], colors=['k', 'k'], alpha=[.5, 1], lw=2)
    plt.title(title + ' timeseries ')
    plt.show()

def plot_cyclepoints(sig_low, peaks, troughs, title):
    plot_cyclepoints_array(sig_low, fs, peaks=peaks, troughs=troughs, xlim=(4, 10))
    plt.title(title + ' cycle points')

def plot_cycles_amplitude(df_theta_phasic, df_theta_tonic, cycles_phasic, cycles_tonic, title):
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
    plt.title(title + ' cycle amplitudes')

def plot_cycles_wo_weightning(cycles_phasic, cycles_tonic, title):
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
    plt.title(title + ' cycle amplitudes w/o weightning')
    

def plot_cycle_period(cycles_phasic, cycles_tonic, title):
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
    plt.title(title + ' cycle period')
    
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


def analyse(dataREM, file_REM):

    lfpREM = dataREM.flatten()

    if lfpREM.shape[0] < 25000:
        return
    
    print(lfpREM.shape)

    title_name = create_name(str(file_REM))
    sig_low = filter_signal(lfpREM, fs, 'lowpass', f_lowpass, n_seconds=n_seconds_filter, remove_edges=False)
    times = np.arange(0, len(lfpREM)/fs, 1/fs)
    peaks, troughs = find_extrema(sig_low, fs, f_theta, filter_kwargs={'n_seconds':n_seconds_theta})
    localize_rises_decays(sig_low, peaks, troughs, title_name)
    df_features = compute_features(lfpREM, fs, f_theta,threshold_kwargs=threshold_kwargs, center_extrema='peak')
    df_theta_phasic, df_theta_tonic = separate_phasic_tonic(df_features)
    cycles_phasic, cycles_tonic = get_cycles_phasic_tonic(df_theta_phasic, df_theta_tonic)
    
    
    plot_signal(lfpREM, sig_low, times, title_name)
    plot_cyclepoints(sig_low, peaks, troughs, title_name)    
    sig_filt = filter_signal(lfpREM, fs, 'bandpass', (4, 10), n_seconds=.75, plot_properties=True)
    plot_burst_detect_summary(df_features, lfpREM, fs, threshold_kwargs,figsize=(16, 3), plot_only_result=True)
    plot_cycles_amplitude(df_theta_phasic, df_theta_tonic, cycles_phasic, cycles_tonic, title_name)
    plot_cycles_wo_weightning(cycles_phasic, cycles_tonic, , title)
    
    cycles_phasic_period = df_theta_phasic['period']/ fs * 2500
    cycles_tonic_period  = df_theta_tonic['period']/ fs * 2500
    plot_cycle_period(cycles_phasic_period, cycles_tonic_period, title_name)
    idx_peak = df_theta_phasic['sample_peak'].values
    time_peak_phas = times[idx_peak]
    is_osc = determine_bursting_parts(df_features, lfpREM)
    compute_instanteneous_freq(lfpREM, is_osc)
    
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
    





def main():
    REM_files = find_matching_files(input_directory=INPUT_DIR, expression=REM)
    for file in REM_files:
        print(file)
        print(create_name(str(file)))
        dataREMs = load_data(file)
        for dataREM in dataREMs:
            analyse(dataREM, file)
            break



if __name__ == '__main__':
    main()