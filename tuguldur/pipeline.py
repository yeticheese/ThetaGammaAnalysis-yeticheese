import emd
import numpy as np
from src.utils.helper import load_config
from bycycle.features import compute_features

def compute_range(x):
    return x.max() - x.min()

def asc2desc(x):
    """Ascending to Descending ratio ( A / A+D )."""
    pt = emd.cycles.cf_peak_sample(x, interp=True)
    tt = emd.cycles.cf_trough_sample(x, interp=True)
    if (pt is None) or (tt is None):
        return np.nan
    asc = pt + (len(x) - tt)
    desc = tt - pt
    return asc / len(x)

def peak2trough(x):
    """Peak to trough ratio ( P / P+T )."""
    des = emd.cycles.cf_descending_zero_sample(x, interp=True)
    if des is None:
        return np.nan
    return des / len(x)


def segment(array, timestamps, skip_threshold=2500):
    """
    Segment an array based on trough timestamps.

    Parameters:
        array (numpy.ndarray): The array to be segmented.
        timestamps (pandas.DataFrame): DataFrame containing 'sample_last_trough' and 'sample_next_trough' columns.
        skip_threshold (int, optional): Minimum number of samples to skip between troughs.

    Returns:
        list: A list of segments as numpy arrays.
    """

    # Extract the last trough and next trough timestamps as NumPy arrays
    l1 = timestamps["sample_last_trough"].to_numpy()
    l2 = timestamps["sample_next_trough"].to_numpy()

    # List to store extracted segments
    segments = []
    # Assign the start of the first segment
    start = l1[0]
    # Initialize the end of the first segment
    end = 0

    for i, l1_i in enumerate(l1):
        if (l1_i - l2[i - 1]) > skip_threshold:
            # Set the end of the current segment
            end = l2[i - 1]
            # Append the current segment to the list
            segments.append((start, end))
            # Update the start for the next segment
            start = l1_i

    # Set the end of the last segment
    end = l2[-1]
    # Append the last segment to the list
    segments.append(array[start:end])

    return segments


def get_periods(timestamps, skip_threshold=2500):
    """
    Extract periods based on troughs from timestamp.

    Parameters:
        timestamps (pandas.DataFrame): DataFrame containing 'sample_last_trough' and 'sample_next_trough' columns.
        skip_threshold (int, optional): Minimum number of samples to skip between troughs.

    Returns:
        list: A list of tuples representing periods [(start1, end1), (start2, end2), ...].
    """

    # Extract the last trough and next trough timestamps as NumPy arrays
    l1 = timestamps["sample_last_trough"].to_numpy()
    l2 = timestamps["sample_next_trough"].to_numpy()

    # List to store extracted periods
    periods = []
    # Assign the start of the first period
    start = l1[0]
    # Initialize the end of the first period
    end = 0

    for i, l1_i in enumerate(l1):
        if (l1_i - l2[i - 1]) > skip_threshold:
            # Set the end of the current period
            end = l2[i - 1]
            # Append the current period to the list
            periods.append((start, end))
            # Update the start for the next period
            start = l1_i
    # Set the end of the last period
    end = l2[-1]
    # Append the last period to the list
    periods.append((start, end))  

    return periods

def emd_analysis(signal, fs):
    """
    Perform Empirical Mode Decomposition (EMD) analysis on a given signal.

    Args:
        signal (numpy.ndarray): The input signal for EMD analysis.
        fs (int or float): The sampling frequency of the signal.

    Returns:
        pandas.DataFrame: A DataFrame containing cycle metrics after extraction.
    """

    # Perform EMD and compute cycle metrics
    IP, IF, IA = emd.spectra.frequency_transform(signal, fs, 'hilbert', smooth_phase=3)
    C = emd.cycles.Cycles(IP.flatten())
    print("Detected cycles before extraction:")
    print(C)

    # Compute cycle metrics
    C.compute_cycle_metric('start_sample', np.arange(len(C.cycle_vect)), emd.cycles.cf_start_value)
    C.compute_cycle_metric('stop_sample', signal, emd.cycles.cf_end_value)
    C.compute_cycle_metric('peak_sample', signal, emd.cycles.cf_peak_sample)
    C.compute_cycle_metric('desc_sample', signal, emd.cycles.cf_descending_zero_sample)
    C.compute_cycle_metric('trough_sample', signal, emd.cycles.cf_trough_sample)
    C.compute_cycle_metric('duration_samples', signal, len)
    C.compute_cycle_metric('max_amp', IA, np.max)
    C.compute_cycle_metric('mean_if', IF, np.mean)
    C.compute_cycle_metric('max_if', IF, np.max)
    C.compute_cycle_metric('range_if', IF, compute_range)  # Make sure 'compute_range' is defined

    C.compute_cycle_metric('asc2desc', signal, asc2desc)  # Make sure 'asc2desc' is defined
    C.compute_cycle_metric('peak2trough', signal, peak2trough)  # Make sure 'peak2trough' is defined

    print('\nFinished computing the cycles metrics\n')

    # Extract a subset of the cycles
    amp_thresh = np.percentile(IA, 25)
    lo_freq_duration = fs / 5
    hi_freq_duration = fs / 12
    conditions = ['is_good==1',
                  f'duration_samples<{lo_freq_duration}',
                  f'duration_samples>{hi_freq_duration}',
                  f'max_amp>{amp_thresh}']

    print("Cycles after extraction:")
    df_emd = C.get_metric_dataframe(conditions=conditions)
    print(f'{len(df_emd)}')
    return df_emd

def pipeline(thetaSignal, phasic=True):

    # Load the parameters from the config file
    args = load_config("/home/miranjo/phasic_tonic/configs/test.yaml")
    f_theta = (args.pop("f_theta_lower", 4), args.pop("f_theta_upper", 12))
    threshold_kwargs = args.pop("threshold_kwargs", None)
    n_skip = args.pop("n_skip", 1.0)

    # Run Cycle by Cycle algorithm for burst detection
    df = compute_features(thetaSignal.filtered, thetaSignal.sampling_rate, f_range=f_theta, threshold_kwargs=threshold_kwargs, center_extrema='peak')
    result_msg = "{} periods in the {} signal: {}"

    if(phasic):
        df = df[df['is_burst']]
        print(result_msg.format("Phasic", thetaSignal.filter_type, len(df)))
        if(len(df) == 0):
            print("No phasic periods detected")
            return None
    else:
        df = df[df['is_burst' ] == False]
        print(result_msg.format("Tonic", thetaSignal.filter_type, len(df)))
        if(len(df) == 0):
            print("No tonic periods detected")
            return None

    timestamps = df[["sample_last_trough", "sample_next_trough"]]
    
    skip_threshold = int(n_skip * thetaSignal.sampling_rate)
    periods = get_periods(timestamps, skip_threshold=skip_threshold)
    print("Periods: \n", periods)
    return periods

    #segments = []
    #for period in periods:
    #    start, end = period
    #    segments.append(thetaSignal.filtered[start:end])
    #
    #print("Length: \n", [len(p) for p in segments])
    #
    #extracted_dfs = []
    #for seg in segments:
    #    extracted_dfs.append(emd_analysis(seg, thetaSignal.sampling_rate))
    #
    #return extracted_dfs
