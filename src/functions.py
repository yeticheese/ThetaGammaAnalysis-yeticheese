import emd.sift as sift
import emd.spectra as spectra
import numpy as np
import pingouin as pg
import sails
from scipy.signal import convolve2d
from scipy.stats import zscore, binned_statistic
from scipy.ndimage import center_of_mass
from skimage.feature import peak_local_max


def get_rem_states(states, sample_rate):
    """
    Extract consecutive REM (Rapid Eye Movement) sleep states from a binary sleep state vector.

    Parameters:
    states (numpy.ndarray): A sleep state vector where 5 represents REM sleep and other values indicate non-REM.
    sample_rate (int or float): The sampling rate of the data.

    Returns:
    numpy.ndarray: An array of consecutive REM sleep state intervals in seconds, represented as (start, end) pairs.

    Notes:
    - This function processes a binary sleep state vector and identifies consecutive REM sleep intervals.
    - It calculates the start and end times of each REM state interval based on the provided sample rate.
    - The resulting intervals are returned as a numpy array of (start, end) pairs in seconds.
    """
    states = np.squeeze(states)
    rem_state_indices = np.where(states == 5)[0]
    rem_state_changes = np.diff(rem_state_indices)
    split_indices = np.where(rem_state_changes != 1)[0] + 1
    split_indices = np.concatenate(([0], split_indices, [len(rem_state_indices)]))
    consecutive_rem_states = np.empty((len(split_indices) - 1, 2))
    for i, (start, end) in enumerate(zip(split_indices, split_indices[1:])):
        start = rem_state_indices[start] * int(sample_rate)
        end = rem_state_indices[end - 1] * int(sample_rate)
        consecutive_rem_states[i] = np.array([start, end])
    consecutive_rem_states = np.array(consecutive_rem_states)
    null_states_mask = np.squeeze(np.diff(consecutive_rem_states) > 0)
    consecutive_rem_states = consecutive_rem_states[null_states_mask]
    return consecutive_rem_states


def morlet_wt(x, sample_rate, frequencies=np.arange(1, 200, 1), n=5, mode='complex'):
    """
        Compute the Morlet Wavelet Transform of a signal.

        Parameters:
        x (numpy.ndarray): The input signal for which the Morlet Wavelet Transform is computed.
        sample_rate (int or float): The sampling rate of the input signal.
        frequencies (numpy.ndarray, optional): An array of frequencies at which to compute the wavelet transform.
            Default is a range from 1 to 200 Hz with a step of 1 Hz.
        n (int, optional): The number of cycles of the Morlet wavelet. Default is 5.
        mode (str, optional): The return mode for the wavelet transform. Options are 'complex' (default),'amplitude', and 'power'.

        Returns:
        numpy.ndarray: The Morlet Wavelet Transform of the input signal.

        Notes:
        - This function computes the Morlet Wavelet Transform of a given signal.
        - The wavelet transform is computed at specified frequencies.
        - The number of cycles for the Morlet wavelet can be adjusted using the 'n' parameter.
        - The result can be returned in either complex or magnitude form either as amplitude or power, as specified by the 'mode' parameter.
        """
    wavelet_transform = sails.wavelet.morlet(x, freqs=frequencies, sample_rate=sample_rate, ncycles=n,
                                             ret_mode=mode, normalise=None)
    return wavelet_transform


def tg_split(mask_freq, theta_range=(5, 12)):
    """
        Split a frequency vector into sub-theta, theta, and supra-theta components.

        Parameters:
        mask_freq (numpy.ndarray): A frequency vector or array of frequency values.
        theta_range (tuple, optional): A tuple defining the theta frequency range (lower, upper).
            Default is (5, 12).

        Returns:
        tuple: A tuple containing boolean masks for sub-theta, theta, and supra-theta frequency components.

        Notes:
        - This function splits a frequency mask into three components based on a specified theta frequency range.
        - The theta frequency range is defined by the 'theta_range' parameter.
        - The resulting masks 'sub', 'theta', and 'supra' represent sub-theta, theta, and supra-theta frequency components.
        """
    lower = np.min(theta_range)
    upper = np.max(theta_range)
    mask_index = np.logical_and(mask_freq >= lower, mask_freq < upper)
    sub_mask_index = mask_freq < lower
    supra_mask_index = mask_freq > upper
    sub = sub_mask_index
    theta = mask_index
    supra = supra_mask_index

    return sub, theta, supra


def zero_cross(x):
    """
    Find the indices of zero-crossings in a 1D signal.

    Parameters:
    x (numpy.ndarray): The input 1D signal.

    Returns:
    numpy.ndarray: An array of indices where zero-crossings occur in the input signal.

    Notes:
    - This function identifies the indices where zero-crossings occur in a given 1D signal.
    - It detects both rising and falling zero-crossings.
    """
    decay = np.logical_and((x > 0)[1:], ~(x > 0)[:-1]).nonzero()[0]
    rise = np.logical_and((x <= 0)[1:], ~(x <= 0)[:-1]).nonzero()[0]
    zero_xs = np.sort(np.append(rise, decay))
    return zero_xs


def extrema(x):
    """
    Find extrema (peaks, troughs) and zero crossings in a 1D signal.

    Parameters:
    x (numpy.ndarray): The input 1D signal.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: Indices of zero-crossings in the input signal.
        - numpy.ndarray: Indices of troughs in the input signal.
        - numpy.ndarray: Indices of peaks in the input signal.

    Notes:
    - This function identifies and returns the indices of zero-crossings, troughs, and peaks in a given 1D signal.
    - Zero-crossings are points where the signal crosses the zero axis.
    - Troughs are local minima, and peaks are local maxima in the signal.
    """
    zero_xs = zero_cross(x)
    peaks = np.empty((0,)).astype(int)
    troughs = np.empty((0,)).astype(int)
    for t1, t2 in zip(zero_xs, zero_xs[1:]):
        extrema0 = np.argmax(np.abs(x[t1:t2])).astype(int) + t1
        if bool(x[extrema0] > 0):
            peaks = np.append(peaks, extrema0)
        else:
            troughs = np.append(troughs, extrema0)
    return zero_xs, troughs, peaks


def get_cycles_data(x, rem_states, sample_rate, theta_range=(5, 12)):
    """
    Generate a nested dictionary containing extracted data and desired metadata of each REM epochs in the input sleep
    signal

    Parameters:
    x (numpy.ndarray): The input 1D sleep signal.
    rem_states (numpy.ndarray): A sleep state vector where 5 represents REM sleep and other values indicate non-REM.
    sample_rate (int or float): The sampling rate of the data.
    theta_range (tuple, optional): A tuple defining the theta frequency range (lower, upper).
            Default is (5, 12).

    Returns:
    rem_dict: A nested dictionary of extracted signal data and signal source metadata

    Notes:
    - The dictionary output structure comes out as below:
        |----REM 1
        |    |----start_end:
        |    |----IMFs:
        |    |----IMF Frequencies:
        |    |----Instantaneous Phases:
        |    |----Instantaneous Frequencies:
        |    |----Instantaneous Amplitudes:
        |    |----Cycles:
        |----REM (...)
        |    |--------(...)
    """

    # Detect REM periods
    consecutive_rem_states = get_rem_states(rem_states, sample_rate)

    # Intializing variables
    rem_imf = []
    rem_mask_freq = []
    instantaneous_phase = []
    instantaneous_freq = []
    instantaneous_amp = []
    sub_theta_sig = np.empty((0,))
    theta_peak_sig = np.empty((0,))
    cycles = np.empty((0, 5)).astype(int)
    rem_dict = {}
    sub_dict = rem_dict

    # Loop through each REM epoch
    for i, rem in enumerate(consecutive_rem_states, start=1):
        sub_dict.setdefault(f'REM {i}', {})
        start = int(rem[0])
        end = int(rem[1])
        signal = x[start:end]

        # Extraction of IMFs and IMF Frequencies for current REM epoch
        imf, mask_freq = sift.iterated_mask_sift(signal,
                                                 mask_0='zc',
                                                 sample_rate=sample_rate,
                                                 ret_mask_freq=True)

        # Extract Instantaneous Phase, Frequencies and Amplitudes of each IMF for current REM epoch
        IP, IF, IA = spectra.frequency_transform(imf, sample_rate, 'nht')

        # Identify sub-theta, theta, and supra-theta frequencies
        sub_theta, theta, _ = tg_split(mask_freq, theta_range)

        rem_imf.append(imf)
        rem_mask_freq.append(mask_freq)
        instantaneous_phase.append(IP)
        instantaneous_freq.append(IF)
        instantaneous_amp.append(IA)

        # Generate the theta signal to detect cycles
        theta_sig = np.sum(imf.T[theta], axis=0)

        # Parse the sub-theta signal of all REM periods into one variable to set amplitude threshold
        sub_theta_sig = np.append(sub_theta_sig, np.sum(imf.T[sub_theta], axis=0))

        # Generate extrema locations and zero crossing on the generated theta signal
        zero_x, trough, peak = extrema(np.sum(imf.T[theta], axis=0))

        # Create the cycles array for the current REM epoch
        zero_x = np.vstack((zero_x[:-2:2], zero_x[1:-1:2], zero_x[2::2])).T

        size_adjust = np.min([trough.shape[0], zero_x.shape[0], peak.shape[0]])
        zero_x = zero_x[:size_adjust]
        cycle = np.empty((size_adjust, 5)).astype(int)
        cycle[:, [0, 2, 4]] = zero_x
        if trough[0] < peak[0]:
            cycle[:, 1] = trough[:zero_x.shape[0]]
            cycle[:, 3] = peak[:zero_x.shape[0]]
        else:
            cycle[:, 3] = trough[:zero_x.shape[0]]
            cycle[:, 1] = peak[:zero_x.shape[0]]

        broken_cycle = cycle[~np.all(np.diff(cycle, axis=1) > 0, axis=1)]
        broken_cycle_mask = np.diff(broken_cycle, axis=1) > 0

        adjust_condition = np.all(np.all(broken_cycle_mask[1:] == [True, False, False, True],
                                         axis=0) == True)
        adjust_loc = np.where(np.all(np.diff(cycle, axis=1) > 0, axis=1) == False)[0][1:-1]

        fixed_cycle = broken_cycle[1:-1]
        if adjust_condition:
            fixed_cycle[:, 1] = cycle[adjust_loc - 1, 1]
            fixed_cycle[:, 3] = cycle[adjust_loc + 1, 3]
        else:
            fixed_cycle[:, 3] = cycle[adjust_loc - 1, 3]
            fixed_cycle[:, 1] = cycle[adjust_loc + 1, 1]

        cycle = cycle[np.all(np.diff(cycle, axis=1) > 0, axis=1)]
        cycle = np.vstack((cycle, fixed_cycle))
        if trough[0] < peak[0]:
            cycle = np.hstack((cycle[:-1, 1:-1], cycle[1:, :2]))
        else:
            cycle = np.hstack((cycle[:-1, 3].reshape((-1, 1)), cycle[1:, :-1]))

        # Create an array of amplitudes at the peaks
        theta_peak_sig = np.append(theta_peak_sig, theta_sig[cycle[:, 2]])
        cycles = np.vstack((cycles, cycle + start))

    # Set the minimum amplitude threshold and discard unsatisfactory theta peaks
    min_peak_amp = 2 * sub_theta_sig.std()
    peak_mask = theta_peak_sig > min_peak_amp

    # Set the frequency threshold and discard and unsatisfactory difference between trough pairs
    upper_diff = np.floor(1000 / np.min(theta_range))
    lower_diff = np.floor(1000 / np.max(theta_range))
    diff_mask = np.logical_and(np.diff(cycles[:, [0, -1]], axis=1) * (1000 / sample_rate) > lower_diff,
                               np.diff(cycles[:, [0, -1]], axis=1) * (1000 / sample_rate) <= upper_diff)

    # Create a boolean mask that satisfy both the frequency and amplitude threshold criteria
    extrema_mask = np.logical_and(np.squeeze(diff_mask), peak_mask)

    # Pass the boolean mask on the cycles array to discard any unsatisfactory cycles
    cycles = cycles[extrema_mask]

    # Place outputs in a nested dictionary
    for j, rem in enumerate(rem_dict.values()):
        rem['start-end'] = consecutive_rem_states[j]
        rem['IMFs'] = rem_imf[j]
        rem['IMF_Frequencies'] = rem_mask_freq[j]
        rem['Instantaneous Phases'] = instantaneous_phase[j]
        rem['Instantaneous Frequencies'] = instantaneous_freq[j]
        rem['Instantaneous Amplitudes'] = instantaneous_amp[j]
        cycles_mask = (cycles > consecutive_rem_states[j, 0]) & (cycles < consecutive_rem_states[j, 1])
        cycles_mask = np.all(cycles_mask == True, axis=1)
        rem_cycles = cycles[cycles_mask]
        rem['Cycles'] = rem_cycles

    return rem_dict


def bin_tf_to_fpp(x, power, bin_count):

    """
       Bin time-frequency power data into Frequency Phase Power (FPP) plots using specified time intervals of cycles.

       Parameters:
       x (numpy.ndarray): A 1D or 2D array specifying time intervals of cycles for binning.
           - If 1D, it represents a single time interval [start, end].
           - If 2D, it represents multiple time intervals, where each row is [start, end].
       power (numpy.ndarray): The time-frequency power spectrum data to be binned.
       bin_count (int): The number of bins to divide the time intervals into.

       Returns:
       fpp(numpy.ndarray): Returns FPP plots

       Notes:
       - This function takes time-frequency power data and divides it into FPP plots based on specified
         time intervals.
       - The 'x' parameter defines the time intervals, which can be a single interval or multiple intervals.
       - The 'power' parameter is the time-frequency power data to be binned.
       - The 'bin_count' parameter determines the number of bins within each time interval.
       """

    if x.ndim == 1:  # Handle the case when x is of size (2)
        bin_ranges = np.arange(x[0], x[1], 1)
        fpp = binned_statistic(bin_ranges, power[:, x[0]:x[1]], 'mean', bins=bin_count)[0]
        fpp = np.expand_dims(fpp, axis=0)  # Add an extra dimension to match the desired output shape
    elif x.ndim == 2:  # Handle the case when x is of size (n, 2)
        fpp = []
        for i in range(x.shape[0]):
            bin_ranges = np.arange(x[i, 0], x[i, 1], 1)
            fpp_row = binned_statistic(bin_ranges, power[:, x[i, 0]:x[i, 1]], 'mean', bins=bin_count)[0]
            fpp.append(fpp_row)
        fpp = np.array(fpp)
    else:
        raise ValueError("Invalid size for x")

    return fpp


def calculate_cog(frequencies, angles, amplitudes, ratio):
    """
       Calculate the Center of Gravity (CoG) of an FPP plots of cycles.

       Parameters:
       frequencies (numpy.ndarray): An array of frequencies corresponding to FPP frequencies.
       angles (numpy.ndarray): An array of phase angles in degrees.
       amplitudes (numpy.ndarray): An array of magnitude values (can be power).
           - If 2D, it represents magnitude values for multiple frequency bins.
           - If 3D, it represents magnitude values for multiple frequency bins across multiple trials or subjects.
       ratio (float): A ratio threshold for selecting magnitude values in the phase direction .

       Returns:
       numpy.ndarray: The Center of Gravity (CoG) for the FPP plot as frequency and phase.
           - For 2D amplitudes: A 2D array containing CoG values for frequency and phase.
           - For 3D amplitudes: A 2D array containing CoG values for frequency and phase cycle.

       Notes:
       - This function calculates the Center of Gravity (CoG) of the FPP plots.
       - It can handle 2D or 3D amplitude arrays, representing either single or multiple cycles.
       """
    angles = np.deg2rad(angles)
    cog = np.empty((0, 2))

    if amplitudes.ndim == 2:
        numerator = np.sum(frequencies * np.sum(amplitudes, axis=1))
        denominator = np.sum(amplitudes)
        cog_f = numerator / denominator
        floor = np.floor(cog_f).astype(int) - frequencies[0]
        ceil = np.ceil(cog_f).astype(int) - frequencies[0]
        new_fpp = np.where(amplitudes >= np.max(amplitudes[[floor, ceil], :]) * ratio, amplitudes, 0)
        cog_ph = np.rad2deg(pg.circ_mean(angles, w=np.sum(new_fpp, axis=0)))
        cog = np.array([cog_f, cog_ph])

    elif amplitudes.ndim == 3:
        indices_to_subset = np.empty((amplitudes.shape[0], 2)).astype(int)
        cog = np.empty((amplitudes.shape[0], 2))
        numerator = np.sum(frequencies * np.sum(amplitudes, axis=2), axis=1)
        denominator = np.sum(amplitudes, axis=(1, 2))
        cog_f = (numerator / denominator)

        vectorized_floor = np.vectorize(np.floor)
        vectorized_ceil = np.vectorize(np.ceil)
        indices_to_subset[:, 0] = vectorized_floor(cog_f) - frequencies[0]
        indices_to_subset[:, 1] = vectorized_ceil(cog_f) - frequencies[0]

        max_amps = np.max(amplitudes[np.arange(amplitudes.shape[0])[:, np.newaxis], indices_to_subset, :], axis=(1, 2))
        print(max_amps.shape)

        for i, max_amp in enumerate(max_amps):
            new_fpp = np.where(amplitudes[i] >= max_amp * ratio, amplitudes[i], 0)
            cog[i, 1] = np.rad2deg(pg.circ_mean(angles, w=np.sum(new_fpp, axis=0)))
        cog[:, 0] = cog_f

    return cog


def boxcar_smooth(x, boxcar_window):
    """
    Apply a boxcar smoothing filter to a 1D or 2D signal.

    Parameters:
    x (numpy.ndarray): The input signal to be smoothed.
    boxcar_window (int or tuple): The size of the boxcar smoothing window.
        - If int, it specifies the window size in both dimensions for a 2D signal.
        - If tuple, it specifies the window size as (time_window, frequency_window) for a 2D signal.

    Returns:
    numpy.ndarray: The smoothed signal after applying the boxcar smoothing filter.

    Notes:
    - This function applies a boxcar smoothing filter to a 1D or 2D signal.
    - The size of the smoothing window can be specified as an integer (square window) or a tuple (rectangular window).
    - For 2D signals, the boxcar smoothing is applied in both the time and frequency dimensions.
    """
    if x.ndim == 1:
        if boxcar_window % 2 == 0:
            boxcar_window += 1
        window = np.ones((1, boxcar_window)) / boxcar_window
        x_spectrum = np.convolve(x, window, mode='same')
    else:
        bool_window = np.where(~boxcar_window % 2 == 0, boxcar_window, boxcar_window + 1)
        window_t = np.ones((1, bool_window[0])) / bool_window[0]
        window_f = np.ones((1, bool_window[1])) / bool_window[1]
        x_spectrum_t = convolve2d(x, window_t, mode='same')
        x_spectrum = convolve2d(x_spectrum_t, window_f.T, mode='same')

    return x_spectrum


def peak_cog(frequencies, angles, amplitudes, ratio):
    """
       Calculate the Center of Gravity (CoG) and snap to the nearest peak in the FPP array.

       Parameters:
       frequencies (numpy.ndarray): An array of frequencies.
       angles (numpy.ndarray): An array of phase angles in degrees.
       amplitudes (numpy.ndarray): An array of magnitude values (can be power).
           - If 2D, it represents magnitude values for multiple phase bins.
           - If 3D, it represents magnitude values for multiple phase bins across multiple trials or subjects.
       ratio (float): A ratio threshold for selecting magnitude values in the phase direction.

       Returns:
       numpy.ndarray: Snapped peaks of each FPP cycle .
           - For 2D amplitudes: A 2D array containing CoG peaks for frequency and phase.
           - For 3D amplitudes: A 2D array containing CoG peaks for frequency and phase for each cycle.

       Notes:
       - This function calculates the Center of Gravity (CoG) of the passed FPP cycles and their respective magnitude
       peaks.
       - The CoG is then shifted to the nearest peak of Euclidean distance treating frequency as linear and phase as
       circular.
       """
    def nearest_peaks(frequency, angle, amplitude, ratio):
        peak_indices = peak_local_max(amplitude, min_distance=1, threshold_abs=0)
        cog_f = calculate_cog(frequency, angle, amplitude, ratio)

        if peak_indices.shape[0] == 0:
            cog_peak = cog_f
        else:
            cog_fx = np.array([cog_f[0], cog_f[0] * np.cos(np.deg2rad(cog_f[1] - angle[0])),
                               cog_f[0] * np.sin(np.deg2rad(cog_f[1] - angle[0]))])
            peak_loc = peak_loc = np.empty((peak_indices.shape[0], 4))
            peak_loc[:, [0, 1]] = np.array([frequency[peak_indices.T[0]], angle[peak_indices.T[1]]]).T
            peak_loc[:, 2] = peak_loc[:, 0] * np.cos(np.deg2rad(peak_loc[:, 1] - angle[0]))
            peak_loc[:, 3] = peak_loc[:, 0] * np.sin(np.deg2rad(peak_loc[:, 1] - angle[0]))
            peak_loc = peak_loc[:, [0, 2, 3]]
            distances = np.abs(peak_loc - cog_fx)

            cog_pos = peak_indices[np.argmin(np.linalg.norm(distances, axis=1))]

            cog_peak = np.array([frequency[cog_pos[0]], angle[cog_pos[1]]])

        return cog_peak

    if amplitudes.ndim == 2:
        cog = nearest_peaks(frequencies, angles, amplitudes, ratio)
    elif amplitudes.ndim == 3:
        cog = np.empty((amplitudes.shape[0], 2))
        for i, fpp in enumerate(amplitudes):
            cog[i] = nearest_peaks(frequencies, angles, fpp, ratio)
    return cog


def max_peaks(amplitudes):
    """
    Extract maximum peaks from amplitude distributions.

    Parameters:
    amplitudes (numpy.ndarray): An array of magnitude values of FPP cycles, can be amplitude or power.
        - If 2D, it represents magnitude values for multiple phase bins.
        - If 3D, it represents magnitude values for multiple phase bins across multiple cycles.
        - Magnitude values are only z-score normalized arrays from the original.

    Returns:
    numpy.ndarray: An array with maximum peaks extracted from the input magnitudes.
        - For 2D amplitudes: A 2D array with peak values at the same positions as the input.
        - For 3D amplitudes: A 3D array with peak values at the same positions as the input for each cycle.

    Notes:
    - This function extracts boundary values from FPP cycles.
    - It can handle 2D or 3D FPP arrays, representing either single or multiple cycles.
    - Peaks are extracted using the skimage `peak_local_max` function with specified parameters.
    """
    new_fpp = np.zeros(amplitudes.shape)
    if amplitudes.ndim == 2:
        peak_indices = peak_local_max(amplitudes, min_distance=1, threshold_abs=0)
        if peak_indices.shape[0] == 0:
            new_fpp = np.where(amplitudes > 0, amplitudes, 0)
        else:
            new_fpp[peak_indices.T[0], peak_indices.T[1]] = amplitudes[peak_indices.T[0], peak_indices.T[1]]
    elif amplitudes.ndim == 3:
        for i, fpp in enumerate(amplitudes):
            peak_indices = peak_local_max(fpp, min_distance=1, threshold_abs=0)
            if peak_indices.shape[0] == 0:
                new_fpp[i] = np.where(fpp > 0, fpp, 0)
            else:
                new_fpp[i, peak_indices.T[0], peak_indices.T[1]] = fpp[peak_indices.T[0], peak_indices.T[1]]
    return new_fpp


def boundary_peaks(amplitudes):
    """
    Extract maximum peaks from amplitude distributions.

    Parameters:
    amplitudes (numpy.ndarray): An array of magnitude values of FPP cycles, can be amplitude or power.
        - If 2D, it represents magnitude values for multiple phase bins.
        - If 3D, it represents magnitude values for multiple phase bins across multiple cycles.
        - Magnitude values are only z-score normalized arrays from the original.

    Returns:
    numpy.ndarray: An array with boundary values between largest peak value and 95 percent of lowest peak value,
     extracted from the input magnitudes.
        - For 2D amplitudes: A 2D array with boundary values at the same positions as the input.
        - For 3D amplitudes: A 3D array with boundary values at the same positions as the input for each cycle.

    Notes:
    - This function extracts boundary values from FPP cycles.
    - It can handle 2D or 3D FPP arrays, representing either single or multiple cycles.
    - Peaks are extracted using the skimage `peak_local_max` function with specified parameters.
    """
    adjusted_fpp = np.zeros(amplitudes.shape)
    if amplitudes.ndim == 2:
        peak_indices = peak_local_max(amplitudes, min_distance=1, threshold_abs=0)
        if peak_indices.shape[0] == 0:
            adjusted_fpp = np.where(amplitudes > 0, amplitudes, 0)
        else:
            new_fpp = amplitudes[peak_indices.T[0], peak_indices.T[1]]
            maximum = np.max(new_fpp)
            minimum = np.min(new_fpp)
            adjusted_fpp = np.where((amplitudes <= maximum) & (amplitudes >= 0.95 * minimum), amplitudes, 0)
    elif amplitudes.ndim == 3:
        for i, fpp in enumerate(amplitudes):
            peak_indices = peak_local_max(fpp, min_distance=1, threshold_abs=0)
            print(peak_indices.shape)
            if peak_indices.shape[0] == 0:
                adjusted_fpp[i] = np.where(fpp > 0, fpp, 0)
            else:
                maximum = np.max(fpp[peak_indices.T[0], peak_indices.T[1]])
                minimum = np.min(fpp[peak_indices.T[0], peak_indices.T[1]])
                adjusted_fpp[i] = np.where((fpp <= maximum) & (fpp >= 0.95 * minimum), fpp, 0)
    return adjusted_fpp


def rem_fpp_gen(rem_dict, x, sample_rate, frequencies, angles, ratio, boxcar_window=None, norm='', fpp_method='',
                cog_method=''):
    x = np.squeeze(x)
    cycles_dict = rem_dict
    rem_dict = {}
    sub_dict = rem_dict
    for key, value in cycles_dict.items():
        print(key)
        if 'Cycles' in value.keys():
            sub_dict.setdefault(key, {})
            t = value['start-end'].astype(int)
            print(t, t[0], t[1])
            sig = x[t[0]:t[1]]
            print(sig.shape)
            power = morlet_wt(sig, sample_rate, frequencies, mode='power')
            cycles = value['Cycles'][:, [0, -1]] - t[0]
            if boxcar_window is not None:
                power = boxcar_smooth(power, boxcar_window)
            if norm == 'simple_x':
                power = power / np.sum(power, axis=0)
            elif norm == 'simple_y':
                power = power / np.sum(power, axis=1)[:, np.newaxis]
            elif norm == 'zscore_y':
                power = zscore(power, axis=0)
            elif norm == 'zscore_x':
                power = zscore(power, axis=1)
            fpp_plots = bin_tf_to_fpp(cycles, power, 19)
            sub_dict[key]['FPP_cycles'] = fpp_plots
            if fpp_method == 'max_peaks':
                fpp_plots = max_peaks(fpp_plots)
                print(fpp_plots.shape)
            elif fpp_method == 'boundary_peaks':
                fpp_plots = boundary_peaks(fpp_plots)
            if cog_method == 'nearest':
                cog = peak_cog(frequencies, angles, fpp_plots, ratio)
            else:
                cog = calculate_cog(frequencies, angles, fpp_plots, ratio)
            sub_dict[key]['CoG'] = cog
        else:
            continue
    return rem_dict
