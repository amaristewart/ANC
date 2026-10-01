import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz, welch
import soundfile as sf

## Configuration
SAMPLE_RATE = 16000
DURATION_SECONDS = 1
NUM_TAPS = 32
RANDOM_SEED = 0
CONVERGENCE_PLOT_SAMPLES = 5000
SMOOTHING_WINDOW = 100
SENSOR_NOISE_SNR_DB = 30

LINE_STYLES = {
    'LMS': ('-', 4),
    'NLMS': ('--', 2),
    'FxLMS': ('-', 2),
}

## Signal generation

# White noise picked up by reference mic
def generate_reference_noise(sample_rate, duration_seconds, seed):
    rng = np.random.default_rng(seed)
    num_samples = int(sample_rate * duration_seconds)
    return rng.standard_normal(num_samples) / np.sqrt(2)

# White noise from error mic, scaled to a target SNR against clean_signal
def generate_sensor_noise(clean_signal, snr_db, rng):
    signal_power = np.mean(clean_signal ** 2)
    sensor_noise_power = signal_power / 10 ** (snr_db / 10)
    return rng.standard_normal(len(clean_signal)) * np.sqrt(sensor_noise_power)

## Helper functions

# Prepend zeros so a full-length window exists at first 
def pad_with_zeros(signal, num_taps):
    return np.concatenate([np.zeros(num_taps - 1), signal])

# Return last num_taps sample ending at sample_index, newest first
def get_recent_samples(padded_signal, sample_index, num_taps):
    return padded_signal[sample_index:sample_index + num_taps][::-1]

## Adaptive filters

def lms_filter(reference_signal, desired_signal, step_size, num_taps):
    num_samples = len(reference_signal)
    filter_weights = np.zeros(num_taps)
    error_signal = np.zeros(num_samples)
    padded_reference = pad_with_zeros(reference_signal, num_taps)
    
    for sample_index in range(num_samples):
        recent_reference = get_recent_samples(padded_reference, sample_index, num_taps)
        filter_output = filter_weights @ recent_reference
        error_signal[sample_index] = desired_signal[sample_index] - filter_output
        filter_weights += step_size * error_signal[sample_index] * recent_reference
    
    return filter_weights, error_signal ** 2
        
def nlms_filter(reference_signal, desired_signal, step_size, num_taps, regularization=1e-6):
    num_samples = len(reference_signal)
    filter_weights = np.zeros(num_taps)
    error_signal = np.zeros(num_samples)
    padded_reference = pad_with_zeros(reference_signal, num_taps)

    for sample_index in range(num_samples):
        recent_reference = get_recent_samples(padded_reference, sample_index, num_taps)
        filter_output = filter_weights @ recent_reference
        error_signal[sample_index] = desired_signal[sample_index] - filter_output
        reference_energy = recent_reference @ recent_reference
        filter_weights += (step_size * error_signal[sample_index] * recent_reference / (reference_energy + regularization))
    
    return filter_weights, error_signal ** 2

def fxlms_filter(reference_signal, desired_signal, step_size, num_taps, secondary_path, secondary_path_estimate):
    num_samples = len(reference_signal)
    filter_weights = np.zeros(num_taps)
    error_signal = np.zeros(num_samples)
 
    # Update uses reference after it passes through secondary path estimate
    filtered_reference = lfilter(secondary_path_estimate, 1, reference_signal)
    padded_reference = pad_with_zeros(reference_signal, num_taps)
    padded_filtered_reference = pad_with_zeros(filtered_reference, num_taps)
 
    recent_controller_outputs = np.zeros(len(secondary_path))
 
    for sample_index in range(num_samples):
        recent_reference = get_recent_samples(padded_reference, sample_index, num_taps)
        recent_filtered_reference = get_recent_samples(padded_filtered_reference, sample_index, num_taps)
 
        # Controller output (anti-noise) goes through real secondary path to error mic
        recent_controller_outputs = np.roll(recent_controller_outputs, 1)
        recent_controller_outputs[0] = filter_weights @ recent_reference
        anti_noise_at_error_mic = secondary_path @ recent_controller_outputs
 
        error_signal[sample_index] = desired_signal[sample_index] - anti_noise_at_error_mic
        filter_weights += step_size * error_signal[sample_index] * recent_filtered_reference
 
    return filter_weights, error_signal ** 2

## Calculation functions

# Convert a power value or array to dB, with a floor to avoid log(0)
def to_db_power(power):
    return 10 * np.log10(power + 1e-12)
 
# Moving average over full windows only, so the learning curve is readable on noisy data
# The output is shorter than input by window - 1 samples
# Output index k corresponds to input sample k + window - 1
def smooth_squared_error(squared_error, window):
    return np.convolve(squared_error, np.ones(window) / window, mode='valid')

# Return frequencies in Hz and magnitude response in dB for FIR filter
def calculate_frequency_response(filter_weights, sample_rate):
    frequencies_hz, frequency_response = freqz(filter_weights, worN=2000, fs=sample_rate)
    magnitude_db = 20 * np.log10(np.abs(frequency_response) + 1e-12)
    return frequencies_hz, magnitude_db

# Acoustic noise left at error mic after learned filter's anti-noise is applied
def calculate_residual(noise_at_error_mic, reference_noise, filter_weights, secondary_path=None):
    anti_noise = lfilter(filter_weights, 1, reference_noise)
    # Pass secondary_path for FxLMS, where anti-noise travels through speaker-to-mic path before it reaches error mic
    if secondary_path is not None:
        anti_noise = lfilter(secondary_path, 1, anti_noise)
    return noise_at_error_mic - anti_noise
 
# Noise reduction in dB. Positive values mean the noise got quieter
def calculate_noise_reduction(noise_before, residual_after):
    power_before = np.mean(noise_before ** 2)
    power_after = np.mean(residual_after ** 2)
    return 10 * np.log10(power_before / power_after)
 
 
## Plotting functions
 
def plot_convergence(squared_errors_by_name, num_samples, smoothing_window,
                     noise_floor_db=None, save_path=None):
    plt.figure()
    for name, squared_error in squared_errors_by_name.items():
        smoothed = smooth_squared_error(squared_error[:num_samples], smoothing_window)
        iterations = np.arange(smoothing_window - 1, num_samples)
        plt.plot(iterations, to_db_power(smoothed), label=name)
    if noise_floor_db is not None:
        plt.axhline(noise_floor_db, color='gray', linestyle=':', label='Error mic noise floor')
    plt.xlabel('Iteration')
    plt.ylabel(f'Mean squared error (dB, {smoothing_window}-sample average)')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
 
def plot_frequency_responses(weights_by_name, primary_path, sample_rate, save_path=None):
    plt.figure()
    for name, weights in weights_by_name.items():
        line_style, line_width = LINE_STYLES.get(name, ('-', 2))
        frequencies_hz, magnitude_db = calculate_frequency_response(weights, sample_rate)
        plt.plot(frequencies_hz, magnitude_db,
                 linestyle=line_style, linewidth=line_width, label=name)
 
    frequencies_hz, primary_path_db = calculate_frequency_response(primary_path, sample_rate)
    plt.plot(frequencies_hz, primary_path_db,
             color='red', linestyle=':', linewidth=2,
             label='Primary path (target for LMS/NLMS)')
 
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
 
# Overlay the PSD before cancellation with the residual PSD for each algorithm
def plot_power_spectral_densities(noise_before, residuals_by_name, sample_rate, save_path=None):
    plt.figure(figsize=(10, 6))
 
    frequencies_hz, psd_before = welch(noise_before, fs=sample_rate)
    plt.semilogy(frequencies_hz, psd_before, color='black', linewidth=2,
                 label='Before cancellation')
 
    for name, residual in residuals_by_name.items():
        frequencies_hz, psd_after = welch(residual, fs=sample_rate)
        plt.semilogy(frequencies_hz, psd_after, label=f'After {name}')
 
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (power/Hz)')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
 
## Audio output

# Normalization
def write_audio_files(audio_by_filename, sample_rate, peak_limit=0.9):
    peak = max(np.max(np.abs(audio)) for audio in audio_by_filename.values())
    gain = min(1.0, peak_limit / peak)
    for filename, audio in audio_by_filename.items():
        sf.write(filename, audio * gain, sample_rate)
    return gain
 
def calculate_filter_performance(reference_noise, noise_at_error_mic, primary_path, secondary_path, secondary_path_estimate, step_sizes, num_taps_values):
    convergence_speed = {}
    steady_state_error = {}
    divergence_threshold = {}
    residuals_by_name = {}

    for step_size in step_sizes:
        for num_taps in num_taps_values:
            weights_lms, squared_error_lms = lms_filter(
                reference_noise, noise_at_error_mic, step_size=step_size, num_taps=num_taps)
            weights_nlms, squared_error_nlms = nlms_filter(
                reference_noise, noise_at_error_mic, step_size=step_size, num_taps=num_taps)
            weights_fxlms, squared_error_fxlms = fxlms_filter(
                reference_noise, noise_at_error_mic, step_size=step_size, num_taps=num_taps,
                secondary_path=secondary_path, secondary_path_estimate=secondary_path_estimate)

            convergence_speed[(step_size, num_taps)] = np.mean(np.diff(squared_error_lms[:200]))
            steady_state_error[(step_size, num_taps)] = np.mean(squared_error_lms[200:])
            divergence_threshold[(step_size, num_taps)] = np.max(squared_error_lms)

            residuals_lms = calculate_residual(noise_at_error_mic, reference_noise, weights_lms)
            residuals_nlms = calculate_residual(noise_at_error_mic, reference_noise, weights_nlms)
            residuals_fxlms = calculate_residual(noise_at_error_mic, reference_noise, weights_fxlms, secondary_path)

            # Store residuals in dictionary
            residuals_by_name['LMS'] = residuals_lms
            residuals_by_name['NLMS'] = residuals_nlms
            residuals_by_name['FxLMS'] = residuals_fxlms

    return convergence_speed, steady_state_error, divergence_threshold, residuals_by_name

def plot_results(convergence_speed, steady_state_error, divergence_threshold, save_path=None):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    for step_size, num_taps in convergence_speed:
        plt.plot(num_taps, convergence_speed[(step_size, num_taps)], label=f'Step size: {step_size}')
    plt.xlabel('Number of Taps')
    plt.ylabel('Convergence Speed (dB/iteration)')
    plt.legend()

    plt.subplot(1, 3, 2)
    for step_size, num_taps in steady_state_error:
        plt.plot(num_taps, steady_state_error[(step_size, num_taps)], label=f'Step size: {step_size}')
    plt.xlabel('Number of Taps')
    plt.ylabel('Steady-State Error (dB)')
    plt.legend()

    plt.subplot(1, 3, 3)
    for step_size, num_taps in divergence_threshold:
        plt.plot(num_taps, divergence_threshold[(step_size, num_taps)], label=f'Step size: {step_size}')
    plt.xlabel('Number of Taps')
    plt.ylabel('Divergence Threshold (dB)')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def main():
    # Signal setup
    reference_noise = generate_reference_noise(SAMPLE_RATE, DURATION_SECONDS, RANDOM_SEED)
    clean_signal = np.zeros_like(reference_noise)
 
    # Primary path: how noise travels from its source to the error mic
    primary_path = np.array([0.0, 0.0, 0.8, 0.4, 0.2, 0.1])
    noise_at_error_mic = lfilter(primary_path, 1, reference_noise)
 
    # Secondary path: how anti-noise travels from the speaker to the error mic
    secondary_path = np.array([0.0, 0.0, 0.5, 0.3, 0.1])
    secondary_path_estimate = secondary_path.copy()  # perfect estimate for now
 
    # Vary step size and number of taps
    step_sizes = np.linspace(0.01, 0.1, 10)
    num_taps_values = np.linspace(16, 64, 10)
    num_taps_values = np.round(num_taps_values).astype(int)
    num_taps_values = np.unique(num_taps_values)  # Ensure unique values

    # Calculate filter performance
    convergence_speed, steady_state_error, divergence_threshold, residuals_by_name = calculate_filter_performance(
        reference_noise, noise_at_error_mic, primary_path, secondary_path, secondary_path_estimate, step_sizes, num_taps_values)

    # Plot results
    plot_results(convergence_speed, steady_state_error, divergence_threshold)

    # Audio
    gain = write_audio_files({
        'noise_before_cancellation.wav': noise_at_error_mic,
        'residual_after_lms.wav': residuals_by_name['LMS'],
        'residual_after_nlms.wav': residuals_by_name['NLMS'],
        'residual_after_fxlms.wav': residuals_by_name['FxLMS'],
    }, SAMPLE_RATE)
    if gain < 1.0:
        print(f'Audio files scaled by {gain:.2f} to avoid clipping (same gain for all files)')
 
 
if __name__ == '__main__':
    main()