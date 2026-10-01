import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz, welch
import soundfile as sf

## Configuration
SAMPLE_RATE = 16000
DURATION_SECONDS = 1
NUM_TAPS = 32
RANDOM_SEED = 0
CONVERGENCE_PLOT_SAMPLES = 2000

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

# Convert squared error to dB, with a floor to avoid log(0)
def squared_error_to_db(squared_error):
    return 10 * np.log10(squared_error + 1e-12)
 
# Return frequencies in Hz and magnitude response in dB for an FIR filter
def calculate_frequency_response(filter_weights, sample_rate):
    frequencies_hz, frequency_response = freqz(filter_weights, worN=2000, fs=sample_rate)
    magnitude_db = 20 * np.log10(np.abs(frequency_response) + 1e-12)
    return frequencies_hz, magnitude_db
 
# Noise left at error mic after learned filter's anti-noise is applied
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
 
def plot_convergence(squared_errors_by_name, num_samples, save_path=None):
    plt.figure()
    for name, squared_error in squared_errors_by_name.items():
        plt.plot(squared_error_to_db(squared_error[:num_samples]), label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Squared error (dB)')
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
 
def write_audio_files(audio_by_filename, sample_rate):
    for filename, audio in audio_by_filename.items():
        sf.write(filename, audio, sample_rate)
 
 
## Main
 
def main():
    # Signal setup
    reference_noise = generate_reference_noise(SAMPLE_RATE, DURATION_SECONDS, RANDOM_SEED)
 
    # Primary path: how noise travels from its source to the error mic
    primary_path = np.array([0.0, 0.0, 0.8, 0.4, 0.2, 0.1])
    noise_at_error_mic = lfilter(primary_path, 1, reference_noise)
 
    # Secondary path: how anti-noise travels from the speaker to the error mic
    secondary_path = np.array([0.0, 0.0, 0.5, 0.3, 0.1])
    secondary_path_estimate = secondary_path.copy()  # perfect estimate for now
 
    # Run filters
    weights_lms, squared_error_lms = lms_filter(
        reference_noise, noise_at_error_mic, step_size=0.01, num_taps=NUM_TAPS)
    weights_nlms, squared_error_nlms = nlms_filter(
        reference_noise, noise_at_error_mic, step_size=0.5, num_taps=NUM_TAPS)
    weights_fxlms, squared_error_fxlms = fxlms_filter(
        reference_noise, noise_at_error_mic, step_size=0.005, num_taps=NUM_TAPS,
        secondary_path=secondary_path, secondary_path_estimate=secondary_path_estimate)
 
    squared_errors_by_name = {
        'LMS': squared_error_lms,
        'NLMS': squared_error_nlms,
        'FxLMS': squared_error_fxlms,
    }
    weights_by_name = {
        'LMS': weights_lms,
        'NLMS': weights_nlms,
        'FxLMS': weights_fxlms,
    }
 
    # Residual noise after cancellation
    residuals_by_name = {
        'LMS': calculate_residual(noise_at_error_mic, reference_noise, weights_lms),
        'NLMS': calculate_residual(noise_at_error_mic, reference_noise, weights_nlms),
        'FxLMS': calculate_residual(noise_at_error_mic, reference_noise, weights_fxlms,
                                    secondary_path=secondary_path),
    }
 
    # Noise reduction summary
    for name, residual in residuals_by_name.items():
        noise_reduction_db = calculate_noise_reduction(noise_at_error_mic, residual)
        print(f'{name}: {noise_reduction_db:.1f} dB noise reduction')
 
    # Plots
    plot_convergence(squared_errors_by_name, CONVERGENCE_PLOT_SAMPLES,
                     save_path='convergence_plot.png')
    plot_frequency_responses(weights_by_name, primary_path, SAMPLE_RATE,
                             save_path='frequency_response.png')
    plot_power_spectral_densities(noise_at_error_mic, residuals_by_name, SAMPLE_RATE,
                                  save_path='power_spectral_density.png')
    plt.show()
 
    # Audio
    write_audio_files({
        'noise_before_cancellation.wav': noise_at_error_mic,
        'residual_after_lms.wav': residuals_by_name['LMS'],
        'residual_after_nlms.wav': residuals_by_name['NLMS'],
        'residual_after_fxlms.wav': residuals_by_name['FxLMS'],
    }, SAMPLE_RATE)
 
 
if __name__ == '__main__':
    main()