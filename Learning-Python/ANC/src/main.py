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

# Step sizes for the single baseline comparison
BASELINE_STEP_SIZES = {'LMS': 0.01, 'NLMS': 0.5, 'FxLMS': 0.02}

# Parameter sweep. Each algorithm gets its own step size grid because their stable ranges differ:
# LMS and FxLMS depend on input power and tap count, while NLMS is stable for 0 < step size < 2
RUN_PARAMETER_SWEEP = True
SWEEP_NUM_TAPS = [8, 16, 32, 64]
SWEEP_STEP_SIZES = {
    'LMS': np.logspace(-2.3, -0.5, 8),
    'NLMS': np.array([0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 1.9, 2.1]),
    'FxLMS': np.logspace(-2.3, -0.5, 8),
}
# A run counts as converged when its smoothed error is within this many dB of the error mic noise floor
CONVERGENCE_TOLERANCE_DB = 3

LINE_STYLES = {
    'LMS': ('-', 4),
    'NLMS': ('--', 2),
    'FxLMS': ('-', 2),
}

## Signal generation

# White noise picked up by reference mic
def generate_reference_noise(num_samples, rng):
    return rng.standard_normal(num_samples) / np.sqrt(2)

# White noise from error mic, scaled to a target SNR against clean_signal
def generate_sensor_noise(clean_signal, snr_db, rng):
    signal_power = np.mean(clean_signal ** 2)
    sensor_noise_power = signal_power / 10 ** (snr_db / 10)
    return rng.standard_normal(len(clean_signal)) * np.sqrt(sensor_noise_power)

## Helper functions

# Prepend zeros so a full-length window exists at first sample
def pad_with_zeros(signal, num_taps):
    return np.concatenate([np.zeros(num_taps - 1), signal])

# Return last num_taps samples ending at sample_index, newest first
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

# Run one adaptive filter by name
def run_adaptive_filter(name, reference_signal, desired_signal, step_size, num_taps,
                        secondary_path, secondary_path_estimate):
    if name == 'LMS':
        return lms_filter(reference_signal, desired_signal, step_size, num_taps)
    if name == 'NLMS':
        return nlms_filter(reference_signal, desired_signal, step_size, num_taps)
    if name == 'FxLMS':
        return fxlms_filter(reference_signal, desired_signal, step_size, num_taps,
                            secondary_path, secondary_path_estimate)
    raise ValueError(f'Unknown algorithm: {name}')

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

# Residual for a named algorithm. Only FxLMS sends its output through the secondary path
def calculate_algorithm_residual(name, noise_at_error_mic, reference_noise, filter_weights, secondary_path):
    path = secondary_path if name == 'FxLMS' else None
    return calculate_residual(noise_at_error_mic, reference_noise, filter_weights, path)

# Noise reduction in dB. Positive values mean the noise got quieter
def calculate_noise_reduction(noise_before, residual_after):
    power_before = np.mean(noise_before ** 2)
    power_after = np.mean(residual_after ** 2)
    return 10 * np.log10(power_before / power_after)

# A run has diverged if its error became non-finite or ended larger than it started
def is_diverged(squared_error, window):
    if not np.all(np.isfinite(squared_error)):
        return True
    return np.mean(squared_error[-window:]) > np.mean(squared_error[:window])

# First iteration where the smoothed error is within tolerance_db of the error mic noise floor
# Returns the run length if the run never gets there
def calculate_convergence_iterations(squared_error, window, noise_floor_power, tolerance_db):
    smoothed = smooth_squared_error(squared_error, window)
    reached = smoothed <= noise_floor_power * 10 ** (tolerance_db / 10)
    if not np.any(reached):
        return len(squared_error)
    return int(np.argmax(reached)) + window - 1

## Parameter sweep

# Run every algorithm over its step size grid and every tap count
# Returns, per algorithm, arrays shaped (num tap values, num step sizes)
def run_parameter_sweep(reference_noise, noise_at_error_mic, measured_noise_at_error_mic,
                        sensor_noise_power, secondary_path, secondary_path_estimate,
                        step_sizes_by_name, num_taps_values):
    results = {}
    for name, step_sizes in step_sizes_by_name.items():
        shape = (len(num_taps_values), len(step_sizes))
        diverged = np.zeros(shape, dtype=bool)
        convergence_iterations = np.full(shape, np.nan)
        noise_reduction_db = np.full(shape, np.nan)

        for tap_index, num_taps in enumerate(num_taps_values):
            for step_index, step_size in enumerate(step_sizes):
                # Diverging runs overflow by design, so their warnings are silenced
                with np.errstate(over='ignore', invalid='ignore'):
                    weights, squared_error = run_adaptive_filter(
                        name, reference_noise, measured_noise_at_error_mic, step_size, num_taps,
                        secondary_path, secondary_path_estimate)

                if is_diverged(squared_error, SMOOTHING_WINDOW):
                    diverged[tap_index, step_index] = True
                    continue

                residual = calculate_algorithm_residual(
                    name, noise_at_error_mic, reference_noise, weights, secondary_path)
                convergence_iterations[tap_index, step_index] = calculate_convergence_iterations(
                    squared_error, SMOOTHING_WINDOW, sensor_noise_power, CONVERGENCE_TOLERANCE_DB)
                noise_reduction_db[tap_index, step_index] = calculate_noise_reduction(
                    noise_at_error_mic, residual)

            print(f'{name}: finished {num_taps} taps')

        results[name] = {
            'diverged': diverged,
            'convergence_iterations': convergence_iterations,
            'noise_reduction_db': noise_reduction_db,
        }
    return results

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

# One heatmap of a sweep metric. Rows are tap counts, columns are step sizes
# Diverged runs are gray and labeled "div"
def plot_sweep_heatmap(axis, values, diverged, step_sizes, num_taps_values,
                       colormap_name, title, number_format):
    colormap = plt.get_cmap(colormap_name).copy()
    colormap.set_bad('lightgray')
    image = axis.imshow(np.ma.masked_invalid(values), cmap=colormap, aspect='auto', origin='lower')

    axis.set_xticks(range(len(step_sizes)))
    axis.set_xticklabels([f'{step_size:.3g}' for step_size in step_sizes])
    axis.set_yticks(range(len(num_taps_values)))
    axis.set_yticklabels(num_taps_values)
    axis.set_xlabel('Step size')
    axis.set_ylabel('Number of taps')
    axis.set_title(title)

    for tap_index in range(len(num_taps_values)):
        for step_index in range(len(step_sizes)):
            if diverged[tap_index, step_index]:
                label = 'div'
            else:
                label = number_format.format(values[tap_index, step_index])
            axis.text(step_index, tap_index, label, ha='center', va='center', fontsize=7,
                      bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.6, linewidth=0))
    plt.colorbar(image, ax=axis)

def plot_sweep_results(results, step_sizes_by_name, num_taps_values, num_samples, save_path=None):
    figure, axes = plt.subplots(len(results), 2, figsize=(14, 3.4 * len(results)))
    for row, (name, result) in enumerate(results.items()):
        step_sizes = step_sizes_by_name[name]
        plot_sweep_heatmap(axes[row, 0], result['convergence_iterations'], result['diverged'],
                           step_sizes, num_taps_values, 'viridis_r',
                           f'{name}: iterations to reach noise floor', '{:.0f}')
        plot_sweep_heatmap(axes[row, 1], result['noise_reduction_db'], result['diverged'],
                           step_sizes, num_taps_values, 'viridis',
                           f'{name}: noise reduction (dB)', '{:.1f}')
    figure.text(0.5, 0.005,
                f'div = diverged. {num_samples} iterations = did not reach the noise floor within the run.',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(0, 0.02, 1, 1))
    if save_path:
        figure.savefig(save_path, dpi=300, bbox_inches='tight')

## Audio output

# Write WAV files without clipping
# If any signal exceeds the peak limit, every file is scaled by the same gain so the level difference between files is preserved
def write_audio_files(audio_by_filename, sample_rate, peak_limit=0.9):
    peak = max(np.max(np.abs(audio)) for audio in audio_by_filename.values())
    if not np.isfinite(peak):
        raise ValueError('Audio contains inf or NaN values, which usually means a filter diverged')
    gain = min(1.0, peak_limit / peak)
    for filename, audio in audio_by_filename.items():
        sf.write(filename, audio * gain, sample_rate)
    return gain

## Main stages

# One run of each algorithm at its baseline step size, with plots and audio
def run_baseline_comparison(reference_noise, noise_at_error_mic, measured_noise_at_error_mic,
                            sensor_noise_power, primary_path, secondary_path, secondary_path_estimate):
    weights_by_name = {}
    squared_errors_by_name = {}
    residuals_by_name = {}

    for name, step_size in BASELINE_STEP_SIZES.items():
        weights, squared_error = run_adaptive_filter(
            name, reference_noise, measured_noise_at_error_mic, step_size, NUM_TAPS,
            secondary_path, secondary_path_estimate)
        weights_by_name[name] = weights
        squared_errors_by_name[name] = squared_error
        # Acoustic residual excludes the mic's own noise, so it is what a listener at the error mic hears
        residuals_by_name[name] = calculate_algorithm_residual(
            name, noise_at_error_mic, reference_noise, weights, secondary_path)

    print(f'Error mic SNR: {SENSOR_NOISE_SNR_DB} dB')
    for name, residual in residuals_by_name.items():
        print(f'{name}: {calculate_noise_reduction(noise_at_error_mic, residual):.1f} dB noise reduction')

    plot_convergence(squared_errors_by_name, CONVERGENCE_PLOT_SAMPLES, SMOOTHING_WINDOW,
                     noise_floor_db=to_db_power(sensor_noise_power), save_path='convergence_plot.png')
    plot_frequency_responses(weights_by_name, primary_path, SAMPLE_RATE,
                             save_path='frequency_response.png')
    plot_power_spectral_densities(noise_at_error_mic, residuals_by_name, SAMPLE_RATE,
                                  save_path='power_spectral_density.png')

    gain = write_audio_files({
        'noise_before_cancellation.wav': noise_at_error_mic,
        'residual_after_lms.wav': residuals_by_name['LMS'],
        'residual_after_nlms.wav': residuals_by_name['NLMS'],
        'residual_after_fxlms.wav': residuals_by_name['FxLMS'],
    }, SAMPLE_RATE)
    if gain < 1.0:
        print(f'Audio files scaled by {gain:.2f} to avoid clipping (same gain for all files)')

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    num_samples = int(SAMPLE_RATE * DURATION_SECONDS)

    # Primary path: how noise travels from its source to the error mic
    primary_path = np.array([0.0, 0.0, 0.8, 0.4, 0.2, 0.1])

    # Secondary path: how anti-noise travels from the speaker to the error mic
    secondary_path = np.array([0.0, 0.0, 0.5, 0.3, 0.1])
    secondary_path_estimate = secondary_path.copy()  # perfect estimate for now

    # Signals
    reference_noise = generate_reference_noise(num_samples, rng)
    noise_at_error_mic = lfilter(primary_path, 1, reference_noise)
    sensor_noise = generate_sensor_noise(noise_at_error_mic, SENSOR_NOISE_SNR_DB, rng)
    sensor_noise_power = np.mean(sensor_noise ** 2)

    # The filters adapt on what the error mic actually measures, which includes its own noise
    measured_noise_at_error_mic = noise_at_error_mic + sensor_noise

    run_baseline_comparison(reference_noise, noise_at_error_mic, measured_noise_at_error_mic,
                            sensor_noise_power, primary_path, secondary_path, secondary_path_estimate)

    if RUN_PARAMETER_SWEEP:
        results = run_parameter_sweep(
            reference_noise, noise_at_error_mic, measured_noise_at_error_mic, sensor_noise_power,
            secondary_path, secondary_path_estimate, SWEEP_STEP_SIZES, SWEEP_NUM_TAPS)
        plot_sweep_results(results, SWEEP_STEP_SIZES, SWEEP_NUM_TAPS, num_samples,
                           save_path='parameter_sweep.png')

    plt.show()

if __name__ == '__main__':
    main()