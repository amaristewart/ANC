import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.signal import lfilter, freqz, welch
from numba.core.errors import NumbaPerformanceWarning
from anc_core import run_anc, reduction_db
import soundfile as sf

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

## Configuration
SAMPLE_RATE = 16000
DURATION_SECONDS = 1
NUM_TAPS = 32
RANDOM_SEED = 0
CONVERGENCE_PLOT_SAMPLES = 5000
SMOOTHING_WINDOW = 100
SENSOR_NOISE_SNR_DB = 30

# Output folders sit next to src, so ANC/src/main.py writes to ANC/output/<scenario>/audio and plots
# Paths are built from this file's location, so they do not depend on where the script is run from
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'

# Open the plot windows at the end of the run. Plots are always saved to disk
SHOW_PLOTS = False

# Primary path: how noise travels from its source to the error mic (simulated)
PRIMARY_PATH = np.array([0.0, 0.0, 0.8, 0.4, 0.2, 0.1])

# Secondary path scenarios: how anti-noise travels from the speaker to the error mic (simulated)
# Every scenario uses the same reference signal, primary path and sensor noise, so only this path changes
#   no_delay: mild coloration, no propagation delay. LMS, NLMS and FxLMS can all be compared
#   delay_2:  2 sample delay. Plain LMS and NLMS fail because the update ignores the path's phase
#   delay_4:  4 sample delay, longer than the primary path's 2. The anti-noise cannot arrive in time
SECONDARY_PATH_SCENARIOS = {
    'no_delay': np.array([0.5, 0.3, 0.1]),
    'delay_2': np.array([0.0, 0.0, 0.5, 0.3, 0.1]),
    'delay_4': np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.1]),
}

# Step sizes for the single baseline comparison
BASELINE_STEP_SIZES = {'LMS': 0.01, 'NLMS': 0.5, 'FxLMS': 0.02}

# Parameter sweep. Each algorithm gets its own step size grid because their stable ranges differ:
# LMS and FxLMS depend on input power and tap count. NLMS is stable for 0 < step size < 2 when there is no
# secondary path, and with one in the loop that limit shifts with the path's gain, so 2.1 may not diverge
RUN_PARAMETER_SWEEP = True
SWEEP_NUM_TAPS = [8, 16, 32, 64]
SWEEP_STEP_SIZES = {
    'LMS': np.logspace(-2.3, -0.5, 8),
    'NLMS': np.array([0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 1.9, 2.1]),
    'FxLMS': np.logspace(-2.3, -0.5, 8),
}
# A run counts as converged when its smoothed error is within this many dB of the error mic noise floor
CONVERGENCE_TOLERANCE_DB = 3

# Steady-state reduction is measured over the last quarter of the run, after the startup transient
TAIL_FRACTION_START = 0.75

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
def run_parameter_sweep(reference_noise, noise_at_error_mic, sensor_noise,
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
                result = run_anc(name, reference_noise, noise_at_error_mic, sensor_noise,
                                 num_taps, step_size, secondary_path, secondary_path_estimate)
                with np.errstate(over='ignore', invalid='ignore'):
                    squared_error = result['error'] ** 2

                if is_diverged(squared_error, SMOOTHING_WINDOW):
                    diverged[tap_index, step_index] = True
                    continue

                convergence_iterations[tap_index, step_index] = calculate_convergence_iterations(
                    squared_error, SMOOTHING_WINDOW, sensor_noise_power, CONVERGENCE_TOLERANCE_DB)
                # Steady-state reduction from the actual adaptive run, last 25% of samples
                tail_start = int(TAIL_FRACTION_START * len(noise_at_error_mic))
                noise_reduction_db[tap_index, step_index] = reduction_db(
                    noise_at_error_mic, result['residual'], tail_start)

            print(f'{name}: finished {num_taps} taps')

        results[name] = {
            'diverged': diverged,
            'convergence_iterations': convergence_iterations,
            'noise_reduction_db': noise_reduction_db,
        }
    return results

## Plotting functions

def plot_convergence(squared_errors_by_name, num_samples, smoothing_window,
                     noise_floor_db=None, save_path=None, title=None):
    plt.figure()
    if title:
        plt.title(title)
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

def plot_frequency_responses(weights_by_name, primary_path, secondary_path, sample_rate,
                             save_path=None, title=None):
    plt.figure()
    if title:
        plt.title(title)
    for name, weights in weights_by_name.items():
        line_style, line_width = LINE_STYLES.get(name, ('-', 2))
        frequencies_hz, magnitude_db = calculate_frequency_response(weights, sample_rate)
        plt.plot(frequencies_hz, magnitude_db,
                 linestyle=line_style, linewidth=line_width, label=name)

    # With the secondary path in the loop, the ideal controller is W = P / S
    frequencies_hz, primary_response = freqz(primary_path, worN=2000, fs=sample_rate)
    _, secondary_response = freqz(secondary_path, worN=2000, fs=sample_rate)
    ideal_db = 20 * np.log10(np.abs(primary_response / secondary_response) + 1e-12)
    plt.plot(frequencies_hz, ideal_db, color='red', linestyle=':', linewidth=2,
             label='Ideal controller (P / S)')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Overlay the PSD before cancellation with the residual PSD for each algorithm
def plot_power_spectral_densities(noise_before, residuals_by_name, sample_rate, save_path=None, title=None):
    plt.figure(figsize=(10, 6))
    if title:
        plt.title(title)

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
    if not np.all(diverged):
        plt.colorbar(image, ax=axis)

def plot_sweep_results(results, step_sizes_by_name, num_taps_values, num_samples, save_path=None, title=None):
    figure, axes = plt.subplots(len(results), 2, figsize=(14, 3.4 * len(results)))
    if title:
        figure.suptitle(title)
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
    figure.tight_layout(rect=(0, 0.02, 1, 0.97 if title else 1))
    if save_path:
        figure.savefig(save_path, dpi=300, bbox_inches='tight')

## Output folders

# Create the output folders for one scenario if they do not exist yet
def create_output_folders(scenario_name):
    audio_dir = OUTPUT_DIR / scenario_name / 'audio'
    plots_dir = OUTPUT_DIR / scenario_name / 'plots'
    audio_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir, plots_dir

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
# Returns one summary row per algorithm for the cross-scenario table
def run_baseline_comparison(scenario_name, audio_dir, plots_dir, reference_noise, noise_at_error_mic,
                            sensor_noise, sensor_noise_power, primary_path, secondary_path,
                            secondary_path_estimate):
    weights_by_name = {}
    squared_errors_by_name = {}
    residuals_by_name = {}
    summary_rows = []
    tail_start = int(TAIL_FRACTION_START * len(noise_at_error_mic))

    for name, step_size in BASELINE_STEP_SIZES.items():
        result = run_anc(name, reference_noise, noise_at_error_mic, sensor_noise,
                         NUM_TAPS, step_size, secondary_path, secondary_path_estimate)
        with np.errstate(over='ignore', invalid='ignore'):
            squared_error = result['error'] ** 2
        row = {'scenario': scenario_name, 'algorithm': name, 'step_size': step_size,
               'status': 'diverged', 'whole_run_db': None, 'last_25_percent_db': None,
               'iterations_to_noise_floor': None}
        if is_diverged(squared_error, SMOOTHING_WINDOW):
            print(f'{name}: diverged at step size {step_size}, excluded from plots and audio')
            summary_rows.append(row)
            continue

        iterations = calculate_convergence_iterations(
            squared_error, SMOOTHING_WINDOW, sensor_noise_power, CONVERGENCE_TOLERANCE_DB)
        row.update(status='ok',
                   whole_run_db=round(float(reduction_db(noise_at_error_mic, result['residual'])), 1),
                   last_25_percent_db=round(float(reduction_db(noise_at_error_mic, result['residual'], tail_start)), 1),
                   iterations_to_noise_floor=None if iterations == len(squared_error) else iterations)
        summary_rows.append(row)
        weights_by_name[name] = result['weights']
        squared_errors_by_name[name] = squared_error
        residuals_by_name[name] = result['residual']

    if not residuals_by_name:
        print('No algorithm converged at its baseline step size, nothing to plot')
        return summary_rows

    print(f'Error mic SNR: {SENSOR_NOISE_SNR_DB} dB')
    for row in summary_rows:
        if row['status'] == 'ok':
            iterations = row['iterations_to_noise_floor']
            reached = f'{iterations} iterations to noise floor' if iterations else 'did not reach the noise floor'
            print(f"{row['algorithm']}: {row['whole_run_db']:.1f} dB whole run, "
                  f"{row['last_25_percent_db']:.1f} dB last 25%, {reached}")

    title = f'Simulated paths, scenario: {scenario_name}'
    plot_convergence(squared_errors_by_name, CONVERGENCE_PLOT_SAMPLES, SMOOTHING_WINDOW,
                     noise_floor_db=to_db_power(sensor_noise_power),
                     save_path=plots_dir / 'convergence_plot.png', title=title)
    plot_frequency_responses(weights_by_name, primary_path, secondary_path, SAMPLE_RATE,
                             save_path=plots_dir / 'frequency_response.png', title=title)
    plot_power_spectral_densities(noise_at_error_mic, residuals_by_name, SAMPLE_RATE,
                                  save_path=plots_dir / 'power_spectral_density.png', title=title)

    audio_by_filename = {audio_dir / 'noise_before_cancellation.wav': noise_at_error_mic}
    for name, residual in residuals_by_name.items():
        audio_by_filename[audio_dir / f'residual_after_{name.lower()}.wav'] = residual
    gain = write_audio_files(audio_by_filename, SAMPLE_RATE)
    if gain < 1.0:
        print(f'Audio files scaled by {gain:.2f} to avoid clipping (same gain for all files)')
    print(f'Saved baseline plots to {plots_dir}', flush=True)
    print(f'Saved audio to {audio_dir}', flush=True)
    return summary_rows

# Print and save the baseline results of every scenario in one table
def save_summary_table(summary_rows, path):
    columns = ['scenario', 'algorithm', 'step_size', 'status', 'whole_run_db',
               'last_25_percent_db', 'iterations_to_noise_floor']
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    print('\nSummary at baseline step sizes (noise reduction in dB, last 25% of the run)')
    print(f'{"scenario":10s} ' + ' '.join(f'{name:>10s}' for name in BASELINE_STEP_SIZES))
    for scenario_name in SECONDARY_PATH_SCENARIOS:
        cells = []
        for name in BASELINE_STEP_SIZES:
            row = next(r for r in summary_rows if r['scenario'] == scenario_name and r['algorithm'] == name)
            cells.append('diverged' if row['status'] == 'diverged' else f"{row['last_25_percent_db']:.1f}")
        print(f'{scenario_name:10s} ' + ' '.join(f'{cell:>10s}' for cell in cells))
    print(f'Saved summary table to {path}', flush=True)

def main():
    print(f'Running {Path(__file__).resolve()}', flush=True)
    print(f'Output folder: {OUTPUT_DIR}', flush=True)
    rng = np.random.default_rng(RANDOM_SEED)
    num_samples = int(SAMPLE_RATE * DURATION_SECONDS)

    # Signals are generated once and shared by every scenario
    reference_noise = generate_reference_noise(num_samples, rng)
    noise_at_error_mic = lfilter(PRIMARY_PATH, 1, reference_noise)
    sensor_noise = generate_sensor_noise(noise_at_error_mic, SENSOR_NOISE_SNR_DB, rng)
    sensor_noise_power = np.mean(sensor_noise ** 2)

    summary_rows = []
    for scenario_name, secondary_path in SECONDARY_PATH_SCENARIOS.items():
        print(f'\n=== Scenario: {scenario_name} ===', flush=True)
        audio_dir, plots_dir = create_output_folders(scenario_name)
        secondary_path_estimate = secondary_path.copy()  # perfect estimate for now

        summary_rows += run_baseline_comparison(
            scenario_name, audio_dir, plots_dir, reference_noise, noise_at_error_mic, sensor_noise,
            sensor_noise_power, PRIMARY_PATH, secondary_path, secondary_path_estimate)

        if RUN_PARAMETER_SWEEP:
            print('Running parameter sweep', flush=True)
            results = run_parameter_sweep(
                reference_noise, noise_at_error_mic, sensor_noise, sensor_noise_power,
                secondary_path, secondary_path_estimate, SWEEP_STEP_SIZES, SWEEP_NUM_TAPS)
            plot_sweep_results(results, SWEEP_STEP_SIZES, SWEEP_NUM_TAPS, num_samples,
                               save_path=plots_dir / 'parameter_sweep.png',
                               title=f'Simulated paths, scenario: {scenario_name}')
            print(f'Saved sweep plot to {plots_dir}', flush=True)

        if not SHOW_PLOTS:
            plt.close('all')

    save_summary_table(summary_rows, OUTPUT_DIR / 'scenario_summary.csv')

    if SHOW_PLOTS:
        plt.show()

if __name__ == '__main__':
    main()