import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz
import soundfile as sf

## Signal setup for testing
SAMPLE_RATE = 16000
t = np.arange(0, 1, 1/SAMPLE_RATE)
reference_noise = np.random.randn(len(t)) / np.sqrt(2) # noise picked up by reference mic

# Primary path: how noise travels from its source to error mic
primary_path = np.array([0.0, 0.0, 0.8, 0.4, 0.2, 0.1])
noise_at_error_mic = lfilter(primary_path, 1, reference_noise)

# Secondary path: how anti-noise travels from speaker to error mic
secondary_path = np.array([0.0, 0.0, 0.5, 0.3, 0.1])
secondary_path_estimate = secondary_path.copy() # perfect estimate for now

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


## Run filters

NUM_TAPS = 32
 
weights_lms, squared_error_lms = lms_filter(
    reference_noise, noise_at_error_mic, step_size=0.01, num_taps=NUM_TAPS)
weights_nlms, squared_error_nlms = nlms_filter(
    reference_noise, noise_at_error_mic, step_size=0.5, num_taps=NUM_TAPS)
weights_fxlms, squared_error_fxlms = fxlms_filter(
    reference_noise, noise_at_error_mic, step_size=0.005, num_taps=NUM_TAPS,
    secondary_path=secondary_path, secondary_path_estimate=secondary_path_estimate)

## Convergence plot

PLOT_SAMPLES = 2000
plt.figure()
plt.plot(10 * np.log10(squared_error_lms[:PLOT_SAMPLES] + 1e-12), label='LMS')
plt.plot(10 * np.log10(squared_error_nlms[:PLOT_SAMPLES] + 1e-12), label='NLMS')
plt.plot(10 * np.log10(squared_error_fxlms[:PLOT_SAMPLES] + 1e-12), label='FxLMS')
plt.xlabel('Iteration')
plt.ylabel('Squared error (dB)')
plt.legend()
plt.show()

## Frequency response of learned filters

plt.figure()
frequencies_hz, primary_path_response = freqz(primary_path, worN=2000, fs=SAMPLE_RATE)
 
for learned_weights, label, line_style, line_width in [
        (weights_lms, 'LMS', '-', 4),        # thick solid line so it stays visible under NLMS
        (weights_nlms, 'NLMS', '--', 2),
        (weights_fxlms, 'FxLMS', '-', 2)]:
    frequencies_hz, frequency_response = freqz(learned_weights, worN=2000, fs=SAMPLE_RATE)
    plt.plot(frequencies_hz, 20 * np.log10(np.abs(frequency_response) + 1e-12),
             linestyle=line_style, linewidth=line_width, label=label)

plt.plot(frequencies_hz, 20 * np.log10(np.abs(primary_path_response) + 1e-12),
         color='red', linestyle=':', linewidth=2, label='Primary path (target for LMS/NLMS)')
    
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.show()

## Residual audio after cancellation

residual_lms = noise_at_error_mic - lfilter(weights_lms, 1, reference_noise)
residual_nlms = noise_at_error_mic - lfilter(weights_nlms, 1, reference_noise)
residual_fxlms = noise_at_error_mic - lfilter(
    secondary_path, 1, lfilter(weights_fxlms, 1, reference_noise))
 
sf.write('noise_before_cancellation.wav', noise_at_error_mic, SAMPLE_RATE)
sf.write('residual_after_lms.wav', residual_lms, SAMPLE_RATE)
sf.write('residual_after_nlms.wav', residual_nlms, SAMPLE_RATE)
sf.write('residual_after_fxlms.wav', residual_fxlms, SAMPLE_RATE)