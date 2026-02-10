"""
Telegraph Transmission Theory Visualization
Based on Nyquist's "Certain Topics in Telegraph Transmission Theory" (1928)

Comprehensive educational application showing the complete signal journey
from Morse code encoding to reconstruction at the receiving end.
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import math

app = Flask(__name__)

# International Morse Code
MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
    '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.', ' ': ' '
}

def text_to_morse(text):
    """Convert text to Morse code string."""
    return ' '.join(MORSE_CODE.get(c.upper(), '') for c in text)

def morse_to_signal_elements(morse_code):
    """
    Convert Morse code to signal elements (magnitude factors).
    Dot = 1 unit ON, Dash = 3 units ON
    Between elements = 1 unit OFF
    Between letters = 3 units OFF
    Between words = 7 units OFF
    """
    elements = []
    element_descriptions = []

    i = 0
    while i < len(morse_code):
        char = morse_code[i]
        if char == '.':
            elements.append(1)  # Dot: 1 unit high
            element_descriptions.append({'type': 'dot', 'value': 1, 'duration': 1})
            if i + 1 < len(morse_code) and morse_code[i + 1] not in [' ']:
                elements.append(0)  # Inter-element gap
                element_descriptions.append({'type': 'gap', 'value': 0, 'duration': 1})
        elif char == '-':
            elements.extend([1, 1, 1])  # Dash: 3 units high
            element_descriptions.append({'type': 'dash', 'value': 1, 'duration': 3})
            if i + 1 < len(morse_code) and morse_code[i + 1] not in [' ']:
                elements.append(0)  # Inter-element gap
                element_descriptions.append({'type': 'gap', 'value': 0, 'duration': 1})
        elif char == ' ':
            # Check for word space (double space in morse representation)
            if i + 1 < len(morse_code) and morse_code[i + 1] == ' ':
                elements.extend([0, 0, 0, 0, 0, 0, 0])  # Word gap: 7 units
                element_descriptions.append({'type': 'word_gap', 'value': 0, 'duration': 7})
                i += 1  # Skip next space
            else:
                elements.extend([0, 0, 0])  # Letter gap: 3 units
                element_descriptions.append({'type': 'letter_gap', 'value': 0, 'duration': 3})
        i += 1

    return elements, element_descriptions

def generate_rectangular_wave(elements, samples_per_unit=50):
    """Generate rectangular (DC) telegraph wave from elements."""
    signal = []
    time_points = []
    for i, val in enumerate(elements):
        for j in range(samples_per_unit):
            signal.append(float(val))
            time_points.append((i * samples_per_unit + j) / samples_per_unit)
    return np.array(signal), np.array(time_points)

def compute_fourier_series(signal, num_harmonics=50):
    """
    Compute Fourier series decomposition with full mathematical details.
    Returns coefficients and step-by-step explanation.
    """
    n = len(signal)
    T = 1.0  # Period normalized to 1
    t = np.linspace(0, T, n, endpoint=False)

    # DC component (a_0)
    a_0 = np.mean(signal)

    components = [{
        'k': 0,
        'frequency': 0,
        'a_k': float(a_0),
        'b_k': 0.0,
        'magnitude': float(abs(a_0)),
        'phase': 0.0,
        'formula': f'a₀ = (1/T)∫f(t)dt = {a_0:.4f}',
        'explanation': 'DC component (average value of signal)'
    }]

    for k in range(1, num_harmonics + 1):
        # Fourier coefficients
        cos_term = np.cos(2 * np.pi * k * t / T)
        sin_term = np.sin(2 * np.pi * k * t / T)

        a_k = (2 / n) * np.sum(signal * cos_term)
        b_k = (2 / n) * np.sum(signal * sin_term)

        magnitude = np.sqrt(a_k**2 + b_k**2)
        phase = np.arctan2(b_k, a_k) if magnitude > 1e-10 else 0

        components.append({
            'k': k,
            'frequency': k,
            'a_k': float(a_k),
            'b_k': float(b_k),
            'magnitude': float(magnitude),
            'phase': float(phase),
            'formula': f'a_{k} = {a_k:.4f}, b_{k} = {b_k:.4f}',
            'explanation': f'Harmonic {k}: frequency = {k}ω₀'
        })

    return components

def build_signal_progressively(components, num_samples, max_harmonic):
    """Build signal step by step, adding one harmonic at a time."""
    t = np.linspace(0, 1, num_samples, endpoint=False)
    progressive_signals = []

    current_signal = np.zeros(num_samples)

    for k in range(min(max_harmonic + 1, len(components))):
        comp = components[k]
        if k == 0:
            current_signal = np.full(num_samples, comp['a_k'])
        else:
            harmonic = comp['a_k'] * np.cos(2 * np.pi * k * t) + \
                      comp['b_k'] * np.sin(2 * np.pi * k * t)
            current_signal = current_signal + harmonic

        progressive_signals.append({
            'harmonic': k,
            'signal': current_signal.copy().tolist(),
            'formula': f"Σ(k=0 to {k})" if k > 0 else "a₀ (DC)",
            'description': f"Signal with {k+1} component{'s' if k > 0 else ''}"
        })

    return progressive_signals

def apply_channel_effects(components, bandwidth, channel_type='ideal', noise_level=0):
    """
    Simulate channel transmission effects.
    Returns filtered components and explanation of what happened.
    """
    filtered = []
    channel_effects = []

    for comp in components:
        new_comp = comp.copy()
        k = comp['k']

        if channel_type == 'ideal':
            # Brick-wall filter
            if k <= bandwidth:
                attenuation = 1.0
                effect = 'passed'
            else:
                attenuation = 0.0
                effect = 'blocked'
        elif channel_type == 'butterworth':
            # Butterworth low-pass (smooth rolloff)
            order = 2
            attenuation = 1.0 / np.sqrt(1 + (k / max(bandwidth, 0.1))**(2 * order))
            effect = f'attenuated to {attenuation:.2%}'
        elif channel_type == 'rc':
            # RC low-pass filter
            attenuation = 1.0 / np.sqrt(1 + (k / max(bandwidth, 0.1))**2)
            effect = f'attenuated to {attenuation:.2%}'
        else:
            attenuation = 1.0
            effect = 'passed'

        new_comp['a_k'] *= attenuation
        new_comp['b_k'] *= attenuation
        new_comp['magnitude'] *= attenuation
        new_comp['attenuation'] = attenuation

        filtered.append(new_comp)
        channel_effects.append({
            'harmonic': k,
            'original_magnitude': comp['magnitude'],
            'filtered_magnitude': new_comp['magnitude'],
            'attenuation': attenuation,
            'effect': effect
        })

    return filtered, channel_effects

def reconstruct_signal(components, num_samples):
    """Reconstruct signal from Fourier components."""
    t = np.linspace(0, 1, num_samples, endpoint=False)
    signal = np.zeros(num_samples)

    for comp in components:
        k = comp['k']
        if k == 0:
            signal += comp['a_k']
        else:
            signal += comp['a_k'] * np.cos(2 * np.pi * k * t)
            signal += comp['b_k'] * np.sin(2 * np.pi * k * t)

    return signal

def sample_and_decode(signal, num_elements, samples_per_unit, threshold=0.5):
    """Sample signal at center of each time unit and decode."""
    samples = []
    decoded = []

    for i in range(num_elements):
        # Sample at center of time unit
        sample_idx = int((i + 0.5) * samples_per_unit)
        if sample_idx < len(signal):
            value = signal[sample_idx]
            samples.append({
                'element': i,
                'sample_index': sample_idx,
                'value': float(value),
                'decoded': 1 if value > threshold else 0
            })
            decoded.append(1 if value > threshold else 0)

    return samples, decoded

def calculate_intersymbol_interference(signal, num_elements, samples_per_unit):
    """Calculate ISI at each sampling point."""
    isi_analysis = []

    for i in range(num_elements):
        sample_idx = int((i + 0.5) * samples_per_unit)
        if sample_idx < len(signal):
            value = signal[sample_idx]
            # Ideal value would be 0 or 1
            ideal = round(value)
            isi = abs(value - ideal)
            isi_analysis.append({
                'element': i,
                'actual_value': float(value),
                'ideal_value': ideal,
                'isi': float(isi),
                'isi_percent': float(isi * 100)
            })

    return isi_analysis


# ============== Advanced Concepts from Nyquist's Paper ==============

def compute_shape_factor(wave_form, num_samples, num_harmonics=30):
    """
    Compute the Shape Factor F(ω) as defined by Nyquist (Appendix I).

    Shape factor depends ONLY on wave form, NOT on the intelligence transmitted.
    For rectangular wave (DC telegraphy):
        F(ω) = sin(ω/(2S)) / (ω/(2S))
    where S is the signaling speed (number of elements per unit time).

    The shape factor is the same for all signals using the same wave form,
    regardless of what message is being transmitted.
    """
    shape_factors = []
    n = len(wave_form)  # Number of signal elements = signaling speed S

    for k in range(num_harmonics + 1):
        # ω = 2πk/T where T is the repetition period
        # For rectangular wave at the k-th harmonic:
        # F(ω) = sin(πk/n) / (πk/n) - the sinc function
        if k == 0:
            # DC component: F(0) = 1 for rectangular wave
            f_magnitude = 1.0
            f_phase = 0.0
        else:
            # For k-th harmonic: argument is πk/n
            x = np.pi * k / n
            if abs(x) < 1e-10:
                f_magnitude = 1.0
            else:
                f_magnitude = abs(np.sin(x) / x)
            # Phase shift due to rectangular wave centered at t=T/2
            f_phase = -np.pi * k / n

        f_real = f_magnitude * np.cos(f_phase)
        f_imag = f_magnitude * np.sin(f_phase)

        shape_factors.append({
            'k': k,
            'frequency': k / n,  # Normalized frequency (fraction of signaling speed)
            'real': float(f_real),
            'imag': float(f_imag),
            'magnitude': float(f_magnitude),
            'phase': float(f_phase)
        })

    return shape_factors


def compute_discrimination_factor(elements, num_harmonics=30):
    """
    Compute the Discrimination Factor D(ω) as defined by Nyquist (Appendix I, equations 8-9).
    D(ω) depends only on the intelligence (magnitude factors) being transmitted.

    From the paper:
    a_k = (2/n) * Σ_{j=0}^{n-1} m_j * cos(2πkj/n)
    b_k = (2/n) * Σ_{j=0}^{n-1} m_j * sin(2πkj/n)

    The discrimination factor C(ω) = a_k - j*b_k (complex form)
    """
    n = len(elements)
    discrimination_factors = []

    for k in range(num_harmonics + 1):
        # Nyquist's formula: sum over all signal elements
        # a_k = (2/n) * Σ m_j * cos(2πkj/n)
        # b_k = (2/n) * Σ m_j * sin(2πkj/n)
        a_k = 0
        b_k = 0
        for j, m in enumerate(elements):
            angle = 2 * np.pi * k * j / n
            a_k += m * np.cos(angle)
            b_k += m * np.sin(angle)

        # Apply the 2/n factor (except for k=0 where it's 1/n)
        if k == 0:
            a_k = a_k / n
            b_k = b_k / n
        else:
            a_k = 2 * a_k / n
            b_k = 2 * b_k / n

        discrimination_factors.append({
            'k': k,
            'frequency': k / n,  # Normalized frequency
            'real': float(a_k),
            'imag': float(-b_k),  # Complex form: a_k - j*b_k
            'magnitude': float(np.sqrt(a_k**2 + b_k**2)),
            'phase': float(np.arctan2(-b_k, a_k))
        })

    return discrimination_factors


def demonstrate_band_redundancy(elements, num_harmonics=50):
    """
    Demonstrate Nyquist's Band Redundancy concept.
    Frequency bands of width equal to the signaling speed contain identical information.
    """
    n = len(elements)
    signal, _ = generate_rectangular_wave(elements, 50)
    components = compute_fourier_series(signal, num_harmonics)

    # Group components into bands of width = signaling speed (n components per band)
    bands = []
    num_bands = (num_harmonics // n) + 1

    for band_idx in range(min(num_bands, 5)):  # Show up to 5 bands
        start_k = band_idx * n
        end_k = min((band_idx + 1) * n, num_harmonics + 1)

        band_components = []
        for k in range(start_k, end_k):
            if k < len(components):
                band_components.append({
                    'k': k,
                    'k_in_band': k - start_k,
                    'magnitude': components[k]['magnitude'],
                    'a_k': components[k]['a_k'],
                    'b_k': components[k]['b_k']
                })

        bands.append({
            'band_index': band_idx,
            'frequency_range': f'{start_k} to {end_k - 1}',
            'components': band_components
        })

    return {
        'explanation': '''From Nyquist's paper: "When the frequency axis is divided into parts each
        being a frequency band of width numerically equal to the speed of signaling, it is found
        that the information conveyed in any band is substantially identical with that conveyed in
        any other; and the bands may be said to be mutually redundant."

        This is shown mathematically by the periodicity of the discrimination factor:
        C(ω + ω_s) = -C(ω), where ω_s is the angular signaling frequency.''',
        'signaling_speed': n,
        'bands': bands,
        'key_insight': '''The minimum band width required for unambiguous interpretation is
        substantially equal, numerically, to the speed of signaling and is substantially
        independent of the number of current values employed. (Nyquist's Key Result #3)'''
    }


def generate_ideal_shape_factors(num_points=200):
    """
    Generate ideal shape factors as shown in Nyquist's Figure 2.
    These are shape factors that produce nondistorting waves.
    """
    x = np.linspace(-2, 2, num_points)
    shapes = []

    # Shape (a): Rectangular in frequency domain = sinc in time
    # F(ω) = 1 for |ω| ≤ π, 0 otherwise → f(t) = sinc(πt)
    shape_a = np.where(np.abs(x) <= 1, 1.0, 0.0)
    shapes.append({
        'name': 'Ideal Low-pass (Brick-wall)',
        'description': 'F(ω) = 1 for |ω| ≤ ω_s, 0 otherwise. Produces sinc pulses in time domain.',
        'x': x.tolist(),
        'y': shape_a.tolist(),
        'formula': 'F(ω) = rect(ω/2ω_s)',
        'property': 'Zero ISI at sampling instants t = nT'
    })

    # Shape (b): Raised cosine
    for beta in [0.25, 0.5, 1.0]:
        shape_b = np.zeros_like(x)
        for i, xi in enumerate(x):
            xi_abs = abs(xi)
            if xi_abs <= (1 - beta):
                shape_b[i] = 1.0
            elif xi_abs <= (1 + beta):
                shape_b[i] = 0.5 * (1 + np.cos(np.pi * (xi_abs - (1 - beta)) / (2 * beta)))
            else:
                shape_b[i] = 0.0

        shapes.append({
            'name': f'Raised Cosine (β = {beta})',
            'description': f'Rolloff factor β = {beta}. Excess bandwidth = {beta*100}%',
            'x': x.tolist(),
            'y': shape_b.tolist(),
            'formula': f'cos²-rolloff with β = {beta}',
            'property': 'Zero ISI at sampling instants, smoother transitions'
        })

    return {
        'explanation': '''Nyquist's Figure 2 shows ideal shape factors for distortionless transmission.
        Curve (a) represents the basic ideal shape factor: constant from zero to the signaling frequency,
        then zero above. Curves like (c) show that shape factors SYMMETRIC about the signaling frequency
        (except for sign change) can be ADDED without affecting the wave value at sampling instants.
        This is because such symmetrical components contribute zero at the middle of each time unit
        due to the cos(πk/n) factor evaluating to zero at these points.''',
        'shapes': shapes,
        'key_insight': '''From Appendix II-A: "Shape factors symmetrical about the speed of signaling
        except for a change in sign do not contribute anything to the current value at the middle
        of the signal elements." The Nyquist criterion: F(ω) + F(ω_s - ω) = constant (Vestigial Sideband)'''
    }


def generate_carrier_modulation(elements, carrier_freq=10, samples_per_unit=100):
    """
    Demonstrate carrier wave modulation as discussed in Nyquist's paper.
    Shows baseband signal, carrier, and modulated signal with sidebands.
    """
    # Generate baseband signal
    baseband, time = generate_rectangular_wave(elements, samples_per_unit)
    total_samples = len(baseband)
    t = np.linspace(0, len(elements), total_samples)

    # Generate carrier
    carrier = np.cos(2 * np.pi * carrier_freq * t / len(elements))

    # Amplitude modulation: s(t) = [1 + m(t)] * cos(ωc*t)
    # where m(t) is the baseband signal normalized to [-1, 1]
    modulated = (0.5 + 0.5 * baseband) * carrier

    # Compute spectra
    baseband_fft = np.fft.fft(baseband)[:total_samples//2]
    modulated_fft = np.fft.fft(modulated)[:total_samples//2]
    freqs = np.fft.fftfreq(total_samples, 1/samples_per_unit)[:total_samples//2]

    return {
        'time': t.tolist(),
        'baseband': baseband.tolist(),
        'carrier': carrier.tolist(),
        'modulated': modulated.tolist(),
        'baseband_spectrum': np.abs(baseband_fft[:50]).tolist(),
        'modulated_spectrum': np.abs(modulated_fft[:50]).tolist(),
        'frequencies': freqs[:50].tolist(),
        'carrier_frequency': carrier_freq,
        'explanation': {
            'main': '''Nyquist showed that amplitude modulation creates upper and lower sidebands
            around the carrier frequency. Each sideband contains the complete baseband signal
            information, making carrier telegraphy use TWICE the bandwidth of DC telegraphy.''',
            'bandwidth': 'Baseband bandwidth B → Carrier needs 2B bandwidth',
            'sidebands': 'Upper sideband: fc + baseband, Lower sideband: fc - baseband'
        }
    }


def generate_single_sideband(elements, carrier_freq=10, samples_per_unit=100):
    """
    Demonstrate single sideband transmission to recover bandwidth efficiency.
    """
    baseband, _ = generate_rectangular_wave(elements, samples_per_unit)
    total_samples = len(baseband)
    t = np.linspace(0, len(elements), total_samples)

    # Carrier
    carrier = np.cos(2 * np.pi * carrier_freq * t / len(elements))
    carrier_quad = np.sin(2 * np.pi * carrier_freq * t / len(elements))

    # Double sideband (for comparison)
    dsb = (0.5 + 0.5 * baseband) * carrier

    # Single sideband (upper) - using Hilbert transform approximation
    # SSB = m(t)*cos(ωt) - m̂(t)*sin(ωt) where m̂ is Hilbert transform
    from scipy.signal import hilbert
    analytic = hilbert(baseband)
    ssb_upper = baseband * carrier - np.imag(analytic) * carrier_quad

    # Compute spectra
    dsb_fft = np.abs(np.fft.fft(dsb)[:total_samples//2])
    ssb_fft = np.abs(np.fft.fft(ssb_upper)[:total_samples//2])

    return {
        'time': t.tolist(),
        'dsb': dsb.tolist(),
        'ssb': ssb_upper.tolist(),
        'dsb_spectrum': dsb_fft[:50].tolist(),
        'ssb_spectrum': ssb_fft[:50].tolist(),
        'carrier_frequency': carrier_freq,
        'explanation': {
            'main': '''Nyquist described Single Sideband (SSB) transmission as a method to
            overcome the inefficiency of carrier telegraphy. By transmitting only one sideband,
            the bandwidth requirement is reduced to match DC telegraphy.''',
            'requirements': [
                'Phase correction at band edges',
                'Elimination of quadrature component',
                'Carrier positioned at edge of transmitted band'
            ],
            'benefit': 'Same bandwidth efficiency as DC telegraphy'
        }
    }


def generate_eye_diagram(elements, bandwidth, channel_type='ideal', num_periods=3, samples_per_unit=100):
    """
    Generate eye diagram to visualize intersymbol interference.
    The "eye" opening indicates signal quality and timing margin.
    """
    # Generate signal
    signal, _ = generate_rectangular_wave(elements, samples_per_unit)
    components = compute_fourier_series(signal, 50)
    filtered, _ = apply_channel_effects(components, bandwidth, channel_type)
    reconstructed = reconstruct_signal(filtered, len(signal))

    # Create eye diagram by overlaying symbol periods
    samples_per_symbol = samples_per_unit
    num_symbols = len(elements)

    # Collect traces for eye diagram (2 symbol periods)
    eye_traces = []
    trace_length = samples_per_symbol * 2

    for i in range(num_symbols - 1):
        start_idx = i * samples_per_symbol
        end_idx = start_idx + trace_length
        if end_idx <= len(reconstructed):
            trace = reconstructed[start_idx:end_idx].tolist()
            eye_traces.append(trace)

    # Calculate eye opening metrics
    if eye_traces:
        traces_array = np.array(eye_traces)
        center_samples = traces_array[:, samples_per_symbol // 2]
        high_samples = center_samples[center_samples > 0.5]
        low_samples = center_samples[center_samples <= 0.5]

        eye_height = 0
        if len(high_samples) > 0 and len(low_samples) > 0:
            eye_height = np.min(high_samples) - np.max(low_samples)

        eye_metrics = {
            'eye_height': float(max(0, eye_height)),
            'eye_opening_percent': float(max(0, eye_height) * 100),
            'min_high': float(np.min(high_samples)) if len(high_samples) > 0 else 0.5,
            'max_low': float(np.max(low_samples)) if len(low_samples) > 0 else 0.5,
            'quality': 'Good' if eye_height > 0.3 else 'Fair' if eye_height > 0.1 else 'Poor'
        }
    else:
        eye_metrics = {'eye_height': 0, 'eye_opening_percent': 0, 'quality': 'N/A'}

    return {
        'traces': eye_traces,
        'time_axis': list(range(trace_length)),
        'metrics': eye_metrics,
        'explanation': {
            'main': '''The eye diagram overlays consecutive symbol periods to visualize
            intersymbol interference. A wide, open "eye" indicates good signal quality
            and timing margins.''',
            'interpretation': {
                'eye_height': 'Noise margin - larger is better',
                'eye_width': 'Timing margin - wider allows more timing error',
                'crossing_points': 'Where transitions occur - jitter affects these'
            }
        }
    }


def generate_nondistorting_wave_demo(elements, bandwidth, samples_per_unit=100):
    """
    Demonstrate Nyquist's concept of nondistorting waves.
    A wave can be deformed but still be nondistorting if sampling is correct.
    """
    signal, _ = generate_rectangular_wave(elements, samples_per_unit)
    t = np.linspace(0, len(elements), len(signal))

    # Generate various filtered versions
    waves = []

    for bw in [bandwidth, bandwidth // 2, bandwidth * 2]:
        if bw < 3:
            bw = 3
        components = compute_fourier_series(signal, 50)
        filtered, _ = apply_channel_effects(components, bw, 'ideal')
        reconstructed = reconstruct_signal(filtered, len(signal))

        # Sample at center of each element
        samples = []
        decoded = []
        errors = 0
        for i in range(len(elements)):
            idx = int((i + 0.5) * samples_per_unit)
            if idx < len(reconstructed):
                val = reconstructed[idx]
                dec = 1 if val > 0.5 else 0
                samples.append({'index': idx, 'value': float(val), 'decoded': dec})
                decoded.append(dec)
                if dec != elements[i]:
                    errors += 1

        waves.append({
            'bandwidth': bw,
            'signal': reconstructed.tolist(),
            'samples': samples,
            'decoded': decoded,
            'errors': errors,
            'is_nondistorting': errors == 0
        })

    return {
        'original': signal.tolist(),
        'time': t.tolist(),
        'waves': waves,
        'explanation': {
            'main': '''From Nyquist: "The term nondistorting wave will be defined as a wave which
            produces perfect signals. A nondistorting wave may or may not be deformed."

            The criterion for a nondistorting wave depends on the receiving method. In this paper,
            when not otherwise stated, a wave is nondistorting when the value at the mid-instant
            of any time unit is proportional to the magnitude factor for the corresponding element.''',
            'criterion': '''Nyquist's Criterion (Appendix II-A): "A wave will be said to be
            nondistorting when the value at the mid-instant of any time unit is proportional to
            the magnitude factor for the corresponding element."''',
            'practical': '''With bandwidth = signaling speed (S Hz), all signals can be transmitted
            without errors at sampling instants. The Nyquist rate: S_max = 2B symbols/second for
            bandwidth B Hz. A channel with bandwidth equal to S/2 Hz is the MINIMUM required.'''
        }
    }


def generate_progressive_harmonics(elements, max_harmonics=20, samples_per_unit=50):
    """
    Show how signal quality improves as more harmonics are added.
    Demonstrates the relationship between bandwidth and signal fidelity.
    """
    signal, _ = generate_rectangular_wave(elements, samples_per_unit)
    components = compute_fourier_series(signal, max_harmonics)

    steps = []
    t = np.linspace(0, 1, len(signal), endpoint=False)

    for num_harm in [1, 3, 5, 7, 10, 15, max_harmonics]:
        if num_harm > max_harmonics:
            continue

        # Build signal with limited harmonics
        reconstructed = np.zeros(len(signal))
        for k in range(min(num_harm + 1, len(components))):
            comp = components[k]
            if k == 0:
                reconstructed += comp['a_k']
            else:
                reconstructed += comp['a_k'] * np.cos(2 * np.pi * k * t)
                reconstructed += comp['b_k'] * np.sin(2 * np.pi * k * t)

        # Calculate error metrics
        mse = np.mean((signal - reconstructed) ** 2)

        # Check decoding accuracy
        errors = 0
        for i in range(len(elements)):
            idx = int((i + 0.5) * samples_per_unit)
            if idx < len(reconstructed):
                decoded = 1 if reconstructed[idx] > 0.5 else 0
                if decoded != elements[i]:
                    errors += 1

        steps.append({
            'num_harmonics': num_harm,
            'signal': reconstructed.tolist(),
            'mse': float(mse),
            'errors': errors,
            'nyquist_ratio': num_harm / (len(elements) / 2),
            'bandwidth_sufficient': num_harm >= len(elements) / 2
        })

    return {
        'original': signal.tolist(),
        'steps': steps,
        'explanation': '''This visualization shows how adding more frequency components
        improves signal reconstruction. Nyquist proved that the minimum bandwidth needed
        equals half the signaling speed (S_max = 2B). Below this threshold, errors occur.'''
    }


# ============== Flask Routes ==============

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/morse_encode', methods=['POST'])
def morse_encode():
    """Convert text to Morse code and signal elements."""
    data = request.json
    text = data.get('text', 'SOS')

    morse = text_to_morse(text)
    elements, descriptions = morse_to_signal_elements(morse)

    return jsonify({
        'text': text,
        'morse': morse,
        'elements': elements,
        'descriptions': descriptions,
        'num_elements': len(elements),
        'explanation': {
            'dot': '1 time unit ON (represented as 1)',
            'dash': '3 time units ON (represented as 1,1,1)',
            'inter_element_gap': '1 time unit OFF between dots/dashes',
            'letter_gap': '3 time units OFF between letters',
            'word_gap': '7 time units OFF between words'
        }
    })


@app.route('/api/full_analysis', methods=['POST'])
def full_analysis():
    """Complete step-by-step signal analysis."""
    data = request.json
    text = data.get('text', 'SOS')
    bandwidth = data.get('bandwidth', 10)
    channel_type = data.get('channelType', 'ideal')
    samples_per_unit = data.get('samplesPerUnit', 50)
    num_harmonics = data.get('numHarmonics', 30)

    # Step 1: Text to Morse
    morse = text_to_morse(text)
    elements, descriptions = morse_to_signal_elements(morse)

    # Step 2: Generate rectangular wave
    signal, time_points = generate_rectangular_wave(elements, samples_per_unit)

    # Step 3: Fourier decomposition
    components = compute_fourier_series(signal, num_harmonics)

    # Step 4: Apply channel effects
    filtered_components, channel_effects = apply_channel_effects(
        components, bandwidth, channel_type
    )

    # Step 5: Reconstruct signal
    reconstructed = reconstruct_signal(filtered_components, len(signal))

    # Step 6: Sample and decode
    samples, decoded = sample_and_decode(
        reconstructed, len(elements), samples_per_unit
    )

    # Step 7: Calculate ISI
    isi_analysis = calculate_intersymbol_interference(
        reconstructed, len(elements), samples_per_unit
    )

    # Check for errors
    errors = sum(1 for i, e in enumerate(elements) if i < len(decoded) and e != decoded[i])

    return jsonify({
        'step1_encoding': {
            'text': text,
            'morse': morse,
            'elements': elements,
            'descriptions': descriptions
        },
        'step2_rectangular_wave': {
            'signal': signal.tolist(),
            'time': time_points.tolist(),
            'num_samples': len(signal)
        },
        'step3_fourier': {
            'components': components,
            'num_components': len(components)
        },
        'step4_channel': {
            'filtered_components': filtered_components,
            'effects': channel_effects,
            'bandwidth': bandwidth,
            'channel_type': channel_type
        },
        'step5_reconstruction': {
            'signal': reconstructed.tolist()
        },
        'step6_sampling': {
            'samples': samples,
            'decoded': decoded
        },
        'step7_analysis': {
            'isi': isi_analysis,
            'errors': errors,
            'error_rate': errors / len(elements) if elements else 0,
            'success': errors == 0
        },
        'nyquist_info': {
            'signaling_speed': len(elements) / (len(elements) * 1.0),
            'min_bandwidth': len(elements) / (len(elements) * 2.0),
            'used_bandwidth': bandwidth,
            'formula': 'B_min = S/2 where S is signaling speed (symbols/sec)'
        }
    })


@app.route('/api/progressive_build', methods=['POST'])
def progressive_build():
    """Build signal progressively by adding harmonics one at a time."""
    data = request.json
    elements = data.get('elements', [1, 0, 1, 0])
    max_harmonic = data.get('maxHarmonic', 10)
    samples_per_unit = data.get('samplesPerUnit', 50)

    signal, _ = generate_rectangular_wave(elements, samples_per_unit)
    components = compute_fourier_series(signal, max_harmonic)
    progressive = build_signal_progressively(components, len(signal), max_harmonic)

    return jsonify({
        'original': signal.tolist(),
        'progressive': progressive,
        'components': components
    })


@app.route('/api/nyquist_theorem', methods=['GET'])
def nyquist_theorem():
    """Explain Nyquist's theorem with examples."""
    return jsonify({
        'theorem': 'The maximum signaling speed is 2B symbols/second, where B is the bandwidth in Hz',
        'formula': 'S_max = 2B',
        'inverse': 'Minimum bandwidth needed: B_min = S/2',
        'examples': [
            {'signaling_speed': 10, 'min_bandwidth': 5, 'explanation': 'At 10 symbols/sec, need at least 5 Hz'},
            {'signaling_speed': 100, 'min_bandwidth': 50, 'explanation': 'At 100 symbols/sec, need at least 50 Hz'},
            {'signaling_speed': 1000, 'min_bandwidth': 500, 'explanation': 'At 1000 symbols/sec, need at least 500 Hz'}
        ],
        'key_insight': 'Any signal can be transmitted through a limited-bandwidth channel if properly shaped'
    })


@app.route('/api/shape_discrimination', methods=['POST'])
def shape_discrimination():
    """Demonstrate shape factor vs discrimination factor separation."""
    data = request.json
    text = data.get('text', 'SOS')
    num_harmonics = data.get('numHarmonics', 30)

    morse = text_to_morse(text)
    elements, _ = morse_to_signal_elements(morse)
    signal, _ = generate_rectangular_wave(elements, 50)

    shape_factors = compute_shape_factor(signal, len(signal), num_harmonics)
    discrimination_factors = compute_discrimination_factor(elements, num_harmonics)

    return jsonify({
        'shape_factors': shape_factors,
        'discrimination_factors': discrimination_factors,
        'explanation': {
            'shape_factor': '''The Shape Factor F(ω) depends only on the wave form (rectangular, sinusoidal, etc.)
            and is INDEPENDENT of the intelligence being transmitted. It is determined by the physical
            characteristics of the signaling method.''',
            'discrimination_factor': '''The Discrimination Factor D(ω) depends only on the particular
            signal (the sequence of magnitude factors m_k) and is INDEPENDENT of the wave shape used.
            It carries the actual information content.''',
            'key_insight': '''Nyquist's brilliant separation: Total signal = F(ω) × D(ω). This means
            we can analyze transmission characteristics independently of the message content.'''
        }
    })


@app.route('/api/band_redundancy', methods=['POST'])
def band_redundancy():
    """Demonstrate band redundancy concept."""
    data = request.json
    text = data.get('text', 'SOS')
    num_harmonics = data.get('numHarmonics', 50)

    morse = text_to_morse(text)
    elements, _ = morse_to_signal_elements(morse)

    result = demonstrate_band_redundancy(elements, num_harmonics)
    return jsonify(result)


@app.route('/api/ideal_shape_factors', methods=['GET'])
def ideal_shape_factors():
    """Get ideal shape factors that produce nondistorting waves."""
    result = generate_ideal_shape_factors()
    return jsonify(result)


@app.route('/api/carrier_modulation', methods=['POST'])
def carrier_modulation():
    """Demonstrate carrier wave modulation and sidebands."""
    data = request.json
    text = data.get('text', 'SOS')
    carrier_freq = data.get('carrierFreq', 10)

    morse = text_to_morse(text)
    elements, _ = morse_to_signal_elements(morse)

    result = generate_carrier_modulation(elements, carrier_freq)
    return jsonify(result)


@app.route('/api/single_sideband', methods=['POST'])
def single_sideband():
    """Demonstrate single sideband vs double sideband transmission."""
    data = request.json
    text = data.get('text', 'SOS')
    carrier_freq = data.get('carrierFreq', 10)

    morse = text_to_morse(text)
    elements, _ = morse_to_signal_elements(morse)

    try:
        result = generate_single_sideband(elements, carrier_freq)
        return jsonify(result)
    except ImportError:
        return jsonify({
            'error': 'scipy not available',
            'explanation': {
                'main': '''Single Sideband (SSB) transmission requires only one sideband,
                reducing bandwidth to match DC telegraphy.'''
            }
        })


@app.route('/api/eye_diagram', methods=['POST'])
def eye_diagram():
    """Generate eye diagram for ISI visualization."""
    data = request.json
    text = data.get('text', 'SOS')
    bandwidth = data.get('bandwidth', 10)
    channel_type = data.get('channelType', 'ideal')

    morse = text_to_morse(text)
    elements, _ = morse_to_signal_elements(morse)

    result = generate_eye_diagram(elements, bandwidth, channel_type)
    return jsonify(result)


@app.route('/api/nondistorting_wave', methods=['POST'])
def nondistorting_wave():
    """Demonstrate nondistorting wave concept."""
    data = request.json
    text = data.get('text', 'SOS')
    bandwidth = data.get('bandwidth', 10)

    morse = text_to_morse(text)
    elements, _ = morse_to_signal_elements(morse)

    result = generate_nondistorting_wave_demo(elements, bandwidth)
    return jsonify(result)


@app.route('/api/progressive_harmonics', methods=['POST'])
def progressive_harmonics():
    """Show signal building with progressive harmonics."""
    data = request.json
    text = data.get('text', 'SOS')
    max_harmonics = data.get('maxHarmonics', 20)

    morse = text_to_morse(text)
    elements, _ = morse_to_signal_elements(morse)

    result = generate_progressive_harmonics(elements, max_harmonics)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5002)
