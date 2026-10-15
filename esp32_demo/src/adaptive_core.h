/**
 * @file adaptive_core.h
 * @brief Pure C++ implementation of Candidate Selector v1.1 for ESP32 / Wokwi.
 * 
 * Implements the frozen Protocol v1.1.0 parameters and decision hierarchy:
 * 1. Fixed-Low time-domain features + Zero-Crossing Frequency (ZCF)
 * 2. Decision Hierarchy:
 *    - Step 1: Confident Anomaly check (low_score >= p_anomaly -> stop at Stage 1, label 1)
 *    - Step 2: Clean ZCF bounds check (zcf < z_min || zcf > z_max -> escalate to Fixed-High)
 *    - Step 3: Confident Normal check (low_score <= p_normal -> stop at Stage 1, label 0)
 *    - Step 4: Uncertain Score check -> escalate to Fixed-High
 * 3. Fixed-High FFT stage (16-band normalized energy + FFT peak frequency)
 * 
 * Free of Arduino/hardware dependencies; fully verifiable with standard C++11.
 */

#ifndef ADAPTIVE_CORE_H
#define ADAPTIVE_CORE_H

#include <cmath>
#include <cstdint>
#include <algorithm>

namespace AdaptiveMonitoring {

// ============================================================================
// Frozen Protocol v1.1.0 Constants (from results/evaluation_protocol_v1.1.json)
// ============================================================================

constexpr int SAMPLE_RATE_HZ = 4000;
constexpr int SAMPLES_PER_WINDOW = 1024;

// Stage operations proxy accounting
constexpr int STAGE1_OPS = 10265;            // 10 * N + 25
constexpr int STAGE2_ADDED_OPS = 29250;      // 2.5 * N * log2(N) + 3.5 * N + 66
constexpr int STAGE2_CUMULATIVE_OPS = 39515; // STAGE1_OPS + STAGE2_ADDED_OPS

// Fixed-Low Reference Statistics (from model-fit normal partition)
constexpr float LOW_REF_MEDIAN[3] = {
    -0.42214824f,  // log(RMS)
     0.62723182f,  // log(crest factor)
     0.48272222f   // log(kurtosis)
};

constexpr float LOW_REF_SCALE[3] = {
    0.13246204f,   // 1.4826 * MAD of log(RMS)
    0.32208571f,   // 1.4826 * MAD of log(crest factor)
    0.11346820f    // 1.4826 * MAD of log(kurtosis)
};

constexpr float LOW_THRESHOLD = 5.27421223f;

// Candidate Selector v1.1 Frozen Decision Boundaries (from dev calibration partition)
constexpr float P_ANOMALY = 7.14050243f;  // 1.2 * max(calib normal low score)
constexpr float P_NORMAL  = 1.77141950f;  // 60th percentile of calib normal low score
constexpr float ZCF_MIN   = 173.593750f;  // Clean normal min ZCF - 10 Hz margin
constexpr float ZCF_MAX   = 263.906250f;  // Clean normal max ZCF + 10 Hz margin

// Fixed-High Reference Statistics (16 bands, 0 to 1200 Hz)
constexpr float HIGH_REF_MEDIAN[16] = {
    0.0016098757f, 0.0011130731f, 0.6476407802f, 0.0176142900f,
    0.0012432271f, 0.0009764492f, 0.0012827108f, 0.0013166743f,
    0.0013932264f, 0.0013515235f, 0.0015330062f, 0.0014491711f,
    0.0014370000f, 0.0012130750f, 0.0011857169f, 0.0010682192f
};

constexpr float HIGH_REF_SCALE[16] = {
    0.0022806419f, 0.0016194550f, 0.5214032120f, 0.0260563367f,
    0.0018205107f, 0.0014019044f, 0.0018727651f, 0.0019152536f,
    0.0020313590f, 0.0019479872f, 0.0022292287f, 0.0021018578f,
    0.0020844645f, 0.0017603556f, 0.0017169219f, 0.0015484890f
};

constexpr float HIGH_THRESHOLD = 55.30946553f;

// ============================================================================
// Result Structure
// ============================================================================

struct AdaptiveOutput {
    int predicted_label;          // 0: normal, 1: anomaly
    float low_stage_score;        // Stage 1 robust z-score distance
    float zcf_hz;                 // Zero-crossing frequency
    float high_stage_score;       // Stage 2 robust z-score distance (-1.0 if not run)
    float frequency_estimate_hz;  // Final frequency estimate (ZCF or FFT peak)
    const char* stage_reached;    // "stage1_low" or "stage2_high"
    bool fft_used;                // True if escalated to Fixed-High
    const char* escalation_reason;// "confident_anomaly", "zcf_out_of_range", "confident_normal", "uncertain_score"
    int op_count_proxy;           // 10265 or 39515
};

// ============================================================================
// Feature Extraction Functions
// ============================================================================

inline float compute_rms(const float* x, int n) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    return std::sqrt(sum_sq / static_cast<float>(n));
}

inline float compute_crest_factor(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
    }
    float mean = sum / static_cast<float>(n);

    float sum_sq = 0.0f;
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - mean;
        sum_sq += diff * diff;
        float abs_val = std::abs(diff);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    float sd = std::sqrt(sum_sq / static_cast<float>(n));
    if (sd <= 1e-12f) {
        return 0.0f;
    }
    return max_abs / sd;
}

inline float compute_kurtosis(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
    }
    float mean = sum / static_cast<float>(n);

    float sum_sq = 0.0f;
    float sum_quad = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - mean;
        float diff_sq = diff * diff;
        sum_sq += diff_sq;
        sum_quad += diff_sq * diff_sq;
    }
    float var = sum_sq / static_cast<float>(n);
    if (var <= 1e-24f) {
        return 0.0f;
    }
    float m4 = sum_quad / static_cast<float>(n);
    return m4 / (var * var);
}

inline float compute_zero_crossing_frequency(const float* x, int n, int sample_rate_hz) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
    }
    float mean = sum / static_cast<float>(n);

    int crossings = 0;
    bool prev_sign = (x[0] - mean) >= 0.0f;
    for (int i = 1; i < n; ++i) {
        bool curr_sign = (x[i] - mean) >= 0.0f;
        if (curr_sign != prev_sign) {
            crossings++;
            prev_sign = curr_sign;
        }
    }
    float duration_s = static_cast<float>(n) / static_cast<float>(sample_rate_hz);
    return static_cast<float>(crossings) / (2.0f * duration_s);
}

inline float compute_fixed_low_score(const float* x, int n) {
    float r = compute_rms(x, n);
    float cf = compute_crest_factor(x, n);
    float k = compute_kurtosis(x, n);

    float log_r = std::log(std::max(r, 1e-8f));
    float log_cf = std::log(std::max(cf, 1e-8f));
    float log_k = std::log(std::max(k, 1e-8f));

    float z0 = (log_r - LOW_REF_MEDIAN[0]) / LOW_REF_SCALE[0];
    float z1 = (log_cf - LOW_REF_MEDIAN[1]) / LOW_REF_SCALE[1];
    float z2 = (log_k - LOW_REF_MEDIAN[2]) / LOW_REF_SCALE[2];

    return std::sqrt(z0 * z0 + z1 * z1 + z2 * z2);
}

// ============================================================================
// Fixed-High FFT & Spectral Energy Functions
// ============================================================================

/**
 * In-place Radix-2 Cooley-Tukey FFT for power-of-2 length (N = 1024).
 */
inline void fft_radix2(float* real, float* imag, int n) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
        int k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    // Butterfly stages
    // Avoid the global PI macro defined by Arduino.h in the firmware build.
    const float PI_RAD = 3.14159265358979323846f;
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * PI_RAD / static_cast<float>(len);
        float wlen_r = std::cos(angle);
        float wlen_i = std::sin(angle);
        int half = len >> 1;

        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (int k = 0; k < half; ++k) {
                float u_r = real[i + k];
                float u_i = imag[i + k];
                float v_r = real[i + k + half] * w_r - imag[i + k + half] * w_i;
                float v_i = real[i + k + half] * w_i + imag[i + k + half] * w_r;

                real[i + k] = u_r + v_r;
                imag[i + k] = u_i + v_i;
                real[i + k + half] = u_r - v_r;
                imag[i + k + half] = u_i - v_i;

                float next_w_r = w_r * wlen_r - w_i * wlen_i;
                float next_w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }
}

/**
 * Computes 16-band normalized spectral energies and FFT peak frequency.
 */
inline void compute_fixed_high_features(
    const float* x,
    int n,
    int sample_rate_hz,
    float* band_energies,
    float* peak_freq_hz
) {
    // Avoid the global PI macro defined by Arduino.h in the firmware build.
    const float PI_RAD = 3.14159265358979323846f;

    // 1. Mean-center and Hann window
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
    }
    float mean = sum / static_cast<float>(n);

    // Static buffers to avoid dynamic allocation on ESP32 stack
    static float real[SAMPLES_PER_WINDOW];
    static float imag[SAMPLES_PER_WINDOW];

    for (int i = 0; i < n; ++i) {
        // Hann window: 0.5 * (1 - cos(2*pi*i / (n - 1)))
        float w = 0.5f - 0.5f * std::cos(2.0f * PI_RAD * static_cast<float>(i) / static_cast<float>(n - 1));
        real[i] = (x[i] - mean) * w;
        imag[i] = 0.0f;
    }

    // 2. Perform in-place FFT
    fft_radix2(real, imag, n);

    // 3. Power spectrum for non-negative frequencies (bins 0 to n/2)
    int n_bins = n / 2 + 1; // 513 bins for n=1024
    float df = static_cast<float>(sample_rate_hz) / static_cast<float>(n); // 3.90625 Hz

    // Find peak frequency among non-DC bins (k = 1 to n/2)
    float max_power = -1.0f;
    int peak_bin = 1;

    for (int k = 1; k < n_bins; ++k) {
        float p = real[k] * real[k] + imag[k] * imag[k];
        if (p > max_power) {
            max_power = p;
            peak_bin = k;
        }
    }
    *peak_freq_hz = static_cast<float>(peak_bin) * df;

    // 4. Accumulate 16 uniform bands across 0 to 1200 Hz (75 Hz bandwidth each)
    constexpr int NUM_BANDS = 16;
    constexpr float BAND_WIDTH = 75.0f;

    for (int b = 0; b < NUM_BANDS; ++b) {
        band_energies[b] = 0.0f;
    }

    for (int k = 0; k < n_bins; ++k) {
        float f_k = static_cast<float>(k) * df;
        float power = real[k] * real[k] + imag[k] * imag[k];

        for (int b = 0; b < NUM_BANDS; ++b) {
            float f_low = static_cast<float>(b) * BAND_WIDTH;
            float f_high = static_cast<float>(b + 1) * BAND_WIDTH;

            if (b == NUM_BANDS - 1) {
                if (f_k >= f_low && f_k <= f_high) {
                    band_energies[b] += power;
                }
            } else {
                if (f_k >= f_low && f_k < f_high) {
                    band_energies[b] += power;
                }
            }
        }
    }

    // 5. Normalize band energies to sum to 1.0
    float total_energy = 0.0f;
    for (int b = 0; b < NUM_BANDS; ++b) {
        total_energy += band_energies[b];
    }

    if (total_energy > 1e-12f) {
        for (int b = 0; b < NUM_BANDS; ++b) {
            band_energies[b] /= total_energy;
        }
    } else {
        for (int b = 0; b < NUM_BANDS; ++b) {
            band_energies[b] = 0.0f;
        }
    }
}

inline float compute_fixed_high_score(const float* band_energies) {
    float sum_sq = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float z = (band_energies[i] - HIGH_REF_MEDIAN[i]) / HIGH_REF_SCALE[i];
        sum_sq += z * z;
    }
    return std::sqrt(sum_sq);
}

// ============================================================================
// Adaptive Decision Core (Protocol v1.1.0 Decision Hierarchy)
// ============================================================================

inline AdaptiveOutput process_window(const float* x, int n = SAMPLES_PER_WINDOW) {
    AdaptiveOutput out;

    // Stage 1: Fixed-Low features and Zero-Crossing Frequency
    out.low_stage_score = compute_fixed_low_score(x, n);
    out.zcf_hz = compute_zero_crossing_frequency(x, n, SAMPLE_RATE_HZ);
    out.high_stage_score = -1.0f;

    // ------------------------------------------------------------------------
    // Step 1: Confident Anomaly Check (decided cheaply at Stage 1)
    // ------------------------------------------------------------------------
    if (out.low_stage_score >= P_ANOMALY) {
        out.predicted_label = 1;
        out.frequency_estimate_hz = out.zcf_hz;
        out.stage_reached = "stage1_low";
        out.fft_used = false;
        out.escalation_reason = "confident_anomaly";
        out.op_count_proxy = STAGE1_OPS;
        return out;
    }

    // ------------------------------------------------------------------------
    // Step 2: Clean ZCF Bounds Check (frequency discrepancy / noise smearing)
    // ------------------------------------------------------------------------
    if (out.zcf_hz < ZCF_MIN || out.zcf_hz > ZCF_MAX) {
        float band_energies[16];
        float peak_freq = 0.0f;
        compute_fixed_high_features(x, n, SAMPLE_RATE_HZ, band_energies, &peak_freq);

        out.high_stage_score = compute_fixed_high_score(band_energies);
        out.predicted_label = (out.high_stage_score > HIGH_THRESHOLD) ? 1 : 0;
        out.frequency_estimate_hz = peak_freq;
        out.stage_reached = "stage2_high";
        out.fft_used = true;
        out.escalation_reason = "zcf_out_of_range";
        out.op_count_proxy = STAGE2_CUMULATIVE_OPS;
        return out;
    }

    // ------------------------------------------------------------------------
    // Step 3: Confident Normal Check (decided cheaply at Stage 1)
    // ------------------------------------------------------------------------
    if (out.low_stage_score <= P_NORMAL) {
        out.predicted_label = 0;
        out.frequency_estimate_hz = out.zcf_hz;
        out.stage_reached = "stage1_low";
        out.fft_used = false;
        out.escalation_reason = "confident_normal";
        out.op_count_proxy = STAGE1_OPS;
        return out;
    }

    // ------------------------------------------------------------------------
    // Step 4: Uncertain Score Check (between P_NORMAL and P_ANOMALY)
    // ------------------------------------------------------------------------
    {
        float band_energies[16];
        float peak_freq = 0.0f;
        compute_fixed_high_features(x, n, SAMPLE_RATE_HZ, band_energies, &peak_freq);

        out.high_stage_score = compute_fixed_high_score(band_energies);
        out.predicted_label = (out.high_stage_score > HIGH_THRESHOLD) ? 1 : 0;
        out.frequency_estimate_hz = peak_freq;
        out.stage_reached = "stage2_high";
        out.fft_used = true;
        out.escalation_reason = "uncertain_score";
        out.op_count_proxy = STAGE2_CUMULATIVE_OPS;
        return out;
    }
}

} // namespace AdaptiveMonitoring

#endif // ADAPTIVE_CORE_H
