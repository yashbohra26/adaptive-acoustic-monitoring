/**
 * @file adaptive_core_v1_2.h
 * @brief Pure C++ implementation of Candidate Selector v1.2 for ESP32 / Wokwi.
 * 
 * Implements the Protocol v1.2.0 parameters and decision hierarchy:
 * - Recalibrates P_NORMAL from 1.771420 (60th percentile, v1.1) down to 1.311028
 *   (50th percentile, v1.2), closing the added harmonic false-negative gap.
 * - Retains all Stage 1 and Stage 2 feature extraction and distance metrics.
 * 
 * Free of Arduino/hardware dependencies; fully verifiable with standard C++11.
 */

#ifndef ADAPTIVE_CORE_V1_2_H
#define ADAPTIVE_CORE_V1_2_H

#include "adaptive_core.h"

namespace AdaptiveMonitoringV12 {

// Inherit common types and functions from AdaptiveMonitoring
using AdaptiveMonitoring::AdaptiveOutput;
using AdaptiveMonitoring::SAMPLE_RATE_HZ;
using AdaptiveMonitoring::SAMPLES_PER_WINDOW;
using AdaptiveMonitoring::STAGE1_OPS;
using AdaptiveMonitoring::STAGE2_ADDED_OPS;
using AdaptiveMonitoring::STAGE2_CUMULATIVE_OPS;
using AdaptiveMonitoring::P_ANOMALY;
using AdaptiveMonitoring::ZCF_MIN;
using AdaptiveMonitoring::ZCF_MAX;
using AdaptiveMonitoring::HIGH_THRESHOLD;
using AdaptiveMonitoring::compute_fixed_low_score;
using AdaptiveMonitoring::compute_zero_crossing_frequency;
using AdaptiveMonitoring::compute_fixed_high_features;
using AdaptiveMonitoring::compute_fixed_high_score;

// ============================================================================
// Recalibrated Protocol v1.2.0 Constants (results/evaluation_protocol_v1.2.json)
// ============================================================================
constexpr float P_NORMAL = 1.31102762f;  // 50th percentile of normal calibration signals

/**
 * @brief Process an audio window using Candidate Selector v1.2 decision hierarchy.
 * 
 * @param x Input audio samples in float32.
 * @param n Number of samples in the window (default 1024).
 * @return AdaptiveOutput Decision, stage reached, escalation reason, and ops proxy.
 */
inline AdaptiveOutput process_window(const float* x, int n = SAMPLES_PER_WINDOW) {
    AdaptiveOutput out;

    // Stage 1: Fixed-Low features and Zero-Crossing Frequency
    out.low_stage_score = compute_fixed_low_score(x, n);
    out.zcf_hz = compute_zero_crossing_frequency(x, n, SAMPLE_RATE_HZ);
    out.high_stage_score = -1.0f;

    // Step 1: Confident Anomaly Check (decided cheaply at Stage 1)
    if (out.low_stage_score >= P_ANOMALY) {
        out.predicted_label = 1;
        out.frequency_estimate_hz = out.zcf_hz;
        out.stage_reached = "stage1_low";
        out.fft_used = false;
        out.escalation_reason = "confident_anomaly";
        out.op_count_proxy = STAGE1_OPS;
        return out;
    }

    // Step 2: Clean ZCF Bounds Check (frequency discrepancy / noise smearing)
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

    // Step 3: Confident Normal Check with v1.2 Recalibrated P_NORMAL (50th percentile)
    if (out.low_stage_score <= P_NORMAL) {
        out.predicted_label = 0;
        out.frequency_estimate_hz = out.zcf_hz;
        out.stage_reached = "stage1_low";
        out.fft_used = false;
        out.escalation_reason = "confident_normal";
        out.op_count_proxy = STAGE1_OPS;
        return out;
    }

    // Step 4: Uncertain Score Check (between P_NORMAL and P_ANOMALY)
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

} // namespace AdaptiveMonitoringV12

#endif // ADAPTIVE_CORE_V1_2_H
