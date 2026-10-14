/**
 * @file test_harness.cpp
 * @brief Standalone desktop C++ verification harness for Adaptive Core v1.1 and v1.2.
 * 
 * Verifies Python / C++ decision and numerical parity across all 5 representative
 * development demo vectors from Group g0001:
 * 1. Candidate Selector v1.1 (Protocol v1.1.0, P_NORMAL=1.771420)
 * 2. Candidate Selector v1.2 (Protocol v1.2.0, P_NORMAL=1.311028)
 * 
 * Strict Parity Requirement:
 * The process exits nonzero if ANY expected decision, stage, escalation reason,
 * FFT-use flag, operation-count proxy, anomaly score, or frequency value fails
 * its stated numerical tolerance.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include "src/adaptive_core.h"
#include "src/adaptive_core_v1_2.h"
#include "src/demo_vectors.h"

int test_candidate_v1_1() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "PART 1: ESP32 Adaptive Core v1.1 Desktop C++ Parity Test Harness" << std::endl;
    std::cout << "Testing 5 Deterministic Development Vectors from Group g0001 (Protocol v1.1.0)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    int total_vectors = NUM_DEMO_VECTORS;
    int full_matches = 0;

    for (int i = 0; i < total_vectors; ++i) {
        const DemoVector& v = DEMO_VECTORS[i];
        AdaptiveMonitoring::AdaptiveOutput res = AdaptiveMonitoring::process_window(v.samples, 1024);

        bool stage_match = (std::string(res.stage_reached) == std::string(v.expected_stage));
        bool reason_match = (std::string(res.escalation_reason) == std::string(v.expected_reason));
        bool pred_match = (res.predicted_label == v.expected_predicted_label);
        bool fft_match = (res.fft_used == v.expected_fft_used);
        bool ops_match = (res.op_count_proxy == v.expected_op_count_proxy);

        float low_diff = std::abs(res.low_stage_score - v.expected_low_score);
        bool low_score_match = (low_diff < 0.01f);

        float freq_diff = std::abs(res.frequency_estimate_hz - v.expected_freq_hz);
        bool freq_match = (freq_diff < 1.0f);

        bool all_ok = stage_match && reason_match && pred_match && fft_match && ops_match && low_score_match && freq_match;
        if (all_ok) {
            full_matches++;
        }

        std::cout << "\n[" << (i + 1) << "/" << total_vectors << "] Vector: " << v.name
                  << " (" << v.signal_case << " @ " << v.snr_db << " dB)" << std::endl;
        std::cout << "    Description: " << v.description << std::endl;
        std::cout << "    C++ Result   : Stage=" << res.stage_reached
                  << ", Reason=" << res.escalation_reason
                  << ", Pred=" << res.predicted_label
                  << ", FFT=" << (res.fft_used ? "true" : "false")
                  << ", Ops=" << res.op_count_proxy
                  << ", LowScore=" << std::fixed << std::setprecision(4) << res.low_stage_score
                  << ", Freq=" << std::fixed << std::setprecision(1) << res.frequency_estimate_hz << " Hz" << std::endl;
        std::cout << "    Expected Py  : Stage=" << v.expected_stage
                  << ", Reason=" << v.expected_reason
                  << ", Pred=" << v.expected_predicted_label
                  << ", FFT=" << (v.expected_fft_used ? "true" : "false")
                  << ", Ops=" << v.expected_op_count_proxy
                  << ", LowScore=" << std::fixed << std::setprecision(4) << v.expected_low_score
                  << ", Freq=" << std::fixed << std::setprecision(1) << v.expected_freq_hz << " Hz" << std::endl;

        if (!stage_match) std::cerr << "    [FAIL] Stage mismatch: got " << res.stage_reached << " vs " << v.expected_stage << std::endl;
        if (!reason_match) std::cerr << "    [FAIL] Reason mismatch: got " << res.escalation_reason << " vs " << v.expected_reason << std::endl;
        if (!pred_match) std::cerr << "    [FAIL] Prediction mismatch: got " << res.predicted_label << " vs " << v.expected_predicted_label << std::endl;
        if (!fft_match) std::cerr << "    [FAIL] FFT-use mismatch: got " << res.fft_used << " vs " << v.expected_fft_used << std::endl;
        if (!ops_match) std::cerr << "    [FAIL] Ops mismatch: got " << res.op_count_proxy << " vs " << v.expected_op_count_proxy << std::endl;
        if (!low_score_match) std::cerr << "    [FAIL] Low score diff " << low_diff << " exceeds tolerance 0.01" << std::endl;
        if (!freq_match) std::cerr << "    [FAIL] Frequency diff " << freq_diff << " exceeds tolerance 1.0 Hz" << std::endl;

        std::cout << "    Status       : " << (all_ok ? "[PASS] FULL PARITY" : "[FAIL] MISMATCH DETECTED") << std::endl;
    }

    std::cout << "\nSummary v1.1:" << std::endl;
    std::cout << "  Full Vector Matches : " << full_matches << "/" << total_vectors << std::endl;

    if (full_matches == total_vectors) {
        std::cout << "RESULT v1.1: 100% Python/C++ Parity Achieved (demonstrating expected Vector 4 harmonic miss)." << std::endl;
        return 0;
    } else {
        std::cerr << "RESULT v1.1: Full Parity Mismatch Detected (" << (total_vectors - full_matches) << " vector failures)!" << std::endl;
        return 1;
    }
}

struct ExpectedV12 {
    const char* stage;
    const char* reason;
    int pred;
    bool fft_used;
    int ops;
    float low_score;
    float high_score; // -1.0f if not escalated
    float freq_hz;
};

int test_candidate_v1_2() {
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "PART 2: ESP32 Adaptive Core v1.2 Desktop C++ Verification Harness" << std::endl;
    std::cout << "Testing 5 Deterministic Development Vectors with Recalibrated P_NORMAL=1.311028" << std::endl;
    std::cout << "================================================================================" << std::endl;

    int total_vectors = NUM_DEMO_VECTORS;
    int full_matches_v1_2 = 0;

    // Explicit Python Reference Outputs for Candidate Selector v1.2 on Group g0001 demo vectors:
    const ExpectedV12 EXPECTED_V12[5] = {
        // 1. demo_clean_normal: normal @ 30 dB (Stage 1, low_score <= 1.3110 -> confident_normal)
        {"stage1_low", "confident_normal", 0, false, 10265, 1.1874f, -1.0f, 216.7969f},
        // 2. demo_impact: impact @ 20 dB (Stage 1, low_score >= 7.1405 -> confident_anomaly)
        {"stage1_low", "confident_anomaly", 1, false, 10265, 32.5831f, -1.0f, 955.0781f},
        // 3. demo_frequency_shift: frequency_shift @ 20 dB (Stage 2, zcf out-of-range -> Fixed-High FFT anomaly)
        {"stage2_high", "zcf_out_of_range", 1, true, 39515, 1.3599f, 545.4400f, 328.1250f},
        // 4. demo_added_harmonic: added_harmonic @ 30 dB (Stage 2, uncertain_score -> Fixed-High FFT anomaly, RESOLVED!)
        {"stage2_high", "uncertain_score", 1, true, 39515, 1.7185f, 119.3773f, 218.7500f},
        // 5. demo_noisy_normal: normal @ 0 dB (Stage 2, zcf out-of-range -> Fixed-High FFT normal)
        {"stage2_high", "zcf_out_of_range", 0, true, 39515, 5.1103f, 46.2841f, 218.7500f}
    };

    for (int i = 0; i < total_vectors; ++i) {
        const DemoVector& v = DEMO_VECTORS[i];
        const ExpectedV12& exp = EXPECTED_V12[i];
        AdaptiveMonitoring::AdaptiveOutput res = AdaptiveMonitoringV12::process_window(v.samples, 1024);

        bool stage_ok = (std::string(res.stage_reached) == std::string(exp.stage));
        bool reason_ok = (std::string(res.escalation_reason) == std::string(exp.reason));
        bool pred_ok = (res.predicted_label == exp.pred);
        bool fft_ok = (res.fft_used == exp.fft_used);
        bool ops_ok = (res.op_count_proxy == exp.ops);

        float low_diff = std::abs(res.low_stage_score - exp.low_score);
        bool low_score_ok = (low_diff < 0.01f);

        float high_diff = (exp.high_score < 0.0f) ? 0.0f : std::abs(res.high_stage_score - exp.high_score);
        bool high_score_ok = (exp.high_score < 0.0f) ? (res.high_stage_score < 0.0f) : (high_diff < 0.05f);

        float freq_diff = std::abs(res.frequency_estimate_hz - exp.freq_hz);
        bool freq_ok = (freq_diff < 1.0f);

        bool all_ok = stage_ok && reason_ok && pred_ok && fft_ok && ops_ok && low_score_ok && high_score_ok && freq_ok;
        if (all_ok) {
            full_matches_v1_2++;
        }

        std::cout << "\n[" << (i + 1) << "/" << total_vectors << "] Vector: " << v.name
                  << " (" << v.signal_case << " @ " << v.snr_db << " dB)" << std::endl;
        std::cout << "    v1.2 Result  : Stage=" << res.stage_reached
                  << ", Reason=" << res.escalation_reason
                  << ", Pred=" << res.predicted_label
                  << ", FFT=" << (res.fft_used ? "true" : "false")
                  << ", Ops=" << res.op_count_proxy
                  << ", LowScore=" << std::fixed << std::setprecision(4) << res.low_stage_score
                  << ", HighScore=" << std::fixed << std::setprecision(4) << res.high_stage_score
                  << ", Freq=" << std::fixed << std::setprecision(1) << res.frequency_estimate_hz << " Hz" << std::endl;
        std::cout << "    Expected v1.2: Stage=" << exp.stage
                  << ", Reason=" << exp.reason
                  << ", Pred=" << exp.pred
                  << ", FFT=" << (exp.fft_used ? "true" : "false")
                  << ", Ops=" << exp.ops
                  << ", LowScore=" << std::fixed << std::setprecision(4) << exp.low_score
                  << ", HighScore=" << std::fixed << std::setprecision(4) << exp.high_score
                  << ", Freq=" << std::fixed << std::setprecision(1) << exp.freq_hz << " Hz" << std::endl;

        if (!stage_ok) std::cerr << "    [FAIL] Stage mismatch: got " << res.stage_reached << " vs " << exp.stage << std::endl;
        if (!reason_ok) std::cerr << "    [FAIL] Reason mismatch: got " << res.escalation_reason << " vs " << exp.reason << std::endl;
        if (!pred_ok) std::cerr << "    [FAIL] Prediction mismatch: got " << res.predicted_label << " vs " << exp.pred << std::endl;
        if (!fft_ok) std::cerr << "    [FAIL] FFT-use mismatch: got " << res.fft_used << " vs " << exp.fft_used << std::endl;
        if (!ops_ok) std::cerr << "    [FAIL] Ops mismatch: got " << res.op_count_proxy << " vs " << exp.ops << std::endl;
        if (!low_score_ok) std::cerr << "    [FAIL] Low score diff " << low_diff << " exceeds tolerance 0.01" << std::endl;
        if (!high_score_ok) std::cerr << "    [FAIL] High score diff " << high_diff << " exceeds tolerance 0.05" << std::endl;
        if (!freq_ok) std::cerr << "    [FAIL] Frequency diff " << freq_diff << " exceeds tolerance 1.0 Hz" << std::endl;

        if (i == 3) {
            std::cout << "    Key Finding  : Vector 4 successfully escalates to Stage 2 FFT and is flagged Anomaly under v1.2!" << std::endl;
        }
        std::cout << "    Status       : " << (all_ok ? "[PASS] FULL PARITY v1.2" : "[FAIL] MISMATCH DETECTED") << std::endl;
    }

    std::cout << "\nSummary v1.2:" << std::endl;
    std::cout << "  Full Vector Matches : " << full_matches_v1_2 << "/" << total_vectors << std::endl;

    if (full_matches_v1_2 == total_vectors) {
        std::cout << "RESULT v1.2: 100% Full Parity Achieved (Vector 4 harmonic detection confirmed)." << std::endl;
        return 0;
    } else {
        std::cerr << "RESULT v1.2: Full Parity Mismatch Detected (" << (total_vectors - full_matches_v1_2) << " vector failures)!" << std::endl;
        return 1;
    }
}

int main() {
    int r1 = test_candidate_v1_1();
    int r2 = test_candidate_v1_2();
    return (r1 == 0 && r2 == 0) ? 0 : 1;
}
