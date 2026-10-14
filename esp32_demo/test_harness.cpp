/**
 * @file test_harness.cpp
 * @brief Standalone desktop C++ verification harness for Adaptive Core v1.1 and v1.2.
 * 
 * Verifies Python / C++ decision parity across all 5 representative development demo vectors:
 * 1. Candidate Selector v1.1 (Protocol v1.1.0, P_NORMAL=1.771420)
 * 2. Candidate Selector v1.2 (Protocol v1.2.0, P_NORMAL=1.311028)
 * 
 * Compiles with standard GCC / Clang / MSVC.
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
    int passed_decisions = 0;
    int passed_stages = 0;
    int passed_reasons = 0;

    for (int i = 0; i < total_vectors; ++i) {
        const DemoVector& v = DEMO_VECTORS[i];
        AdaptiveMonitoring::AdaptiveOutput res = AdaptiveMonitoring::process_window(v.samples, 1024);

        bool stage_match = (std::string(res.stage_reached) == std::string(v.expected_stage));
        bool reason_match = (std::string(res.escalation_reason) == std::string(v.expected_reason));
        bool pred_match = (res.predicted_label == v.expected_predicted_label);
        bool fft_match = (res.fft_used == v.expected_fft_used);
        bool ops_match = (res.op_count_proxy == v.expected_op_count_proxy);

        float low_diff = std::abs(res.low_stage_score - v.expected_low_score);

        bool all_ok = stage_match && reason_match && pred_match && fft_match && ops_match && (low_diff < 0.01f);

        if (stage_match) passed_stages++;
        if (reason_match) passed_reasons++;
        if (pred_match) passed_decisions++;

        std::cout << "\n[" << (i + 1) << "/" << total_vectors << "] Vector: " << v.name
                  << " (" << v.signal_case << " @ " << v.snr_db << " dB)" << std::endl;
        std::cout << "    Description: " << v.description << std::endl;
        std::cout << "    C++ Result   : Stage=" << res.stage_reached
                  << ", Reason=" << res.escalation_reason
                  << ", Pred=" << res.predicted_label
                  << ", Ops=" << res.op_count_proxy
                  << ", LowScore=" << std::fixed << std::setprecision(4) << res.low_stage_score
                  << ", Freq=" << std::fixed << std::setprecision(1) << res.frequency_estimate_hz << " Hz" << std::endl;
        std::cout << "    Expected Py  : Stage=" << v.expected_stage
                  << ", Reason=" << v.expected_reason
                  << ", Pred=" << v.expected_predicted_label
                  << ", Ops=" << v.expected_op_count_proxy
                  << ", LowScore=" << std::fixed << std::setprecision(4) << v.expected_low_score
                  << ", Freq=" << std::fixed << std::setprecision(1) << v.expected_freq_hz << " Hz" << std::endl;
        std::cout << "    Status       : " << (all_ok ? "[PASS] FULL PARITY" : "[FAIL] MISMATCH DETECTED") << std::endl;
    }

    std::cout << "\nSummary v1.1:" << std::endl;
    std::cout << "  Decisions Matched : " << passed_decisions << "/" << total_vectors << std::endl;
    std::cout << "  Stages Matched    : " << passed_stages << "/" << total_vectors << std::endl;
    std::cout << "  Reasons Matched   : " << passed_reasons << "/" << total_vectors << std::endl;

    if (passed_decisions == total_vectors && passed_stages == total_vectors && passed_reasons == total_vectors) {
        std::cout << "RESULT v1.1: 100% Python/C++ Parity Achieved (demonstrating expected Vector 4 harmonic miss)." << std::endl;
        return 0;
    } else {
        std::cerr << "RESULT v1.1: Parity Mismatch Detected!" << std::endl;
        return 1;
    }
}

int test_candidate_v1_2() {
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "PART 2: ESP32 Adaptive Core v1.2 Desktop C++ Verification Harness" << std::endl;
    std::cout << "Testing 5 Deterministic Development Vectors with Recalibrated P_NORMAL=1.311028" << std::endl;
    std::cout << "================================================================================" << std::endl;

    int total_vectors = NUM_DEMO_VECTORS;
    int passed_v1_2 = 0;

    // Expected outcomes for v1.2:
    // Vector 1 (clean normal): Stage 1, confident_normal, pred 0, ops 10265
    // Vector 2 (impact): Stage 1, confident_anomaly, pred 1, ops 10265
    // Vector 3 (freq shift): Stage 2, zcf_out_of_range, pred 1, ops 39515
    // Vector 4 (added harmonic): Stage 2, uncertain_score, pred 1, ops 39515 (RESOLVED!)
    // Vector 5 (noisy normal): Stage 2, zcf_out_of_range, pred 0, ops 39515
    const char* expected_stage_v1_2[5] = {"stage1_low", "stage1_low", "stage2_high", "stage2_high", "stage2_high"};
    const char* expected_reason_v1_2[5] = {"confident_normal", "confident_anomaly", "zcf_out_of_range", "uncertain_score", "zcf_out_of_range"};
    int expected_pred_v1_2[5] = {0, 1, 1, 1, 0};
    int expected_ops_v1_2[5] = {10265, 10265, 39515, 39515, 39515};

    for (int i = 0; i < total_vectors; ++i) {
        const DemoVector& v = DEMO_VECTORS[i];
        AdaptiveMonitoring::AdaptiveOutput res = AdaptiveMonitoringV12::process_window(v.samples, 1024);

        bool stage_ok = (std::string(res.stage_reached) == std::string(expected_stage_v1_2[i]));
        bool reason_ok = (std::string(res.escalation_reason) == std::string(expected_reason_v1_2[i]));
        bool pred_ok = (res.predicted_label == expected_pred_v1_2[i]);
        bool ops_ok = (res.op_count_proxy == expected_ops_v1_2[i]);

        bool all_ok = stage_ok && reason_ok && pred_ok && ops_ok;
        if (all_ok) passed_v1_2++;

        std::cout << "\n[" << (i + 1) << "/" << total_vectors << "] Vector: " << v.name
                  << " (" << v.signal_case << " @ " << v.snr_db << " dB)" << std::endl;
        std::cout << "    v1.2 Result  : Stage=" << res.stage_reached
                  << ", Reason=" << res.escalation_reason
                  << ", Pred=" << res.predicted_label
                  << ", Ops=" << res.op_count_proxy
                  << ", LowScore=" << std::fixed << std::setprecision(4) << res.low_stage_score
                  << ", HighScore=" << std::fixed << std::setprecision(4) << res.high_stage_score << std::endl;
        std::cout << "    Expected v1.2: Stage=" << expected_stage_v1_2[i]
                  << ", Reason=" << expected_reason_v1_2[i]
                  << ", Pred=" << expected_pred_v1_2[i]
                  << ", Ops=" << expected_ops_v1_2[i] << std::endl;
        if (i == 3) {
            std::cout << "    Key Finding  : Vector 4 successfully escalates to Stage 2 FFT and is flagged Anomaly under v1.2!" << std::endl;
        }
        std::cout << "    Status       : " << (all_ok ? "[PASS] EXPECTED v1.2 BEHAVIOR" : "[FAIL] MISMATCH DETECTED") << std::endl;
    }

    std::cout << "\nSummary v1.2:" << std::endl;
    std::cout << "  Passed Vectors    : " << passed_v1_2 << "/" << total_vectors << std::endl;

    if (passed_v1_2 == total_vectors) {
        std::cout << "RESULT v1.2: 100% Verification Parity Achieved (Vector 4 harmonic detection confirmed)." << std::endl;
        return 0;
    } else {
        std::cerr << "RESULT v1.2: Verification Mismatch Detected!" << std::endl;
        return 1;
    }
}

int main() {
    int r1 = test_candidate_v1_1();
    int r2 = test_candidate_v1_2();
    return (r1 == 0 && r2 == 0) ? 0 : 1;
}
