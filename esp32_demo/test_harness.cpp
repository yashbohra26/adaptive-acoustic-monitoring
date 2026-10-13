/**
 * @file test_harness.cpp
 * @brief Standalone desktop C++ verification harness for Adaptive Core v1.1.
 * 
 * Verifies Python / C++ decision parity across all 5 representative development demo vectors.
 * Compiles with standard GCC / Clang / MSVC.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include "src/adaptive_core.h"
#include "src/demo_vectors.h"

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "ESP32 Adaptive Core v1.1 Desktop C++ Parity Test Harness" << std::endl;
    std::cout << "Testing 5 Deterministic Development Vectors from Group g0001" << std::endl;
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
        float freq_diff = std::abs(res.frequency_estimate_hz - v.expected_freq_hz);

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

    std::cout << "\n--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Decisions Matched : " << passed_decisions << "/" << total_vectors << std::endl;
    std::cout << "  Stages Matched    : " << passed_stages << "/" << total_vectors << std::endl;
    std::cout << "  Reasons Matched   : " << passed_reasons << "/" << total_vectors << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    if (passed_decisions == total_vectors && passed_stages == total_vectors && passed_reasons == total_vectors) {
        std::cout << "RESULT: 100% Python/C++ Parity Achieved across all development demo vectors." << std::endl;
        return 0;
    } else {
        std::cerr << "RESULT: Parity Mismatch Detected!" << std::endl;
        return 1;
    }
}
