/**
 * @file main.cpp
 * @brief ESP32 Arduino Demonstration of Candidate Selector v1.1 for Wokwi.
 * 
 * Demonstrates the adaptive decision flow on a simulated ESP32:
 * - Runs Fixed-Low time-domain features first (10,265 ops proxy).
 * - Decides confident cases cheaply at Stage 1.
 * - Escalate to Fixed-High FFT stage (39,515 ops proxy) only when required.
 * - Evaluates 5 representative development vectors from Group g0001.
 * 
 * Target: ESP32 (e.g. ESP32-WROOM-32 on Wokwi)
 * Baud rate: 115200
 */

#include <Arduino.h>
#include "adaptive_core.h"
#include "demo_vectors.h"

void printBanner() {
    Serial.println();
    Serial.println("================================================================================");
    Serial.println("     ESP32 Adaptive Acoustic Anomaly Monitor - Firmware Demonstration           ");
    Serial.println("     Protocol v1.1.0 (Candidate Selector: cand_v1.1_p60_anomaly_first)          ");
    Serial.println("================================================================================");
    Serial.println("Frozen Calibration Boundaries (Development Partition):");
    Serial.print("  P_ANOMALY (Stage 1 Anomaly Threshold) : "); Serial.println(AdaptiveMonitoring::P_ANOMALY, 4);
    Serial.print("  P_NORMAL  (Stage 1 Normal Threshold)  : "); Serial.println(AdaptiveMonitoring::P_NORMAL, 4);
    Serial.print("  ZCF Bounds (Clean Machine Range)      : [");
    Serial.print(AdaptiveMonitoring::ZCF_MIN, 1); Serial.print(" Hz, ");
    Serial.print(AdaptiveMonitoring::ZCF_MAX, 1); Serial.println(" Hz]");
    Serial.print("  HIGH_THRESHOLD (FFT Distance)         : "); Serial.println(AdaptiveMonitoring::HIGH_THRESHOLD, 4);
    Serial.println("--------------------------------------------------------------------------------");
    Serial.println("DISCLAIMER: Firmware-flow illustration on deterministic development vectors.");
    Serial.println("Synthetic signals only. Operation count proxy does NOT represent measured energy.");
    Serial.println("================================================================================\n");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    printBanner();
}

void loop() {
    static int cycle_count = 1;
    Serial.print(">>> Starting Demonstration Cycle #");
    Serial.print(cycle_count++);
    Serial.println(" <<<\n");

    int matched_count = 0;

    for (int i = 0; i < NUM_DEMO_VECTORS; ++i) {
        const DemoVector& vec = DEMO_VECTORS[i];

        // Measure firmware execution timing (microsecond benchmark)
        uint32_t t_start = micros();
        AdaptiveMonitoring::AdaptiveOutput res = AdaptiveMonitoring::process_window(vec.samples, 1024);
        uint32_t t_elapsed_us = micros() - t_start;

        bool parity_ok = (res.predicted_label == vec.expected_predicted_label) &&
                         (res.fft_used == vec.expected_fft_used) &&
                         (res.op_count_proxy == vec.expected_op_count_proxy);
        if (parity_ok) matched_count++;

        Serial.println("--------------------------------------------------------------------------------");
        Serial.print("Vector ["); Serial.print(i + 1); Serial.print("/"); Serial.print(NUM_DEMO_VECTORS); Serial.print("]: ");
        Serial.print(vec.name);
        Serial.print(" (Case: "); Serial.print(vec.signal_case);
        Serial.print(", SNR: "); Serial.print(vec.snr_db, 0); Serial.println(" dB)");
        Serial.print("Context       : "); Serial.println(vec.description);

        Serial.print("Stage 1 Output: LowScore = "); Serial.print(res.low_stage_score, 4);
        Serial.print(", ZCF = "); Serial.print(res.zcf_hz, 1); Serial.println(" Hz");

        Serial.print("Decision Flow : Stage = "); Serial.print(res.stage_reached);
        Serial.print(" (FFT Escalated: "); Serial.print(res.fft_used ? "YES" : "NO");
        Serial.print("), Reason = "); Serial.println(res.escalation_reason);

        if (res.fft_used) {
            Serial.print("Stage 2 FFT   : HighScore = "); Serial.print(res.high_stage_score, 4);
            Serial.print(", PeakFreq = "); Serial.print(res.frequency_estimate_hz, 1); Serial.println(" Hz");
        }

        Serial.print("Final Decision: Predicted Label = ");
        Serial.print(res.predicted_label == 1 ? "ANOMALY (1)" : "NORMAL (0)");
        Serial.print(" [Expected Py: ");
        Serial.print(vec.expected_predicted_label == 1 ? "ANOMALY (1)" : "NORMAL (0)"); Serial.println("]");

        Serial.print("Work Proxy    : "); Serial.print(res.op_count_proxy); Serial.print(" ops");
        Serial.print(" (Execution time: "); Serial.print(t_elapsed_us); Serial.println(" us)");

        Serial.print("Parity Status : ");
        if (parity_ok) {
            Serial.println("[MATCH] Full parity with Python Protocol v1.1.0 output");
        } else {
            Serial.println("[MISMATCH] Discrepancy detected relative to Python");
        }
        Serial.println();

        delay(800);
    }

    Serial.println("================================================================================");
    Serial.print("Cycle Complete. Vectors Matched: "); Serial.print(matched_count);
    Serial.print("/"); Serial.print(NUM_DEMO_VECTORS);
    Serial.println(". Pausing before next cycle...\n");

    delay(4000);
}
