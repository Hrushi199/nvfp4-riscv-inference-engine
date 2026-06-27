//===========================================================
// TESTBENCH: NVFP4 DOT-PRODUCT ACCELERATOR (SystemVerilog)
//===========================================================
//
// Self-checking testbench with hand-verified test vectors.
// Each test provides packed NVFP4 weight/activation blocks,
// a combined FP32 scale factor, and the expected FP32 result.
//
// Test vectors are derived from the Python NVFP4 lookup table
// and manually verified using the formula:
//   result = SUM(decode(w[i]) * decode(a[i])) * scale_w * scale_a
//
// To run with Icarus Verilog (SystemVerilog mode):
//   iverilog -g2012 -o tb_nvfp4 tb_nvfp4_accelerator.sv \
//            nvfp4_accelerator_top.sv nvfp4_dot_product_16.sv \
//            nvfp4_scale_multiply.sv nvfp4_mac_unit.sv \
//            nvfp4_decoder.sv nvfp4_multiplier.sv \
//            nvfp4_extractor.sv
//   vvp tb_nvfp4
//   gtkwave tb_nvfp4_accelerator.vcd   (optional waveform viewer)
//
//===========================================================

`timescale 1ns / 1ps

module tb_nvfp4_accelerator;

    // ---- Clock & Reset ----
    logic clk;
    logic rst_n;

    // ---- DUT Interface ----
    logic         start;
    logic [63:0]  weight_packed;
    logic [63:0]  act_packed;
    logic [31:0]  combined_scale;

    logic         done;
    logic         busy;
    logic [31:0]  fp32_result;
    logic signed [12:0] dbg_dot_int;

    // ---- DUT Instantiation ----
    nvfp4_accelerator_top dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start),
        .done           (done),
        .busy           (busy),
        .weight_packed  (weight_packed),
        .act_packed     (act_packed),
        .combined_scale (combined_scale),
        .fp32_result    (fp32_result),
        .dbg_dot_int    (dbg_dot_int)
    );

    // ---- Clock Generation: 100 MHz (10ns period) ----
    initial clk = 0;
    always #5 clk = ~clk;

    // ---- Test Counters ----
    int pass_count;
    int fail_count;

    // ---- Helper Task: Run One Test Vector ----
    task automatic run_test(
        input logic [63:0]       w_packed,
        input logic [63:0]       a_packed,
        input logic [31:0]       scale,
        input logic [31:0]       expected,
        input logic signed [12:0] expected_dot_int,
        input int                tnum
    );
        // Setup inputs on falling edge (stable before rising)
        @(negedge clk);
        weight_packed  = w_packed;
        act_packed     = a_packed;
        combined_scale = scale;
        start          = 1'b1;

        // Deassert start after one cycle
        @(negedge clk);
        start = 1'b0;

        // Wait for COMPUTE cycle (combinational logic settles)
        @(posedge clk);

        // Wait for done (result registered at end of COMPUTE)
        @(posedge clk);
        #1;  // small delta to read post-NBA values

        // ---- Check integer dot product ----
        if (dbg_dot_int !== expected_dot_int) begin
            $display("  [WARN] TEST %0d: dot_int = %0d (expected %0d)",
                     tnum, dbg_dot_int, expected_dot_int);
        end

        // ---- Check FP32 result ----
        if (fp32_result === expected) begin
            $display("  TEST %0d PASSED: fp32_result = 0x%08h  (dot_int = %0d)",
                     tnum, fp32_result, dbg_dot_int);
            pass_count++;
        end else begin
            $display("  TEST %0d FAILED: fp32_result = 0x%08h  expected = 0x%08h  (dot_int = %0d)",
                     tnum, fp32_result, expected, dbg_dot_int);
            fail_count++;
        end
    endtask

    // =========================================================
    // MAIN TEST SEQUENCE
    // =========================================================
    initial begin
        // ---- Waveform dump (Icarus only; Vivado uses GUI waveform) ----
        `ifndef XILINX_SIMULATOR
            $dumpfile("tb_nvfp4_accelerator.vcd");
            $dumpvars(0, tb_nvfp4_accelerator);
        `endif

        // ---- Initialize ----
        rst_n          = 0;
        start          = 0;
        weight_packed  = 64'd0;
        act_packed     = 64'd0;
        combined_scale = 32'd0;
        pass_count     = 0;
        fail_count     = 0;

        // ---- Reset Sequence ----
        #20;
        rst_n = 1;
        #10;

        $display("");
        $display("==================================================");
        $display("  NVFP4 ACCELERATOR TESTBENCH (SystemVerilog)");
        $display("==================================================");
        $display("");

        // ====================================================
        // TEST 1: All weights=1.0, all activations=1.0
        // ====================================================
        // Code for 1.0 = 0010 (4 bits)
        // Packed: 16 x 0x2 = 0x2222_2222_2222_2222
        // scale_w = scale_a = 1.0
        // combined_scale = (1.0 * 1.0) / 4.0 = 0.25 = 0x3E800000
        // Decoded x2: each = 2, product = 2*2 = 4, sum = 16*4 = 64
        // FP32: 64 * 0.25 = 16.0 = 0x41800000
        run_test(
            64'h2222222222222222,   // weights: all 1.0
            64'h2222222222222222,   // activations: all 1.0
            32'h3E800000,          // combined_scale = 0.25
            32'h41800000,          // expected = 16.0
            13'sd64,               // expected dot_int
            1
        );

        // ====================================================
        // TEST 2: All zeros
        // ====================================================
        // Expected: 0.0 = 0x00000000
        run_test(
            64'h0000000000000000,   // weights: all 0.0
            64'h0000000000000000,   // activations: all 0.0
            32'h3E800000,          // combined_scale = 0.25
            32'h00000000,          // expected = 0.0
            13'sd0,                // expected dot_int
            2
        );

        // ====================================================
        // TEST 3: Single element: 6.0 x 6.0
        // ====================================================
        // Weight[0] = 6.0 (code 0111), rest = 0
        // Act[0] = 6.0 (code 0111), rest = 0
        // packed_w = 0x0000_0000_0000_0007
        // packed_a = 0x0000_0000_0000_0007
        // Decoded x2: 12 * 12 = 144, dot_int = 144
        // FP32: 144 * 0.25 = 36.0 = 0x42100000
        run_test(
            64'h0000000000000007,   // weight[0] = 6.0
            64'h0000000000000007,   // act[0] = 6.0
            32'h3E800000,          // combined_scale = 0.25
            32'h42100000,          // expected = 36.0
            13'sd144,              // expected dot_int
            3
        );

        // ====================================================
        // TEST 4: Mixed positive & negative
        // ====================================================
        // Weight: [6.0, -3.0, 1.5, 0.5, 0, ...]
        //   codes: [0111, 1101, 0011, 0001, 0000, ...]
        //   packed: bits[3:0]=7, [7:4]=D, [11:8]=3, [15:12]=1
        //         = 0x0000_0000_0000_13D7
        //
        // Act:    [1.0, 2.0, 4.0, 6.0, 0, ...]
        //   codes: [0010, 0100, 0110, 0111, 0000, ...]
        //   packed: bits[3:0]=2, [7:4]=4, [11:8]=6, [15:12]=7
        //         = 0x0000_0000_0000_7642
        //
        // scale_w=2.0, scale_a=0.5
        // combined_scale = (2.0 * 0.5) / 4.0 = 0.25 = 0x3E800000
        //
        // Integer products:
        //   (12*2) + (-6*4) + (3*8) + (1*12) = 24 - 24 + 24 + 12 = 36
        //
        // FP32: 36 * 0.25 = 9.0 = 0x41100000
        run_test(
            64'h00000000000013D7,   // mixed weights
            64'h0000000000007642,   // mixed activations
            32'h3E800000,          // combined_scale = 0.25
            32'h41100000,          // expected = 9.0
            13'sd36,               // expected dot_int
            4
        );

        // ====================================================
        // TEST 5: Negative result
        // ====================================================
        // Weight[0] = -6.0 (code 1111 = 0xF), rest = 0
        // Act[0] = 4.0 (code 0110 = 0x6), rest = 0
        // packed_w = 0x000000000000000F
        // packed_a = 0x0000000000000006
        //
        // Decoded x2: (-12) * 8 = -96, dot_int = -96
        // FP32: -96 * 0.25 = -24.0
        //   24 = 11000 = 1.1 x 2^4, exp = 4+127 = 131
        //   -24.0 = 0xC1C00000
        run_test(
            64'h000000000000000F,   // weight[0] = -6.0
            64'h0000000000000006,   // act[0] = 4.0
            32'h3E800000,          // combined_scale = 0.25
            32'hC1C00000,          // expected = -24.0
            -13'sd96,              // expected dot_int
            5
        );

        // ====================================================
        // TEST 6: Small value: 0.5 x 0.5
        // ====================================================
        // Weight[0] = 0.5 (code 0001), rest = 0
        // Act[0] = 0.5 (code 0001), rest = 0
        // Decoded x2: 1 * 1 = 1, dot_int = 1
        // FP32: 1 * 0.25 = 0.25 = 0x3E800000
        run_test(
            64'h0000000000000001,   // weight[0] = 0.5
            64'h0000000000000001,   // act[0] = 0.5
            32'h3E800000,          // combined_scale = 0.25
            32'h3E800000,          // expected = 0.25
            13'sd1,                // expected dot_int
            6
        );

        // ====================================================
        // TEST 7: All max positive: 16 x (6.0 * 6.0) = 576
        // ====================================================
        // All codes = 0111 (6.0)
        // packed = 0x7777_7777_7777_7777
        // dot_int = 16 * (12*12) = 16 * 144 = 2304
        // FP32: 2304 * 0.25 = 576.0
        //   576 = 1001000000 = 1.001 x 2^9, exp = 9+127 = 136
        //   576.0 = 0x44100000
        run_test(
            64'h7777777777777777,   // all 6.0
            64'h7777777777777777,   // all 6.0
            32'h3E800000,          // combined_scale = 0.25
            32'h44100000,          // expected = 576.0
            13'sd2304,             // expected dot_int
            7
        );

        // ====================================================
        // SUMMARY
        // ====================================================
        #20;
        $display("");
        $display("==================================================");
        $display("  TEST SUMMARY: %0d PASSED, %0d FAILED (of %0d)",
                 pass_count, fail_count, pass_count + fail_count);
        $display("==================================================");
        $display("");

        if (fail_count == 0)
            $display("  >>> ALL TESTS PASSED <<<");
        else
            $display("  >>> SOME TESTS FAILED <<<");

        $display("");
        $finish;
    end

endmodule
