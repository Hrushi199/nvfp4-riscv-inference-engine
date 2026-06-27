//===========================================================
// TESTBENCH: NVFP4 APU ADAPTER (SystemVerilog)
//===========================================================
//
// Tests the NVFP4 accelerator through the APU interface,
// simulating exactly what the CV32E40P core does when it
// dispatches FP/custom instructions to the coprocessor.
//
//===========================================================

`timescale 1ns / 1ps

module tb_nvfp4_apu;

    import cv32e40p_apu_core_pkg::*;

    // Clock & Reset
    logic clk, rst_n;

    // APU signals
    logic                              apu_req;
    logic                              apu_gnt;
    logic [   APU_NARGS_CPU-1:0][31:0] apu_operands;
    logic [     APU_WOP_CPU-1:0]       apu_op;
    logic [APU_NDSFLAGS_CPU-1:0]       apu_flags;
    logic                              apu_rvalid;
    logic [                31:0]       apu_result;
    logic [APU_NUSFLAGS_CPU-1:0]       apu_rflags;

    // DUT
    nvfp4_apu_adapter dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .apu_req_i      (apu_req),
        .apu_gnt_o      (apu_gnt),
        .apu_operands_i (apu_operands),
        .apu_op_i       (apu_op),
        .apu_flags_i    (apu_flags),
        .apu_rvalid_o   (apu_rvalid),
        .apu_result_o   (apu_result),
        .apu_rflags_o   (apu_rflags)
    );

    // Clock 100MHz
    initial clk = 0;
    always #5 clk = ~clk;

    // Op codes
    localparam OP_LOAD_W = 6'b000001;
    localparam OP_LOAD_A = 6'b000010;
    localparam OP_MAC    = 6'b000100;
    localparam OP_STORE  = 6'b001000;

    int pass_count, fail_count;

    // ---- APU dispatch task ----
    task automatic apu_dispatch(
        input logic [5:0]  op,
        input logic [31:0] op0,
        input logic [31:0] op1,
        input logic [31:0] op2
    );
        @(negedge clk);
        apu_req        = 1'b1;
        apu_op         = op;
        apu_operands[0] = op0;
        apu_operands[1] = op1;
        apu_operands[2] = op2;
        apu_flags      = '0;

        // Wait for grant
        @(posedge clk);
        #1;
        while (!apu_gnt) @(posedge clk);

        @(negedge clk);
        apu_req = 1'b0;

        // Wait for result valid
        #1;
        while (!apu_rvalid) begin
            @(posedge clk);
            #1;
        end
    endtask

    // ---- Full test: LOAD_W ? LOAD_A ? MAC ? check result ----
    task automatic run_nvfp4_test(
        input logic [63:0] weights,
        input logic [63:0] acts,
        input logic [31:0] scale,
        input logic [31:0] expected,
        input int          tnum,
        input string       desc
    );
        // Step 1: Load weights
        apu_dispatch(OP_LOAD_W, weights[31:0], weights[63:32], 32'd0);

        // Step 2: Load activations
        apu_dispatch(OP_LOAD_A, acts[31:0], acts[63:32], 32'd0);

        // Step 3: MAC (set scale + trigger compute)
        apu_dispatch(OP_MAC, scale, 32'd0, 32'd0);

        // Check result from MAC
        if (apu_result === expected) begin
            $display("  TEST %0d PASSED: result=0x%08h  (%s)", tnum, apu_result, desc);
            pass_count++;
        end else begin
            $display("  TEST %0d FAILED: result=0x%08h expected=0x%08h  (%s)",
                     tnum, apu_result, expected, desc);
            fail_count++;
        end
    endtask

    // =========================================================
    // MAIN
    // =========================================================
    initial begin
        rst_n = 0;
        apu_req = 0;
        apu_op = '0;
        apu_operands = '{default: '0};
        apu_flags = '0;
        pass_count = 0;
        fail_count = 0;

        #30;
        rst_n = 1;
        #20;

        $display("");
        $display("==================================================");
        $display("  NVFP4 APU ADAPTER TESTBENCH");
        $display("  Testing via CV32E40P APU interface");
        $display("==================================================");
        $display("");

        // TEST 1: All 1.0 ? 16.0
        run_nvfp4_test(
            64'h2222222222222222, 64'h2222222222222222,
            32'h3E800000, 32'h41800000, 1, "16x(1.0*1.0)*0.25=16.0"
        );

        // TEST 2: All zeros ? 0.0
        run_nvfp4_test(
            64'h0, 64'h0,
            32'h3E800000, 32'h00000000, 2, "all zeros"
        );

        // TEST 3: 6.0*6.0 ? 36.0
        run_nvfp4_test(
            64'h7, 64'h7,
            32'h3E800000, 32'h42100000, 3, "6.0*6.0=36.0"
        );

        // TEST 4: Mixed ? 9.0
        run_nvfp4_test(
            64'h13D7, 64'h7642,
            32'h3E800000, 32'h41100000, 4, "mixed=9.0"
        );

        // TEST 5: Negative ? -24.0
        run_nvfp4_test(
            64'hF, 64'h6,
            32'h3E800000, 32'hC1C00000, 5, "negative=-24.0"
        );

        // TEST 6: Small ? 0.25
        run_nvfp4_test(
            64'h1, 64'h1,
            32'h3E800000, 32'h3E800000, 6, "small=0.25"
        );

        // TEST 7: All max ? 576.0
        run_nvfp4_test(
            64'h7777777777777777, 64'h7777777777777777,
            32'h3E800000, 32'h44100000, 7, "max=576.0"
        );

        // Summary
        #20;
        $display("");
        $display("==================================================");
        $display("  SUMMARY: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("==================================================");
        if (fail_count == 0) $display("  >>> ALL TESTS PASSED <<<");
        else                 $display("  >>> SOME TESTS FAILED <<<");
        $display("");
        $finish;
    end

endmodule
