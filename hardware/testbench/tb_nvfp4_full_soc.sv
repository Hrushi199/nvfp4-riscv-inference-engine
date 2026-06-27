//===========================================================
// TESTBENCH: FULL SoC - CV32E40P + NVFP4 + BF16 (Mixed-Precision)
//===========================================================
//
// Verifies 7 NVFP4 tests + 7 BF16 tests executed sequentially
// by the CV32E40P CPU via custom ISA instructions.
//
// Data RAM layout (word-addressed):
//   [0..6]  = NVFP4 results
//   [7..13] = BF16  results
//   [14]    = done flag (= 1 when CPU finishes)
//
//===========================================================
`timescale 1ns / 1ps

module tb_nvfp4_full_soc;

    logic clk, rst_n;

    nvfp4_soc_top #(
        .INSTR_RAM_DEPTH (16384),
        .DATA_RAM_DEPTH  (16384)
    ) dut (
        .clk_i  (clk),
        .rst_ni (rst_n)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // ---- NVFP4 Expected Results (data_mem[0..6]) ----
    logic [31:0] nvfp4_exp[7] = '{
        32'h41800000,  // 16.0
        32'h00000000,  //  0.0
        32'h42100000,  // 36.0
        32'h41100000,  //  9.0
        32'hC1C00000,  // -24.0
        32'h3E800000,  //  0.25
        32'h44100000   // 576.0
    };
    real nvfp4_float[7] = '{16.0, 0.0, 36.0, 9.0, -24.0, 0.25, 576.0};

    // ---- BF16 Expected Results (data_mem[7..13]) ----
    logic [31:0] bf16_exp[7] = '{
        32'h40800000,  //  4.0
        32'h41600000,  // 14.0
        32'h41300000,  // 11.0
        32'h00000000,  //  0.0
        32'hC1200000,  // -10.0
        32'h3F700000,  //  0.9375
        32'h42100000   // 36.0
    };
    real bf16_float[7] = '{4.0, 14.0, 11.0, 0.0, -10.0, 0.9375, 36.0};

    localparam MAX_CYCLES = 100000;
    integer cycle_count, errors, i;

    initial begin
        rst_n = 0; #100; rst_n = 1;

        $display("");
        $display("=======================================================");
        $display("  MIXED-PRECISION SoC: CV32E40P + NVFP4 + BF16");
        $display("=======================================================");
        $display("  7 NVFP4 tests  -> data_mem[0..6]");
        $display("  7 BF16  tests  -> data_mem[7..13]");
        $display("  Done flag      -> data_mem[14]");
        $display("=======================================================");
        $display("  Booting CPU...");

        cycle_count = 0;
        while (cycle_count < MAX_CYCLES) begin
            @(posedge clk); cycle_count++;

            // Done flag is at word index 14
            if (dut.data_mem[14] == 32'h00000001) begin

                $display("");
                $display("  CPU finished in %0d cycles.", cycle_count);
                $display("");
                errors = 0;

                // ---- Verify NVFP4 ----
                $display("  +--------------------------------------------------+");
                $display("  |        NVFP4 Results (data_mem[0..6])            |");
                $display("  +------+--------------------+----------+------------+");
                $display("  | Test |       Result       | Expected |   Status   |");
                $display("  +------+--------------------+----------+------------+");
                for (i = 0; i < 7; i++) begin
                    if (dut.data_mem[i] === nvfp4_exp[i])
                        $display("  |  %0d   | 0x%08h (%7.2f) |  %7.2f | [ PASS ]   |",
                                 i+1, dut.data_mem[i], nvfp4_float[i], nvfp4_float[i]);
                    else begin
                        $display("  |  %0d   | 0x%08h          |  0x%08h | [ FAIL ]   |",
                                 i+1, dut.data_mem[i], nvfp4_exp[i]);
                        errors++;
                    end
                end
                $display("  +------+--------------------+----------+------------+");

                // ---- Verify BF16 ----
                $display("");
                $display("  +--------------------------------------------------+");
                $display("  |        BF16  Results (data_mem[7..13])           |");
                $display("  +------+--------------------+----------+------------+");
                $display("  | Test |       Result       | Expected |   Status   |");
                $display("  +------+--------------------+----------+------------+");
                for (i = 0; i < 7; i++) begin
                    if (dut.data_mem[7+i] === bf16_exp[i])
                        $display("  |  %0d   | 0x%08h (%7.4f) |  %7.4f | [ PASS ]   |",
                                 i+1, dut.data_mem[7+i], bf16_float[i], bf16_float[i]);
                    else begin
                        $display("  |  %0d   | 0x%08h          |  0x%08h | [ FAIL ]   |",
                                 i+1, dut.data_mem[7+i], bf16_exp[i]);
                        errors++;
                    end
                end
                $display("  +------+--------------------+----------+------------+");

                $display("");
                $display("  =======================================================");
                if (errors == 0) begin
                    $display("  >>> ALL 14 MIXED-PRECISION TESTS PASSED! <<<");
                    $display("  >>> 7 NVFP4  +  7 BF16  =  14 total <<<");
                end else
                    $display("  >>> FAILED: %0d error(s) out of 14 tests <<<", errors);
                $display("  =======================================================");
                $display("");
                $finish;
            end
        end

        $display("  [TIMEOUT] done flag not asserted within %0d cycles!", MAX_CYCLES);
        $display("  data_mem[14] = 0x%08h", dut.data_mem[14]);
        $finish;
    end

endmodule
