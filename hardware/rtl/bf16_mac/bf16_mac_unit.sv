//===========================================================
// BF16 MAC UNIT (SystemVerilog)
//===========================================================
//
// Computes a dot-product of 4x BF16 weights and 4x BF16
// activations and returns the result in FP32.
//
// BF16 Format: 1-bit sign | 8-bit exponent | 7-bit mantissa
// This is identical to the top 16 bits of IEEE 754 FP32.
//
// Therefore, BF16 multiplication is performed by:
//   1. Sign-extending each BF16 to FP32 (append 16'h0000)
//   2. Using the standard FP32 multiplication algorithm
//   3. Accumulating 4 products in FP32
//
// Since Vivado/XSim supports real arithmetic in simulation
// and since a full IEEE-754 FP multiplier in RTL is complex,
// we implement this using the shortcutting property:
//   BF16 × BF16 = FP32 (exact representation if no rounding)
//
// For synthesis-grade operation, we unpack BF16 to FP32,
// multiply using a pipelined FP32 multiplier model, then
// accumulate with an FP32 adder.
//
// Inputs:
//   clk          : system clock
//   rst_n        : active-low reset
//   start        : begin computation
//   weight_bf16  : 4× BF16 packed [63:0] = {w3,w2,w1,w0}
//   act_bf16     : 4× BF16 packed [63:0] = {a3,a2,a1,a0}
//
// Outputs:
//   done         : result ready (next cycle after start)
//   result_fp32  : FP32 dot product
//
//===========================================================

`timescale 1ns / 1ps

module bf16_mac_unit (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [63:0] weight_bf16,   // 4× BF16 packed
    input  logic [63:0] act_bf16,      // 4× BF16 packed
    output logic        done,
    output logic [31:0] result_fp32    // IEEE 754 FP32
);

    //=========================================================
    // Unpack 4× BF16 from 64-bit packed input
    // BF16 → FP32: append 16'h0000 to the mantissa
    //=========================================================
    logic [31:0] w [0:3];  // weights in FP32
    logic [31:0] a [0:3];  // activations in FP32

    // BF16 is just FP32 with truncated lower 16 mantissa bits
    // So BF16-to-FP32 = {bf16_bits, 16'h0000}
    always_comb begin
        w[0] = {weight_bf16[15:0],  16'h0000};
        w[1] = {weight_bf16[31:16], 16'h0000};
        w[2] = {weight_bf16[47:32], 16'h0000};
        w[3] = {weight_bf16[63:48], 16'h0000};

        a[0] = {act_bf16[15:0],  16'h0000};
        a[1] = {act_bf16[31:16], 16'h0000};
        a[2] = {act_bf16[47:32], 16'h0000};
        a[3] = {act_bf16[63:48], 16'h0000};
    end

    //=========================================================
    // FP32 Multiply-Accumulate
    // Since we are doing simulation-grade RTL, we implement
    // IEEE 754 FP32 multiply as bit-manipulation:
    //   sign    = sign_a XOR sign_b
    //   exp     = exp_a + exp_b - 127 (unbias once)
    //   mantissa= mantissa_a × mantissa_b (24×24 bit)
    // Then accumulate 4 products using FP32 addition.
    //=========================================================

    // FP32 multiply: returns sign-exp-mantissa of a*b
    function automatic logic [31:0] fp32_mul(input logic [31:0] a_in, b_in);
        logic        sa, sb, sr;
        logic [7:0]  ea, eb;
        logic [22:0] ma, mb;
        logic [8:0]  er;
        logic [47:0] mr_full;
        logic [22:0] mr;
        logic        norm_shift;

        // Special case: zero
        if ((a_in[30:0] == 0) || (b_in[30:0] == 0))
            return 32'h00000000;

        sa = a_in[31]; sb = b_in[31];
        ea = a_in[30:23]; eb = b_in[30:23];
        ma = a_in[22:0]; mb = b_in[22:0];

        // Result sign
        sr = sa ^ sb;

        // Multiply mantissas (with implicit 1)
        mr_full = ({1'b1, ma}) * ({1'b1, mb});

        // Result exponent (remove double bias)
        er = {1'b0, ea} + {1'b0, eb} - 9'd127;

        // Normalize: if bit 47 set, shift right by 1
        norm_shift = mr_full[47];
        if (norm_shift) begin
            mr = mr_full[46:24];
            er = er + 1;
        end else begin
            mr = mr_full[45:23];
        end

        // Check for overflow/underflow
        if (er[8]) return {sr, 8'hFF, 23'h0};  // Overflow → Inf
        if (er == 0) return 32'h00000000;       // Underflow → 0

        return {sr, er[7:0], mr};
    endfunction

    // FP32 add: returns a + b
    function automatic logic [31:0] fp32_add(input logic [31:0] a_in, b_in);
        logic        sa, sb, sr;
        logic [7:0]  ea, eb;
        logic [22:0] ma, mb;
        logic [7:0]  diff;
        logic [24:0] ma_full, mb_full, sum_full;
        logic [7:0]  er;
        logic [22:0] mr;
        logic [7:0]  lz;

        // Special case: one zero
        if (a_in[30:0] == 0) return b_in;
        if (b_in[30:0] == 0) return a_in;

        sa = a_in[31]; sb = b_in[31];
        ea = a_in[30:23]; eb = b_in[30:23];
        ma = a_in[22:0]; mb = b_in[22:0];

        // Align to larger exponent
        if (ea >= eb) begin
            diff   = ea - eb;
            ma_full = {2'b01, ma};
            mb_full = ({2'b01, mb}) >> diff;
            er = ea;
        end else begin
            diff   = eb - ea;
            mb_full = {2'b01, mb};
            ma_full = ({2'b01, ma}) >> diff;
            er = eb;
        end

        if (sa == sb) begin
            // Same sign: add
            sr = sa;
            sum_full = ma_full + mb_full;
            if (sum_full[24]) begin
                mr = sum_full[23:1];
                er = er + 1;
            end else begin
                mr = sum_full[22:0];
            end
        end else begin
            // Different sign: subtract
            if (ma_full >= mb_full) begin
                sr = sa;
                sum_full = ma_full - mb_full;
            end else begin
                sr = sb;
                sum_full = mb_full - ma_full;
            end
            if (sum_full == 0) return 32'h00000000;
            // Normalize
            mr = sum_full[22:0];
            // Find leading zero (simplified - check top bits)
            if (!sum_full[23]) begin
                // shift left until 1 in bit 23
                sum_full = sum_full << 1;
                er = er - 1;
                mr = sum_full[22:0];
            end
        end

        if (er == 8'hFF) return {sr, 8'hFF, 23'h0};  // Inf
        return {sr, er, mr};
    endfunction

    //=========================================================
    // Sequential computation (registered for timing)
    //=========================================================
    logic [31:0] p [0:3];   // 4 partial products
    logic [31:0] acc01, acc23, acc_final;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done        <= 1'b0;
            result_fp32 <= 32'h00000000;
        end else begin
            done <= 1'b0;

            if (start) begin
                // Multiply all 4 pairs
                p[0] = fp32_mul(w[0], a[0]);
                p[1] = fp32_mul(w[1], a[1]);
                p[2] = fp32_mul(w[2], a[2]);
                p[3] = fp32_mul(w[3], a[3]);

                // Accumulate
                acc01      = fp32_add(p[0], p[1]);
                acc23      = fp32_add(p[2], p[3]);
                acc_final  = fp32_add(acc01, acc23);

                result_fp32 <= acc_final;
                done        <= 1'b1;
            end
        end
    end

endmodule
