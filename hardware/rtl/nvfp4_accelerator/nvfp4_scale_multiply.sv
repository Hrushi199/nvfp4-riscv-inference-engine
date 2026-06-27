//===========================================================
// NVFP4 SCALE MULTIPLY — SystemVerilog
//===========================================================
//
// Converts the integer dot-product result to IEEE 754 FP32
// and multiplies by the combined block scale factor.
//
// Mathematical justification:
//   The dot product engine computes:
//     dot_int = SUM( decode_x2(w[i]) * decode_x2(a[i]) )
//
//   Since both operands were scaled by 2 (to avoid fractions),
//   the integer result = true_dot_product * 4.
//
//   The software pre-computes:
//     combined_scale = (scale_w * scale_a) / 4.0
//
//   So the final result is:
//     fp32_result = dot_int * combined_scale
//                 = (true_dot * 4) * (scale_w * scale_a / 4)
//                 = true_dot * scale_w * scale_a
//                 = correct_result
//
// Implementation:
//   1. Convert 13-bit signed integer to IEEE 754 FP32
//   2. Multiply with combined_scale using FP32 arithmetic
//
// Assumptions:
//   - combined_scale is always a normalized FP32 number
//   - No denormalized, NaN, or Inf inputs
//   - Rounding: truncation (round toward zero)
//
//===========================================================

module nvfp4_scale_multiply (
    input  logic signed [12:0] dot_int,          // integer dot product from adder tree
    input  logic [31:0]        combined_scale,   // FP32: (scale_w * scale_a) / 4.0
    output logic [31:0]        fp32_result       // FP32 final dot product result
);

    // =========================================================
    // STEP 1: Detect zero inputs
    // =========================================================
    logic is_zero;
    logic scale_is_zero;
    logic either_zero;

    assign is_zero       = (dot_int == 13'sd0);
    assign scale_is_zero = (combined_scale[30:0] == 31'd0);
    assign either_zero   = is_zero | scale_is_zero;

    // =========================================================
    // STEP 2: Get sign and absolute value of the integer
    // =========================================================
    logic int_sign;
    assign int_sign = dot_int[12];

    // Two's complement negate for negative values
    // Valid range: -2304 to +2304, so abs fits in 12 bits
    logic [12:0] abs_val_13;
    logic [11:0] abs_val;

    assign abs_val_13 = int_sign ? (~dot_int + 13'd1) : dot_int;
    assign abs_val    = abs_val_13[11:0];

    // =========================================================
    // STEP 3: Integer to IEEE 754 FP32 conversion
    // =========================================================

    // Priority encoder: find position of the leading one
    // abs_val range: 1 to 2304, leading one at positions 0 to 11
    logic [3:0] lead_pos;

    always_comb begin
        casez (abs_val)
            12'b1???????????: lead_pos = 4'd11;
            12'b01??????????: lead_pos = 4'd10;
            12'b001?????????: lead_pos = 4'd9;
            12'b0001????????: lead_pos = 4'd8;
            12'b00001???????: lead_pos = 4'd7;
            12'b000001??????: lead_pos = 4'd6;
            12'b0000001?????: lead_pos = 4'd5;
            12'b00000001????: lead_pos = 4'd4;
            12'b000000001???: lead_pos = 4'd3;
            12'b0000000001??: lead_pos = 4'd2;
            12'b00000000001?: lead_pos = 4'd1;
            12'b000000000001: lead_pos = 4'd0;
            default:          lead_pos = 4'd0;  // abs_val=0, guarded by is_zero
        endcase
    end

    // Biased exponent: lead_pos + 127
    logic [7:0] int_exp;
    assign int_exp = {4'd0, lead_pos} + 8'd127;

    // Mantissa: shift abs_val left so leading 1 lands at bit 23,
    // then take bits [22:0] (stripping the implicit leading 1).
    //
    // Shift amount = 23 - lead_pos (range: 12 to 23)
    // Container: 35 bits to hold 12-bit value shifted left by up to 23
    logic [4:0]  shift_amount;
    logic [34:0] shifted_val;
    logic [22:0] int_mant;

    assign shift_amount = 5'd23 - {1'b0, lead_pos};
    assign shifted_val  = {23'b0, abs_val} << shift_amount;
    assign int_mant     = shifted_val[22:0];

    // Compose FP32 representation of the integer
    logic [31:0] int_as_fp32;
    assign int_as_fp32 = {int_sign, int_exp, int_mant};

    // =========================================================
    // STEP 4: IEEE 754 FP32 Multiply
    // =========================================================
    // result = int_as_fp32 * combined_scale

    // Extract fields of operand A (converted integer)
    logic        a_sign;
    logic [7:0]  a_exp;
    logic [22:0] a_mant;

    assign a_sign = int_as_fp32[31];
    assign a_exp  = int_as_fp32[30:23];
    assign a_mant = int_as_fp32[22:0];

    // Extract fields of operand B (combined scale)
    logic        b_sign;
    logic [7:0]  b_exp;
    logic [22:0] b_mant;

    assign b_sign = combined_scale[31];
    assign b_exp  = combined_scale[30:23];
    assign b_mant = combined_scale[22:0];

    // Result sign: XOR of input signs
    logic res_sign;
    assign res_sign = a_sign ^ b_sign;

    // Exponent addition with bias correction
    // exp_result = exp_a + exp_b - 127 (removing double bias)
    logic [8:0] exp_sum;
    logic [8:0] exp_unbias;

    assign exp_sum    = {1'b0, a_exp} + {1'b0, b_exp};
    assign exp_unbias = exp_sum - 9'd127;

    // Mantissa multiplication (with implicit leading 1)
    // {1, mantissa_a} * {1, mantissa_b} = 24-bit x 24-bit = 48-bit
    logic [23:0] a_full_mant;
    logic [23:0] b_full_mant;
    logic [47:0] mant_product;

    assign a_full_mant  = {1'b1, a_mant};
    assign b_full_mant  = {1'b1, b_mant};
    assign mant_product = a_full_mant * b_full_mant;

    // Normalization:
    //   If mant_product[47] = 1: product is in [2.0, 4.0)
    //     -> shift right by 1 (take bits [46:24]), increment exponent
    //   If mant_product[47] = 0: product is in [1.0, 2.0)
    //     -> already normalized (take bits [45:23])
    logic mant_overflow;
    assign mant_overflow = mant_product[47];

    logic [22:0] res_mant;
    logic [7:0]  res_exp;

    assign res_mant = mant_overflow ? mant_product[46:24]
                                    : mant_product[45:23];

    assign res_exp = mant_overflow ? exp_unbias[7:0] + 8'd1
                                   : exp_unbias[7:0];

    // =========================================================
    // STEP 5: Compose final FP32 result
    // =========================================================
    assign fp32_result = either_zero ? 32'h00000000
                                     : {res_sign, res_exp, res_mant};

endmodule
