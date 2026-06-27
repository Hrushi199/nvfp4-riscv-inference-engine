//===========================================================
// NVFP4 DECODER (E2M1 Format) — SystemVerilog
//===========================================================
//
// Decodes a 4-bit NVFP4 code (1 sign + 2 exponent + 1 mantissa)
// into a 5-bit signed two's complement integer.
//
// KEY DESIGN DECISION:
//   The decoded value is the actual E2M1 float value MULTIPLIED
//   BY 2 to eliminate fractional bits in hardware. All downstream
//   modules work with these x2 integers. The /4 correction
//   (since both operands are x2) is applied once at the output
//   stage in the scale_multiply module.
//
// E2M1 Lookup Table:
//   Code[2:0]  |  Float Value  |  x2 Integer
//   -----------|---------------|------------
//      000     |     0.0       |      0
//      001     |     0.5       |      1
//      010     |     1.0       |      2
//      011     |     1.5       |      3
//      100     |     2.0       |      4
//      101     |     3.0       |      6
//      110     |     4.0       |      8
//      111     |     6.0       |     12
//
// Bit[3] = sign: 0 = positive, 1 = negative
//
// This directly mirrors the Python fp4_table in nvfp4_utils.py.
//
//===========================================================

module nvfp4_decoder (
    input  logic [3:0]         nvfp4_in,      // 4-bit NVFP4 code
    output logic signed [4:0]  decoded_out    // 5-bit signed integer (value x 2)
);

    // -------------------------------------------------------
    // Magnitude LUT: 3-bit E2M1 code -> unsigned x2 integer
    // -------------------------------------------------------
    logic [3:0] magnitude;  // 4-bit unsigned (max = 12)

    always_comb begin
        case (nvfp4_in[2:0])
            3'd0:    magnitude = 4'd0;     // 0.0 x 2 =  0
            3'd1:    magnitude = 4'd1;     // 0.5 x 2 =  1
            3'd2:    magnitude = 4'd2;     // 1.0 x 2 =  2
            3'd3:    magnitude = 4'd3;     // 1.5 x 2 =  3
            3'd4:    magnitude = 4'd4;     // 2.0 x 2 =  4
            3'd5:    magnitude = 4'd6;     // 3.0 x 2 =  6
            3'd6:    magnitude = 4'd8;     // 4.0 x 2 =  8
            3'd7:    magnitude = 4'd12;    // 6.0 x 2 = 12
            default: magnitude = 4'd0;
        endcase
    end

    // -------------------------------------------------------
    // Sign application
    // -------------------------------------------------------
    // Zero-extend magnitude to 5-bit signed (always positive)
    logic signed [4:0] pos_value;
    assign pos_value = {1'b0, magnitude};

    // If sign bit is set, negate via two's complement
    // Note: -0 = 0 in two's complement, so negative zero
    //       correctly maps to 0.
    assign decoded_out = nvfp4_in[3] ? (-pos_value) : pos_value;

endmodule
