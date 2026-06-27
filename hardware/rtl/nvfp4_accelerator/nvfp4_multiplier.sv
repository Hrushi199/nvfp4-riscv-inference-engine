//===========================================================
// NVFP4 MULTIPLIER — SystemVerilog
//===========================================================
//
// Multiplies two decoded NVFP4 values (5-bit signed integers)
// to produce a 9-bit signed product.
//
// Bit-Width Justification:
//   Input range:  -12 to +12 (5-bit signed)
//   Max product:  12 x 12 = 144
//   Min product: -12 x 12 = -144
//   9-bit signed range: -256 to +255
//   => 9 bits is sufficient for all valid products.
//
// The Verilog multiply produces a 10-bit result (5+5),
// which we truncate to 9 bits. This is safe because
// the mathematical result always fits in 9 bits.
//
//===========================================================

module nvfp4_multiplier (
    input  logic signed [4:0] a,          // decoded weight  (x2 integer)
    input  logic signed [4:0] b,          // decoded activation (x2 integer)
    output logic signed [8:0] product     // 9-bit signed product
);

    // Full 10-bit multiply result
    logic signed [9:0] full_product;
    assign full_product = a * b;

    // Truncate to 9 bits (safe: max |product| = 144 < 256)
    assign product = full_product[8:0];

endmodule
