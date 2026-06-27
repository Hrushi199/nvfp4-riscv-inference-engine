//===========================================================
// NVFP4 MAC (Multiply-Accumulate) UNIT — SystemVerilog
//===========================================================
//
// One complete MAC element for the NVFP4 dot-product engine.
// Takes two raw 4-bit NVFP4 codes (weight + activation),
// decodes both to x2 integers, and multiplies them.
//
// Datapath:
//   weight_code [3:0] --> [nvfp4_decoder] --> w_decoded [4:0]
//                                                          \
//                                                    [nvfp4_multiplier] --> product [8:0]
//                                                          /
//   act_code    [3:0] --> [nvfp4_decoder] --> a_decoded [4:0]
//
// This module is instantiated 16 times in the dot_product_16
// engine (one per micro-block element).
//
//===========================================================

module nvfp4_mac_unit (
    input  logic [3:0]         weight_code,   // raw 4-bit NVFP4 weight code
    input  logic [3:0]         act_code,      // raw 4-bit NVFP4 activation code
    output logic signed [8:0]  product        // decoded_w * decoded_a (x4 integer)
);

    // -------------------------------------------------------
    // Decode both operands
    // -------------------------------------------------------
    logic signed [4:0] w_decoded;
    logic signed [4:0] a_decoded;

    nvfp4_decoder u_dec_w (
        .nvfp4_in    (weight_code),
        .decoded_out (w_decoded)
    );

    nvfp4_decoder u_dec_a (
        .nvfp4_in    (act_code),
        .decoded_out (a_decoded)
    );

    // -------------------------------------------------------
    // Multiply decoded values
    // -------------------------------------------------------
    nvfp4_multiplier u_mul (
        .a       (w_decoded),
        .b       (a_decoded),
        .product (product)
    );

endmodule
