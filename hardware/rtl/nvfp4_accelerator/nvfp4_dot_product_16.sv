//===========================================================
// NVFP4 16-ELEMENT DOT PRODUCT ENGINE — SystemVerilog
//===========================================================
//
// The core compute engine of the NVFP4 accelerator.
// Processes one 16-element micro-block (matching NVIDIA's
// block size used in the Python scheduler).
//
// Architecture:
//   16 parallel MAC units (decode + multiply) feeding into
//   a 4-level binary adder tree for minimum-latency reduction.
//
// Datapath (fully combinational, single-cycle):
//
//   weight_packed [63:0] --+-- [MAC 0]  --> p[0]  --\
//   act_packed    [63:0] --+-- [MAC 1]  --> p[1]  ---+-> s1[0] --\
//                          +-- [MAC 2]  --> p[2]  --\             |
//                          +-- [MAC 3]  --> p[3]  ---+-> s1[1] --+-> s2[0] --\
//                          +-- ...                                            |
//                          +-- [MAC 14] --> p[14] --\                         |
//                          +-- [MAC 15] --> p[15] --+-> s1[7] --+-> ... --> dot_result
//
// Bit-Width Progression:
//   Products:  9-bit signed  (max |val| = 144)
//   Level 1:  10-bit signed  (max |val| = 288)
//   Level 2:  11-bit signed  (max |val| = 576)
//   Level 3:  12-bit signed  (max |val| = 1152)
//   Level 4:  13-bit signed  (max |val| = 2304)
//
// The output is an INTEGER representing the dot product
// of decoded values x4 (since both operands were x2).
// The /4 correction is applied later in the scale module.
//
//===========================================================

module nvfp4_dot_product_16 (
    input  logic [63:0]         weight_packed,   // 16 packed NVFP4 weight codes
    input  logic [63:0]         act_packed,      // 16 packed NVFP4 activation codes
    output logic signed [12:0]  dot_result       // 13-bit signed integer dot product (x4)
);

    // =========================================================
    // STAGE 1: 16 Parallel MAC Units
    // =========================================================
    // Each MAC unit extracts its element via inline bit-slicing,
    // decodes both NVFP4 codes, and multiplies them.

    logic signed [8:0] p [16];   // 16 products

    for (genvar i = 0; i < 16; i++) begin : mac_array
        nvfp4_mac_unit u_mac (
            .weight_code (weight_packed[i*4 +: 4]),
            .act_code    (act_packed[i*4 +: 4]),
            .product     (p[i])
        );
    end

    // =========================================================
    // STAGE 2: 4-Level Binary Adder Tree
    // =========================================================
    // Each level sign-extends operands by 1 bit before addition
    // to prevent overflow.

    // ---- Level 1: 16 -> 8 sums (9-bit -> 10-bit) ----
    logic signed [9:0] s1 [8];

    for (genvar i = 0; i < 8; i++) begin : tree_l1
        assign s1[i] = {p[2*i][8], p[2*i]}
                     + {p[2*i+1][8], p[2*i+1]};
    end

    // ---- Level 2: 8 -> 4 sums (10-bit -> 11-bit) ----
    logic signed [10:0] s2 [4];

    for (genvar i = 0; i < 4; i++) begin : tree_l2
        assign s2[i] = {s1[2*i][9], s1[2*i]}
                     + {s1[2*i+1][9], s1[2*i+1]};
    end

    // ---- Level 3: 4 -> 2 sums (11-bit -> 12-bit) ----
    logic signed [11:0] s3 [2];

    for (genvar i = 0; i < 2; i++) begin : tree_l3
        assign s3[i] = {s2[2*i][10], s2[2*i]}
                     + {s2[2*i+1][10], s2[2*i+1]};
    end

    // ---- Level 4: 2 -> 1 final sum (12-bit -> 13-bit) ----
    assign dot_result = {s3[0][11], s3[0]}
                      + {s3[1][11], s3[1]};

endmodule
