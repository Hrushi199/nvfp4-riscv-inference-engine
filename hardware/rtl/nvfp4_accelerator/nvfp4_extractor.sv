//===========================================================
// NVFP4 PACKED DATA EXTRACTOR — SystemVerilog
//===========================================================
//
// Extracts a single 4-bit NVFP4 code from a 64-bit packed
// data word using an index.
//
// Memory Layout (matching Python nvfp4_utils.py):
//   A 64-bit word contains 16 packed NVFP4 values.
//   Element i occupies bits [i*4+3 : i*4].
//
//   Byte 0: [elem1][elem0]  (upper nibble = elem1)
//   Byte 1: [elem3][elem2]
//   ...
//   Byte 7: [elem15][elem14]
//
// This module provides a clean abstraction for element
// extraction and can be reused in the RISC-V ISA extension
// (Week 6) for NVFP4.LOAD instruction implementation.
//
//===========================================================

module nvfp4_extractor (
    input  logic [63:0] packed_data,   // 16 packed NVFP4 codes
    input  logic [3:0]  index,         // element index (0-15)
    output logic [3:0]  code_out       // extracted 4-bit NVFP4 code
);

    // Indexed part-select: extract 4 bits starting at index*4
    assign code_out = packed_data[index * 4 +: 4];

endmodule
