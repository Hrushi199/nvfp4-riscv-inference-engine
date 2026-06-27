//===========================================================
// CV32E40P + NVFP4 ACCELERATOR TOP (SystemVerilog)
//===========================================================
//
// Modified cv32e40p_top that replaces the FPU with the
// NVFP4 dot-product accelerator via the APU interface.
//
// Key change from original cv32e40p_top.sv:
//   - FPU parameter set to 1 (enables APU dispatch in decoder)
//   - FPU wrapper replaced with nvfp4_apu_adapter
//   - All FP instructions routed to NVFP4 accelerator
//
//===========================================================

module cv32e40p_nvfp4_top
  import cv32e40p_apu_core_pkg::*;
#(
    parameter COREV_PULP      = 0,
    parameter COREV_CLUSTER   = 0,
    parameter NUM_MHPMCOUNTERS = 1
) (
    // Clock and Reset
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        pulp_clock_en_i,
    input  logic        scan_cg_en_i,

    // Boot / Hart config
    input  logic [31:0] boot_addr_i,
    input  logic [31:0] mtvec_addr_i,
    input  logic [31:0] dm_halt_addr_i,
    input  logic [31:0] hart_id_i,
    input  logic [31:0] dm_exception_addr_i,

    // Instruction memory interface
    output logic        instr_req_o,
    input  logic        instr_gnt_i,
    input  logic        instr_rvalid_i,
    output logic [31:0] instr_addr_o,
    input  logic [31:0] instr_rdata_i,

    // Data memory interface
    output logic        data_req_o,
    input  logic        data_gnt_i,
    input  logic        data_rvalid_i,
    output logic        data_we_o,
    output logic [ 3:0] data_be_o,
    output logic [31:0] data_addr_o,
    output logic [31:0] data_wdata_o,
    input  logic [31:0] data_rdata_i,

    // Interrupts
    input  logic [31:0] irq_i,
    output logic        irq_ack_o,
    output logic [ 4:0] irq_id_o,

    // Debug
    input  logic        debug_req_i,
    output logic        debug_havereset_o,
    output logic        debug_running_o,
    output logic        debug_halted_o,

    // CPU Control
    input  logic        fetch_enable_i,
    output logic        core_sleep_o
);

    // =========================================================
    // APU signals between core and NVFP4 adapter
    // =========================================================
    logic                              apu_busy;
    logic                              apu_req;
    logic [   APU_NARGS_CPU-1:0][31:0] apu_operands;
    logic [     APU_WOP_CPU-1:0]       apu_op;
    logic [APU_NDSFLAGS_CPU-1:0]       apu_flags;

    logic                              apu_gnt;
    logic                              apu_rvalid;
    logic [                31:0]       apu_rdata;
    logic [APU_NUSFLAGS_CPU-1:0]       apu_rflags;

    // =========================================================
    // CV32E40P Core (FPU=1 to enable APU dispatch)
    // =========================================================
    cv32e40p_core #(
        .COREV_PULP      (COREV_PULP),
        .COREV_CLUSTER   (COREV_CLUSTER),
        .FPU             (1),             // Enable APU interface
        .FPU_ADDMUL_LAT  (0),
        .FPU_OTHERS_LAT  (0),
        .ZFINX           (1),             // Use integer registers for FP (no F-regs)
        .NUM_MHPMCOUNTERS(NUM_MHPMCOUNTERS)
    ) core_i (
        .clk_i (clk_i),
        .rst_ni(rst_ni),

        .pulp_clock_en_i(pulp_clock_en_i),
        .scan_cg_en_i   (scan_cg_en_i),

        .boot_addr_i        (boot_addr_i),
        .mtvec_addr_i       (mtvec_addr_i),
        .dm_halt_addr_i     (dm_halt_addr_i),
        .hart_id_i          (hart_id_i),
        .dm_exception_addr_i(dm_exception_addr_i),

        .instr_req_o   (instr_req_o),
        .instr_gnt_i   (instr_gnt_i),
        .instr_rvalid_i(instr_rvalid_i),
        .instr_addr_o  (instr_addr_o),
        .instr_rdata_i (instr_rdata_i),

        .data_req_o   (data_req_o),
        .data_gnt_i   (data_gnt_i),
        .data_rvalid_i(data_rvalid_i),
        .data_we_o    (data_we_o),
        .data_be_o    (data_be_o),
        .data_addr_o  (data_addr_o),
        .data_wdata_o (data_wdata_o),
        .data_rdata_i (data_rdata_i),

        // APU interface → NVFP4 adapter (instead of FPU)
        .apu_busy_o    (apu_busy),
        .apu_req_o     (apu_req),
        .apu_gnt_i     (apu_gnt),
        .apu_operands_o(apu_operands),
        .apu_op_o      (apu_op),
        .apu_flags_o   (apu_flags),
        .apu_rvalid_i  (apu_rvalid),
        .apu_result_i  (apu_rdata),
        .apu_flags_i   (apu_rflags),

        .irq_i    (irq_i),
        .irq_ack_o(irq_ack_o),
        .irq_id_o (irq_id_o),

        .debug_req_i      (debug_req_i),
        .debug_havereset_o(debug_havereset_o),
        .debug_running_o  (debug_running_o),
        .debug_halted_o   (debug_halted_o),

        .fetch_enable_i(fetch_enable_i),
        .core_sleep_o  (core_sleep_o)
    );

    // =========================================================
    // NVFP4 APU Adapter (replaces FPU wrapper)
    // =========================================================
    nvfp4_apu_adapter u_nvfp4_apu (
        .clk            (clk_i),
        .rst_n          (rst_ni),
        .apu_req_i      (apu_req),
        .apu_gnt_o      (apu_gnt),
        .apu_operands_i (apu_operands),
        .apu_op_i       (apu_op),
        .apu_flags_i    (apu_flags),
        .apu_rvalid_o   (apu_rvalid),
        .apu_result_o   (apu_rdata),
        .apu_rflags_o   (apu_rflags)
    );

endmodule
