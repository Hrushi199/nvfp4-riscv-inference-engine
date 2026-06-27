//===========================================================
// NVFP4 SoC TOP (SystemVerilog)
//===========================================================
//
// Complete SoC:
//   CV32E40P (with NVFP4 APU) + Instruction RAM + Data RAM
//
// Memory Map:
//   0x0000_0000 : Instruction RAM (64KB)
//   0x2000_0000 : Data RAM (64KB)
//
//===========================================================

module nvfp4_soc_top #(
    parameter INSTR_RAM_DEPTH = 16384,
    parameter DATA_RAM_DEPTH  = 16384
) (
    input logic clk_i,
    input logic rst_ni
);

    // Instruction bus
    logic        instr_req;
    logic [31:0] instr_addr;
    logic        instr_rvalid;
    logic [31:0] instr_rdata;

    // Data bus
    logic        data_req;
    logic [31:0] data_addr;
    logic        data_we;
    logic [3:0]  data_be;
    logic [31:0] data_wdata;
    logic        data_rvalid;
    logic [31:0] data_rdata;

    // =========================================================
    // CPU: CV32E40P + NVFP4 Accelerator
    // =========================================================
    cv32e40p_nvfp4_top #(
        .COREV_PULP       (0),
        .COREV_CLUSTER    (0),
        .NUM_MHPMCOUNTERS (1)
    ) u_cpu (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),
        .pulp_clock_en_i    (1'b1),
        .scan_cg_en_i       (1'b0),
        .boot_addr_i        (32'h0000_0000),
        .mtvec_addr_i       (32'h0000_0001),
        .dm_halt_addr_i     (32'h1A11_0800),
        .hart_id_i          (32'd0),
        .dm_exception_addr_i(32'h1A11_0808),

        .instr_req_o   (instr_req),
        .instr_gnt_i   (1'b1),
        .instr_rvalid_i(instr_rvalid),
        .instr_addr_o  (instr_addr),
        .instr_rdata_i (instr_rdata),

        .data_req_o   (data_req),
        .data_gnt_i   (1'b1),
        .data_rvalid_i(data_rvalid),
        .data_we_o    (data_we),
        .data_be_o    (data_be),
        .data_addr_o  (data_addr),
        .data_wdata_o (data_wdata),
        .data_rdata_i (data_rdata),

        .irq_i              (32'd0),
        .irq_ack_o          (),
        .irq_id_o           (),
        .debug_req_i        (1'b0),
        .debug_havereset_o  (),
        .debug_running_o    (),
        .debug_halted_o     (),
        .fetch_enable_i     (1'b1),
        .core_sleep_o       ()
    );

    // =========================================================
    // Instruction RAM
    // =========================================================
    logic [31:0] instr_mem [0:INSTR_RAM_DEPTH-1];

    initial $readmemh("test_nvfp4.hex", instr_mem);

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            instr_rvalid <= 1'b0;
            instr_rdata  <= 32'd0;
        end else begin
            instr_rvalid <= instr_req;
            instr_rdata  <= instr_mem[instr_addr[15:2]];
        end
    end

    // =========================================================
    // Data RAM
    // =========================================================
    logic [31:0] data_mem [0:DATA_RAM_DEPTH-1];

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            data_rvalid <= 1'b0;
            data_rdata  <= 32'd0;
        end else begin
            data_rvalid <= data_req;
            if (data_req) begin
                if (data_we) begin
                    if (data_be[0]) data_mem[data_addr[15:2]][7:0]   <= data_wdata[7:0];
                    if (data_be[1]) data_mem[data_addr[15:2]][15:8]  <= data_wdata[15:8];
                    if (data_be[2]) data_mem[data_addr[15:2]][23:16] <= data_wdata[23:16];
                    if (data_be[3]) data_mem[data_addr[15:2]][31:24] <= data_wdata[31:24];
                end
                data_rdata <= data_mem[data_addr[15:2]];
            end
        end
    end

endmodule
