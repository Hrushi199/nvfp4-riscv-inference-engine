//===========================================================
// MIXED-PRECISION APU ADAPTER (SystemVerilog)
//===========================================================
//
// Bridges the CV32E40P APU interface to two accelerators:
//   1. NVFP4 dot-product accelerator (4-bit float)
//   2. BF16  dot-product unit        (bfloat16)
//
// APU Protocol:
//   1. Core asserts apu_req with operands + op code
//   2. Adapter asserts apu_gnt (accepts request)
//   3. Adapter routes to selected accelerator
//   4. Adapter asserts apu_rvalid with result
//
// Operation Encoding (apu_op[5:0]):
//   ---- NVFP4 instructions ----
//   6'b000001 = NVFP4.LOAD_W : Load weights  (operands[0:1])
//   6'b000010 = NVFP4.LOAD_A : Load acts     (operands[0:1])
//   6'b000100 = NVFP4.MAC    : Scale+compute (operands[0])
//   6'b001000 = NVFP4.STORE  : Return result
//   ---- BF16 instructions ----
//   6'b010001 = BF16.LOAD_W  : Load BF16 weights  (operands[0:1])
//   6'b010010 = BF16.LOAD_A  : Load BF16 acts     (operands[0:1])
//   6'b010100 = BF16.MAC     : Compute & return result
//
// Operand Mapping (shared between NVFP4 and BF16):
//   LOAD_W: operands[0] = packed[31:0], operands[1] = packed[63:32]
//   LOAD_A: operands[0] = packed[31:0], operands[1] = packed[63:32]
//   MAC:    operands[0] = scale (NVFP4) or unused (BF16)
//   STORE:  no operands needed, returns stored result
//
//===========================================================

module nvfp4_apu_adapter
  import cv32e40p_apu_core_pkg::*;
(
    input  logic                              clk,
    input  logic                              rst_n,

    // --- APU Interface (from CV32E40P core) ---
    input  logic                              apu_req_i,
    output logic                              apu_gnt_o,
    input  logic [   APU_NARGS_CPU-1:0][31:0] apu_operands_i,
    input  logic [     APU_WOP_CPU-1:0]       apu_op_i,
    input  logic [APU_NDSFLAGS_CPU-1:0]       apu_flags_i,
    output logic                              apu_rvalid_o,
    output logic [                31:0]       apu_result_o,
    output logic [APU_NUSFLAGS_CPU-1:0]       apu_rflags_o
);

    // =========================================================
    // NVFP4 Operation Codes
    // =========================================================
    localparam OP_NVFP4_LOAD_W = 6'b000001;
    localparam OP_NVFP4_LOAD_A = 6'b000010;
    localparam OP_NVFP4_MAC    = 6'b000100;
    localparam OP_NVFP4_STORE  = 6'b001000;

    // =========================================================
    // BF16 Operation Codes
    // =========================================================
    localparam OP_BF16_LOAD_W  = 6'b010001;
    localparam OP_BF16_LOAD_A  = 6'b010010;
    localparam OP_BF16_MAC     = 6'b010100;

    // =========================================================
    // Register Banks (separate for NVFP4 and BF16)
    // =========================================================
    logic [63:0] nvfp4_weight_reg;
    logic [63:0] nvfp4_act_reg;
    logic [31:0] nvfp4_scale_reg;
    logic [31:0] nvfp4_result_reg;

    logic [63:0] bf16_weight_reg;
    logic [63:0] bf16_act_reg;
    logic [31:0] bf16_result_reg;

    // =========================================================
    // NVFP4 Accelerator
    // =========================================================
    logic        nvfp4_start;
    logic        nvfp4_done;
    logic        nvfp4_busy;
    logic [31:0] nvfp4_result;
    logic signed [12:0] nvfp4_dbg;

    nvfp4_accelerator_top u_nvfp4 (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (nvfp4_start),
        .done           (nvfp4_done),
        .busy           (nvfp4_busy),
        .weight_packed  (nvfp4_weight_reg),
        .act_packed     (nvfp4_act_reg),
        .combined_scale (nvfp4_scale_reg),
        .fp32_result    (nvfp4_result),
        .dbg_dot_int    (nvfp4_dbg)
    );

    // =========================================================
    // BF16 MAC Unit
    // =========================================================
    logic        bf16_start;
    logic        bf16_done;
    logic [31:0] bf16_result;

    bf16_mac_unit u_bf16 (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (bf16_start),
        .weight_bf16 (bf16_weight_reg),
        .act_bf16    (bf16_act_reg),
        .done        (bf16_done),
        .result_fp32 (bf16_result)
    );

    // =========================================================
    // FSM
    // =========================================================
    typedef enum logic [1:0] {
        S_IDLE    = 2'b00,
        S_COMPUTE = 2'b01,
        S_DONE    = 2'b10
    } state_t;

    state_t state;

    // Grant immediately when idle
    assign apu_gnt_o    = apu_req_i && (state == S_IDLE);
    assign apu_rflags_o = '0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            nvfp4_weight_reg  <= 64'd0;
            nvfp4_act_reg     <= 64'd0;
            nvfp4_scale_reg   <= 32'd0;
            nvfp4_result_reg  <= 32'd0;
            bf16_weight_reg   <= 64'd0;
            bf16_act_reg      <= 64'd0;
            bf16_result_reg   <= 32'd0;
            nvfp4_start       <= 1'b0;
            bf16_start        <= 1'b0;
            apu_rvalid_o      <= 1'b0;
            apu_result_o      <= 32'd0;
        end else begin
            // Defaults
            nvfp4_start  <= 1'b0;
            bf16_start   <= 1'b0;
            apu_rvalid_o <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (apu_req_i) begin
                        case (apu_op_i)

                            // -------- NVFP4 Instructions --------
                            OP_NVFP4_LOAD_W: begin
                                nvfp4_weight_reg <= {apu_operands_i[1], apu_operands_i[0]};
                                apu_rvalid_o     <= 1'b1;
                                apu_result_o     <= 32'd0;
                            end

                            OP_NVFP4_LOAD_A: begin
                                nvfp4_act_reg <= {apu_operands_i[1], apu_operands_i[0]};
                                apu_rvalid_o  <= 1'b1;
                                apu_result_o  <= 32'd0;
                            end

                            OP_NVFP4_MAC: begin
                                nvfp4_scale_reg <= apu_operands_i[0];
                                nvfp4_start     <= 1'b1;
                                state           <= S_COMPUTE;
                            end

                            OP_NVFP4_STORE: begin
                                apu_rvalid_o <= 1'b1;
                                apu_result_o <= nvfp4_result_reg;
                            end

                            // -------- BF16 Instructions --------
                            OP_BF16_LOAD_W: begin
                                bf16_weight_reg <= {apu_operands_i[1], apu_operands_i[0]};
                                apu_rvalid_o    <= 1'b1;
                                apu_result_o    <= 32'd0;
                            end

                            OP_BF16_LOAD_A: begin
                                bf16_act_reg <= {apu_operands_i[1], apu_operands_i[0]};
                                apu_rvalid_o <= 1'b1;
                                apu_result_o <= 32'd0;
                            end

                            OP_BF16_MAC: begin
                                // BF16 unit computes and returns result in 1 cycle
                                bf16_start <= 1'b1;
                                state      <= S_COMPUTE;
                            end

                            default: begin
                                apu_rvalid_o <= 1'b1;
                                apu_result_o <= 32'd0;
                            end
                        endcase
                    end
                end

                S_COMPUTE: begin
                    // Wait for whichever unit was started
                    if (nvfp4_done) begin
                        nvfp4_result_reg <= nvfp4_result;
                        apu_rvalid_o     <= 1'b1;
                        apu_result_o     <= nvfp4_result;
                        state            <= S_IDLE;
                    end else if (bf16_done) begin
                        bf16_result_reg <= bf16_result;
                        apu_rvalid_o    <= 1'b1;
                        apu_result_o    <= bf16_result;
                        state           <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
