//===========================================================
// NVFP4 DOT-PRODUCT ACCELERATOR — TOP LEVEL (SystemVerilog)
//===========================================================
//
// Top-level module integrating the complete NVFP4 dot-product
// pipeline with a simple 2-state FSM for control.
//
// This accelerator is designed to sit alongside a RISC-V core
// as a custom co-processor. The core dispatches NVFP4 block
// operations to this unit via memory-mapped registers or
// custom ISA instructions (Week 6).
//
// Pipeline (2-cycle latency):
//   Cycle 0 (IDLE):    start=1, inputs latched into registers
//   Cycle 1 (COMPUTE): Combinational: extract -> decode ->
//                       multiply -> adder tree -> int-to-float
//                       -> FP32 scale multiply.
//                       Results registered, done asserted.
//   Cycle 2 (IDLE):    done=1, fp32_result valid.
//
// Interface:
//   - Simple handshake: assert start for 1 cycle, wait for done
//   - The RISC-V core pre-computes combined_scale =
//     (scale_w * scale_a) / 4.0 before dispatching
//
// Module Hierarchy:
//   nvfp4_accelerator_top
//     |-- nvfp4_dot_product_16
//     |     |-- nvfp4_mac_unit [0:15]
//     |           |-- nvfp4_decoder (weight)
//     |           |-- nvfp4_decoder (activation)
//     |           |-- nvfp4_multiplier
//     |-- nvfp4_scale_multiply
//
//===========================================================

module nvfp4_accelerator_top (
    input  logic        clk,              // system clock
    input  logic        rst_n,            // active-low synchronous reset

    // Control interface
    input  logic        start,            // pulse high for 1 cycle to begin
    output logic        done,             // pulses high for 1 cycle when result is valid
    output logic        busy,             // high while accelerator is computing

    // Data inputs (active when start=1)
    input  logic [63:0] weight_packed,    // 16 packed NVFP4 weight codes
    input  logic [63:0] act_packed,       // 16 packed NVFP4 activation codes
    input  logic [31:0] combined_scale,   // FP32: (scale_w * scale_a) / 4.0

    // Data output (valid when done=1)
    output logic [31:0] fp32_result,      // FP32 dot product result

    // Debug outputs (optional, for verification)
    output logic signed [12:0] dbg_dot_int  // raw integer dot product
);

    // =========================================================
    // FSM States
    // =========================================================
    typedef enum logic {
        STATE_IDLE    = 1'b0,
        STATE_COMPUTE = 1'b1
    } state_t;

    state_t state;

    // =========================================================
    // Input Registers (latched on start)
    // =========================================================
    logic [63:0] weight_reg;
    logic [63:0] act_reg;
    logic [31:0] scale_reg;

    // =========================================================
    // Combinational Datapath
    // =========================================================

    // Integer dot product from 16 parallel MACs + adder tree
    logic signed [12:0] dot_int_wire;

    nvfp4_dot_product_16 u_dot_product (
        .weight_packed (weight_reg),
        .act_packed    (act_reg),
        .dot_result    (dot_int_wire)
    );

    // FP32 scale multiplication
    logic [31:0] scaled_result_wire;

    nvfp4_scale_multiply u_scale_multiply (
        .dot_int        (dot_int_wire),
        .combined_scale (scale_reg),
        .fp32_result    (scaled_result_wire)
    );

    // Debug output
    assign dbg_dot_int = dot_int_wire;

    // Busy signal
    assign busy = (state != STATE_IDLE);

    // =========================================================
    // FSM + Pipeline Control
    // =========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // ---- Reset ----
            state       <= STATE_IDLE;
            done        <= 1'b0;
            fp32_result <= 32'd0;
            weight_reg  <= 64'd0;
            act_reg     <= 64'd0;
            scale_reg   <= 32'd0;
        end else begin
            // Default: deassert done each cycle
            done <= 1'b0;

            case (state)
                // ---- IDLE: Wait for start pulse ----
                STATE_IDLE: begin
                    if (start) begin
                        // Latch inputs into pipeline registers
                        weight_reg <= weight_packed;
                        act_reg    <= act_packed;
                        scale_reg  <= combined_scale;
                        state      <= STATE_COMPUTE;
                    end
                end

                // ---- COMPUTE: Combinational logic runs, register result ----
                STATE_COMPUTE: begin
                    // The dot product + scale multiply combinational
                    // logic has had one full cycle to settle from
                    // the registered inputs. Capture the result.
                    fp32_result <= scaled_result_wire;
                    done        <= 1'b1;
                    state       <= STATE_IDLE;
                end

                default: begin
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

endmodule
