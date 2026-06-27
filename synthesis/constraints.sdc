#======================================================================
# SDC Constraints File (constraints.sdc)
#======================================================================
# Target clock frequency: 200 MHz (Clock period = 5.0 ns)
# Adjust the clock period value depending on target library capability.
#======================================================================

# 1. Define Clock
# Define a clock signal named "clk" on port "clk_i" with a period of 5.0 ns
create_clock -name clk -period 5.0 [get_ports clk_i]

# 2. Input/Output Delays
# Set basic input and output delays to ensure timing margin at boundaries
# Usually set to 20-30% of the clock period
set_input_delay  -clock clk 1.0 [all_inputs]
set_output_delay -clock clk 1.0 [all_outputs]

# Remove delays from clock port itself to prevent warning
remove_input_delay [get_ports clk_i]

# 3. Environment Constraints (Driver and Load)
# Standard modeling constraints for input drive strength and output load capacity.
# If these library cells don't exist in your PDK, Genus will fallback to default models.
# (Adjust buffer name to match your PDK standard cell naming, e.g. BUFX2, INVX1)
# set_driving_cell -lib_cell BUFX2 [all_inputs]
# set_load 0.05 [all_outputs]
