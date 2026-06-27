#======================================================================
# Cadence Genus Synthesis Script
# Design  : cv32e40p_nvfp4_top (CV32E40P + NVFP4 + BF16 Accelerator)
# Library : gpdk045 / gsclib045 (45nm)
#======================================================================

if {[file exists /proc/cpuinfo]} {
  sh grep "model name" /proc/cpuinfo
  sh grep "cpu MHz"    /proc/cpuinfo
}

puts "Hostname : [info hostname]"

#----------------------------------------------------------------------
# 1. Design and Effort Configuration
#----------------------------------------------------------------------
set DESIGN   cv32e40p_nvfp4_top
set SYN_EFF  medium
set MAP_EFF  medium
set OPT_EFF  medium

set RELEASE [lindex [get_db program_version] end]
set _OUTPUTS_PATH OUTPUT/outputs_${RELEASE}
set _REPORTS_PATH OUTPUT/reports_${RELEASE}

foreach dir [list ${_OUTPUTS_PATH} ${_REPORTS_PATH} \
             ${_OUTPUTS_PATH}/${DESIGN}/generic \
             ${_OUTPUTS_PATH}/${DESIGN}/mapped \
             ${_OUTPUTS_PATH}/${DESIGN}/opt \
             ${_REPORTS_PATH}/${DESIGN}/generic \
             ${_REPORTS_PATH}/${DESIGN}/mapped \
             ${_REPORTS_PATH}/${DESIGN}/opt] {
  if {![file exists $dir]} {
    file mkdir $dir
    puts "Creating directory $dir"
  }
}

#----------------------------------------------------------------------
# 2. Path Variables
#   *** UPDATE THESE to match your server's library installation ***
#----------------------------------------------------------------------
set baseDir    [file normalize [file dirname [info script]]]
set rtlDir     [file normalize ${baseDir}/src]
set scriptDir  [file normalize ${baseDir}]
set libDir [list \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045/timing    \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045_hvt/timing \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045_lvt/timing  \
]

set libList [list \
  fast_vdd1v0_basicCells.lib          \
  fast_vdd1v0_multibitsDFF.lib        \
  fast_vdd1v0_basicCells_hvt.lib      \
  fast_vdd1v0_basicCells_lvt.lib      \
]

#----------------------------------------------------------------------
# 3. Genus Database Settings
#----------------------------------------------------------------------
set_db init_lib_search_path   $libDir
set_db script_search_path     $scriptDir
set_db init_hdl_search_path   [list \
  ${rtlDir}/include \
  ${rtlDir}/cpu     \
  ${rtlDir}/accel   \
  ${rtlDir}/vendor  \
]

set_db max_cpus_per_server  8
set_db information_level    9
set_db tns_opto             true

#----------------------------------------------------------------------
# 4. RTL File List
#   NOTE: cv32e40p_fp_wrapper.sv is intentionally excluded.
#         It instantiates fpnew_top (not in scope — replaced by our
#         custom NVFP4+BF16 APU adapter).
#----------------------------------------------------------------------
set rtlList [list \
  ${rtlDir}/include/cv32e40p_apu_core_pkg.sv \
  ${rtlDir}/include/cv32e40p_fpu_pkg.sv      \
  ${rtlDir}/include/cv32e40p_pkg.sv          \
  ${rtlDir}/vendor/cv32e40p_sim_clock_gate.sv \
  ${rtlDir}/cpu/cv32e40p_aligner.sv          \
  ${rtlDir}/cpu/cv32e40p_alu.sv              \
  ${rtlDir}/cpu/cv32e40p_alu_div.sv          \
  ${rtlDir}/cpu/cv32e40p_apu_disp.sv         \
  ${rtlDir}/cpu/cv32e40p_compressed_decoder.sv \
  ${rtlDir}/cpu/cv32e40p_controller.sv       \
  ${rtlDir}/cpu/cv32e40p_core.sv             \
  ${rtlDir}/cpu/cv32e40p_cs_registers.sv     \
  ${rtlDir}/cpu/cv32e40p_decoder.sv          \
  ${rtlDir}/cpu/cv32e40p_ex_stage.sv         \
  ${rtlDir}/cpu/cv32e40p_ff_one.sv           \
  ${rtlDir}/cpu/cv32e40p_fifo.sv             \
  ${rtlDir}/cpu/cv32e40p_hwloop_regs.sv      \
  ${rtlDir}/cpu/cv32e40p_id_stage.sv         \
  ${rtlDir}/cpu/cv32e40p_if_stage.sv         \
  ${rtlDir}/cpu/cv32e40p_int_controller.sv   \
  ${rtlDir}/cpu/cv32e40p_load_store_unit.sv  \
  ${rtlDir}/cpu/cv32e40p_mult.sv             \
  ${rtlDir}/cpu/cv32e40p_obi_interface.sv    \
  ${rtlDir}/cpu/cv32e40p_popcnt.sv           \
  ${rtlDir}/cpu/cv32e40p_prefetch_buffer.sv  \
  ${rtlDir}/cpu/cv32e40p_prefetch_controller.sv \
  ${rtlDir}/cpu/cv32e40p_register_file_ff.sv \
  ${rtlDir}/cpu/cv32e40p_sleep_unit.sv       \
  ${rtlDir}/cpu/cv32e40p_nvfp4_top.sv        \
  ${rtlDir}/accel/bf16_mac_unit.sv           \
  ${rtlDir}/accel/nvfp4_apu_adapter.sv       \
  ${rtlDir}/accel/nvfp4_accelerator_top.sv   \
  ${rtlDir}/accel/nvfp4_decoder.sv           \
  ${rtlDir}/accel/nvfp4_dot_product_16.sv    \
  ${rtlDir}/accel/nvfp4_extractor.sv         \
  ${rtlDir}/accel/nvfp4_mac_unit.sv          \
  ${rtlDir}/accel/nvfp4_multiplier.sv        \
  ${rtlDir}/accel/nvfp4_scale_multiply.sv    \
]

#----------------------------------------------------------------------
# 5. Load Libraries and Physical LEF Files
#----------------------------------------------------------------------
puts "Loading technology libraries..."
read_libs $libList

read_physical -lefs [list \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045/lef/gsclib045_tech.lef \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045/lef/gsclib045_macro.lef \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045/lef/gsclib045_multibitsDFF.lef \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045_hvt/lef/gsclib045_hvt_macro.lef \
  /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045_lvt/lef/gsclib045_lvt_macro.lef \
]

set_db qrc_tech_file /home/redhatacademy19/Documents/Siddhant/gsclib045_all_v4.8/gsclib045_tech/qrc/qx/gpdk045.tch

#----------------------------------------------------------------------
# 6. Read HDL and Elaborate
#----------------------------------------------------------------------
puts "Reading HDL files..."
read_hdl -sv $rtlList

puts "Elaborating top: ${DESIGN}..."
elaborate $DESIGN
set_top_module $DESIGN

check_design -unresolved

#----------------------------------------------------------------------
# 7. Timing Constraints
#----------------------------------------------------------------------
puts "Reading timing constraints..."
read_sdc ${scriptDir}/constraints.sdc

#----------------------------------------------------------------------
# 8. Synthesis Flow
#----------------------------------------------------------------------

# ---- Generic Synthesis ----
syn_generic

write_snapshot -directory ${_OUTPUTS_PATH}/${DESIGN}/generic -tag generic
report_summary -directory ${_REPORTS_PATH}/${DESIGN}/generic
report_power   > ${_REPORTS_PATH}/${DESIGN}/generic/power.rpt
puts "Runtime & Memory after 'syn_generic'"
time_info GENERIC

# ---- Technology Mapping ----
syn_map

write_snapshot -directory ${_OUTPUTS_PATH}/${DESIGN}/mapped  -tag mapped
report_summary -directory ${_REPORTS_PATH}/${DESIGN}/mapped
report_power   > ${_REPORTS_PATH}/${DESIGN}/mapped/power.rpt
puts "Runtime & Memory after 'syn_map'"
time_info MAPPED

# ---- Optimization ----
syn_opt

write_snapshot -innovus -directory ${_OUTPUTS_PATH}/${DESIGN}/opt -tag opt
report_summary -directory ${_REPORTS_PATH}/${DESIGN}/opt
report_power   > ${_REPORTS_PATH}/${DESIGN}/opt/power.rpt
puts "Runtime & Memory after 'syn_opt'"
time_info OPT

#----------------------------------------------------------------------
# 9. Final Outputs
#----------------------------------------------------------------------
# Gate-level Verilog netlist
write_hdl  > ${_OUTPUTS_PATH}/${DESIGN}/${DESIGN}_netlist.v

# Synthesized SDC for place-and-route
write_sdc  > ${_OUTPUTS_PATH}/${DESIGN}/${DESIGN}_synth.sdc

# Design database (for reuse / incremental runs)
write_db -to_file ${DESIGN}.db

puts "Final Runtime & Memory."
time_info FINAL

puts "=============================="
puts " Synthesis Finished."
puts " Netlist : ${_OUTPUTS_PATH}/${DESIGN}/${DESIGN}_netlist.v"
puts " Reports : ${_REPORTS_PATH}/${DESIGN}/"
puts "=============================="

#quit
