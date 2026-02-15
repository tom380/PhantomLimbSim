# Central simulation constants.

# Parameters
T_CYCLE = 2.0
TOTAL_SIM_TIME = 2 * T_CYCLE

# Force model constants
LENGTH_FEMUR    = 0.29869           # m
LENGTH_TIBIA    = 0.30433           # m
GRAVITY         = 9.81              # m/s²
# Active anthropometric/inertial set (with exosuit mass contribution).
COM_FEMUR       = 0.1523                # m
COM_TIBIA       = 0.1132                # m
MASS_FEMUR      = 2.50513539 + 0.635    # kg
MASS_TIBIA      = 1.22768423 + 0.388    # kg
INERTIA_FEMUR   = 0.0165                # kg·m²
INERTIA_TIBIA   = 0.0130                # kg·m²
