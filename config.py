
NON_LAGGED_SUMMARY_BASES = [
    'distance_sum', 
    'energy_active_sum', 
    'flights_climbed_sum', 
    'energy_basal_mean', 
    'audio_level_std', 
    'walking_symmetry_mean'
]


CORE_FEATURE_BASES = [
    'steps_sum', 
    'sleep_duration_sum', 
    'audio_level_mean', 
    'hrv_mean', 
    'bp_systolic_mean', 
    'bp_diastolic_mean', 
    'sentiment_compound_mean'
]

FINAL_MODEL_FEATURES = []
FINAL_MODEL_FEATURES.extend(NON_LAGGED_SUMMARY_BASES) 

for base in CORE_FEATURE_BASES:
    FINAL_MODEL_FEATURES.append(base) 
    for lag in range(1, 4):
        FINAL_MODEL_FEATURES.append(f'{base}_lag_{lag}')
        
USER_PROFILES = {
    "user1_data_baseline": {
        "description": "User 1: Real Data Baseline", "hrv_mu": 55, "hrv_sigma": 8, 
        "bp_sys_mu": 125, "bp_sys_sigma": 5, "sleep_mu": 27000, "sleep_sigma": 2500, 
        "steps_mu": 7500, "steps_sigma": 1800, 
    },
    "user2_balanced": {
        "description": "User 2: Standard/Healthy", "hrv_mu": 60, "hrv_sigma": 10,
        "bp_sys_mu": 120, "bp_sys_sigma": 4, "sleep_mu": 28800, "sleep_sigma": 3000, 
        "steps_mu": 9000, "steps_sigma": 2000, 
    },
    "user3_chronic_stress": {
        "description": "User 3: Chronic Stress/Low Sleep", "hrv_mu": 40, "hrv_sigma": 5,  
        "bp_sys_mu": 140, "bp_sys_sigma": 7, "sleep_mu": 21600, "sleep_sigma": 1800, 
        "steps_mu": 5000, "steps_sigma": 1500, 
    },
    "user4_athlete": {
        "description": "User 4: Athlete/High HRV", "hrv_mu": 75, "hrv_sigma": 12, 
        "bp_sys_mu": 110, "bp_sys_sigma": 4, "sleep_mu": 32400, "sleep_sigma": 3600, 
        "steps_mu": 15000, "steps_sigma": 3000, 
    },
}


INDUSTRY_TARGETS = {
    'sleep_duration_sum': 28800, 
    'hrv_mean': 50.0,              
    'bp_systolic_mean': 120.0,    
}
ABSOLUTE_RISK_WEIGHT = 0.40