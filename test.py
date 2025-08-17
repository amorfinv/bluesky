import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the output.log file
with open('output.log', 'r') as f:
    lines = f.readlines()

# Parse the data
data = []
current_record = {}

for line in lines:
    line = line.strip()
    if line.startswith('SIM_t'):
        if current_record:
            data.append(current_record)
            current_record = {}
        current_record['SIM_t'] = float(line.split()[1])
    elif line.startswith('CAS_ap'):
        current_record['CAS_ap'] = float(line.split()[1])
    elif line.startswith('TAS_ap'):
        current_record['TAS_ap'] = float(line.split()[1])
    elif line.startswith('TAS_intent'):
        current_record['TAS_intent'] = float(line.split()[1])
    elif line.startswith('CAS_intent'):
        current_record['CAS_intent'] = float(line.split()[1])
    elif line.startswith('CAS_allow'):
        current_record['CAS_allow'] = float(line.split()[1])
    elif line.startswith('TAS_allow'):
        current_record['TAS_allow'] = float(line.split()[1])
    elif line.startswith('TAS_actual'):
        current_record['TAS_actual'] = float(line.split()[1])
    elif line.startswith('CAS_actual'):
        current_record['CAS_actual'] = float(line.split()[1])
    elif line.startswith('MACH_actual'):
        current_record['MACH_actual'] = float(line.split()[1])
    elif line.startswith('GS_actual'):
        current_record['GS_actual'] = float(line.split()[1])

# Add the last record
if current_record:
    data.append(current_record)

# Create DataFrame
df = pd.DataFrame(data)

# Plot all speed parameters
plt.figure(figsize=(12, 8))

tas_columns = ['TAS_intent', 'TAS_allow', 'TAS_actual']
cas_columns = ['CAS_intent', 'CAS_allow', 'CAS_actual']

# Plot in order: TAS first, then others, then CAS
speed_columns = tas_columns  + cas_columns

for col in speed_columns:
    if col in df.columns:
        plt.plot(df['SIM_t'], df[col], label=col, alpha=0.7)

plt.axvline(x=24, color='grey', linestyle='--', linewidth=1, alpha=0.8)
plt.text(24, plt.ylim()[1]*0.95, '(1)', ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
plt.axvline(x=64, color='grey', linestyle='--', linewidth=1, alpha=0.8)
plt.text(64, plt.ylim()[1]*0.95, '(2)', ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
plt.axvline(x=199, color='grey', linestyle='--', linewidth=1, alpha=0.8)
plt.text(199, plt.ylim()[1]*0.95, '(3)', ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
plt.axvline(x=334, color='grey', linestyle='--', linewidth=1, alpha=0.8)
plt.text(334, plt.ylim()[1]*0.95, '(4)', ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
plt.xlabel('Simulation Time (s)')
plt.ylabel('Speed')
plt.title('Aircraft Speed Parameters vs Simulation Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Display DataFrame info
print("DataFrame shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataFrame info:")
print(df.info())



