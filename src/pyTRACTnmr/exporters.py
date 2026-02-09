def generate_fit_script(csv_filename: str, fill_color_alpha: str, fill_color_beta: str) -> str:
    return f"""import numpy as np
import matplotlib.pyplot as plt
import csv

csv_file = '{csv_filename}'
data_raw = []
data_fit = []

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    header_found = False
    for row in reader:
        if not row: continue
        if not header_found:
            if row[0] == 'Delay (s)':
                header_found = True
            continue
        if row[0] != '':
            data_raw.append([float(row[0]), float(row[1]), float(row[2])])
        if len(row) > 4 and row[4] != '':
            data_fit.append([float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])])

data_raw = np.array(data_raw)
data_fit = np.array(data_fit)
delays = data_raw[:, 0]
alpha_intensity = data_raw[:, 1]
beta_intensity = data_raw[:, 2]
x_smooth = data_fit[:, 0]
fit_a = data_fit[:, 1]
ci_a = data_fit[:, 2]
fit_b = data_fit[:, 3]
ci_b = data_fit[:, 4]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(delays, alpha_intensity, 'bo', label='Alpha Data')
plt.plot(delays, beta_intensity, 'ro', label='Beta Data')
plt.plot(x_smooth, fit_a, 'b-', label='Alpha Fit')
plt.fill_between(x_smooth, fit_a - ci_a, fit_a + ci_a, color='{fill_color_alpha}', alpha=0.2)
plt.plot(x_smooth, fit_b, 'r-', label='Beta Fit')
plt.fill_between(x_smooth, fit_b - ci_b, fit_b + ci_b, color='{fill_color_beta}', alpha=0.2)
plt.xlabel('Delay (s)')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.show()
"""

def generate_sliding_script(csv_filename: str, fill_color: str) -> str:
    return f"""import numpy as np
import matplotlib.pyplot as plt

csv_file = '{csv_filename}'
data = np.genfromtxt(csv_file, delimiter=',', comments='#', skip_header=3)
ppms = data[:, 0]
taus = data[:, 1]
errors = data[:, 2]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(ppms, taus, 'b-', label='Tau_c')
plt.fill_between(ppms, taus - errors, taus + errors, color='{fill_color}', alpha=0.2, label='Error')
plt.xlabel('1H (ppm)')
plt.ylabel('Tau_c (ns)')
plt.gca().invert_xaxis()
plt.legend()
plt.grid(True)
plt.show()
"""
