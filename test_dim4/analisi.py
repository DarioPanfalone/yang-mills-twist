import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sostituisci con il nome del tuo file
filename = "dati.dat"

# Carica i dati, saltando la prima riga
df = pd.read_csv(
    filename,
    delim_whitespace=True,
    skiprows=800,
    names=["index_conf", "plaq_s", "plaq_t", "poly_re", "poly_im"]
)

df["plaq_avg"] = (2 * df["plaq_t"] + df["plaq_s"]) / 3
df["plaq_avg_twist"] =  df["plaq_s"]
df["plaq_avg_notwist"] =  df["plaq_t"]

# Media e errore statistico (deviazione standard / sqrt(N))
mean = df["plaq_avg"].mean()
std_err = df["plaq_avg"].std(ddof=1) / np.sqrt(len(df))

mean_twist = df["plaq_avg_twist"].mean()
std_err_twsit = df["plaq_avg_twist"].std(ddof=1) / np.sqrt(len(df))

mean_notwist = df["plaq_avg_notwist"].mean()
std_err_notwsit = df["plaq_avg_notwist"].std(ddof=1) / np.sqrt(len(df))

print(len(df))
print(f"Media_tot: {mean:.6f} ± {std_err:.6f}")
print(f"Media_twist: {mean_twist:.6f} ± {std_err_twsit:.6f}")
print(f"Media_notwist: {mean_notwist:.6f} ± {std_err_notwsit:.6f}")

# Plot
plt.figure(figsize=(10,5))
plt.plot(df["index_conf"], df["plaq_avg"], ".", markersize=2, label="Plaquette")
plt.axhline(mean, color="red", linestyle="--", label=f"Media = {mean:.5f}")
plt.xlabel("idc")
plt.ylabel("Plaquette")
#plt.title("")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
