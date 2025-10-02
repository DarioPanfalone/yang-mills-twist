import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sostituisci con il nome del tuo file
filename = "dati_random.dat"

# Carica i dati, saltando la prima riga
df = pd.read_csv(
    filename,
    delim_whitespace=True,
    skiprows=200,
    names=["index_conf", "plaq_t", "plaq_s", "poly_re", "poly_im"]
)

df["plaq_avg"] = (2 * df["plaq_t"] + df["plaq_s"]) / 3
#df["plaq_avg"] =  df["plaq_s"]

# Media e errore statistico (deviazione standard / sqrt(N))
mean = df["plaq_avg"].mean()
std_err = df["plaq_avg"].std(ddof=1) / np.sqrt(len(df))

print(f"Media: {mean:.6f} ± {std_err:.6f}")

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
