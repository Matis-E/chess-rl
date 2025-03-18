import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv("gym-chess/model_dqn201_gdlrtrain.csv", index_col=0)  # Utilise la première colonne comme index

# Tracer le graphe
plt.figure(figsize=(8, 5))
plt.plot(df.index, df.iloc[:, 0], marker='o', linestyle='-', color='b', label="Valeurs")

# Ajouter titres et labels
plt.xlabel("Index")
plt.ylabel("Valeur")
plt.title("Graphique des valeurs en fonction des index")
plt.legend()
plt.grid(True)

# Afficher le graphe
plt.show()