import streamlit as st
import pandas as pd
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt

# ---- STYLE sombre pour se fondre avec le thème Streamlit ----
plt.style.use("dark_background")

# Personnalisation pour coller exactement au thème Streamlit
plt.rcParams.update({
    "figure.facecolor": "none",      # fond totalement transparent
    "axes.facecolor": "none",        # fond transparent
    "savefig.facecolor": "none",     # export sans fond
    "axes.edgecolor": "#FFFFFF",     # axes blancs
    "axes.labelcolor": "#FFFFFF",    # labels blancs
    "xtick.color": "#DDDDDD",
    "ytick.color": "#DDDDDD",
    "text.color": "#FFFFFF",
})


st.title("Comparaison Modèle / Scénario TRACC (année type)")

# -------- Paramètres --------
scenarios = ["2", "2_VC", "2-7", "2-7_VC", "4", "4_VC"]
villes = ["AGEN", "CARPENTRAS", "MACON", "MARIGNAGE", "NANCY", "RENNES", "TOURS", "TRAPPES"]
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
percentiles_list = [1, 5, 10, 50, 90, 95, 99]

# -------- Choix scénario et ville --------
scenario_sel = st.selectbox("Choisir le scénario :", scenarios)
ville_sel = st.selectbox("Choisir la ville :", villes)
base_folder = "ADEME"

# -------- Upload CSV modèle --------
uploaded = st.file_uploader("Déposer le fichier CSV du modèle (colonne unique T°C) :", type=["csv"])

# -------- Seuils --------
t_sup_thresholds = st.text_input("Seuils Tmax supérieur (°C, séparés par des virgules)", "25,30,35")
t_inf_thresholds = st.text_input("Seuils Tmin inférieur (°C, séparés par des virgules)", "-5,0,5")

if uploaded:
    # Lecture CSV modèle
    model_values = pd.read_csv(uploaded, header=0).iloc[:, 0].values

    # Lecture NetCDF
    nc_file_sel = os.path.join(base_folder, scenario_sel, f"{ville_sel}.nc")
    ds_obs = xr.open_dataset(nc_file_sel, decode_times=True)
    obs_series = ds_obs["T2m"].to_series()
    df_obs = obs_series.reset_index()
    df_obs["year"] = df_obs["time"].dt.year
    df_obs["month"] = df_obs["time"].dt.month
    df_obs["day"] = df_obs["time"].dt.day

    # Supprimer 29 février si présent
    df_obs = df_obs[~((df_obs["month"] == 2) & (df_obs["day"] == 29))]

    # -------- RMSE sur percentiles --------
    def rmse(a, b):
        return np.sqrt(np.nanmean((a - b) ** 2))

    results_rmse = []
    obs_mois_all = []
    start_idx_model = 0

    for mois, nb_heures in enumerate(heures_par_mois, start=1):
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
        mod_sorted = np.sort(mod_mois)
        obs_mois_vals = df_obs[df_obs["month"] == mois]["T2m"].values
        obs_sorted = np.sort(obs_mois_vals)
        obs_mois_all.append(obs_sorted)
        min_len = min(len(mod_sorted), len(obs_sorted))
        val_rmse = rmse(mod_sorted[:min_len], obs_sorted[:min_len])
        results_rmse.append({"Mois": mois, "RMSE_percentiles": round(val_rmse, 2)})
        start_idx_model += nb_heures

    df_rmse = pd.DataFrame(results_rmse).round(2)
    st.subheader("Précision du modèle : RMSE mensuels en °C")
    st.dataframe(df_rmse, hide_index=True)

    # -------- Nombre d'heures sup/inf et écart obs-mod --------
    t_sup_thresholds_list = [float(x.strip()) for x in t_sup_thresholds.split(",")]
    t_inf_thresholds_list = [float(x.strip()) for x in t_inf_thresholds.split(",")]

    stats = []
    for mois, nb_heures in enumerate(heures_par_mois, start=1):
        mod_mois = model_values[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
        obs_mois = obs_mois_all[mois-1]
        for seuil in t_sup_thresholds_list:
            heures_obs = np.sum(obs_mois > seuil)
            nb_heures_mod = np.sum(mod_mois > seuil)
            ecart = heures_obs - nb_heures_mod
            stats.append({
                "Mois": mois,
                "Seuil": seuil,
                "Type": "Supérieur",
                "Nb_heures_obs": heures_obs,
                "Nb_heures_mod": nb_heures_mod,
                "Ecart_obs_mod": ecart
            })
        for seuil in t_inf_thresholds_list:
            heures_obs = np.sum(obs_mois < seuil)
            nb_heures_mod = np.sum(mod_mois < seuil)
            ecart = heures_obs - nb_heures_mod
            stats.append({
                "Mois": mois,
                "Seuil": seuil,
                "Type": "Inférieur",
                "Nb_heures_obs": heures_obs,
                "Nb_heures_mod": nb_heures_mod,
                "Ecart_obs_mod": ecart
            })

    df_stats = pd.DataFrame(stats).round(2)
    st.subheader("Nombre d'heures sup/inf et écart obs-mod")
    st.dataframe(df_stats, hide_index=True)

    # -------- Graphes en barres pour les plages de température (1°C) --------
    st.subheader("Histogrammes journaliers : Observations vs Modèle (barres côte à côte)")

    # Bins fixes
    bins = np.arange(-5, 46, 1)          # -5 à 45
    bin_centers = (bins[:-1] + bins[1:]) / 2   # valeurs numériques
    bin_centers_int = bin_centers.astype(int)  # pour l’ordre
    bin_labels = [str(int(x)) for x in bin_centers]  # labels affichés

    # -------- Graphes en barres pour les plages de température (1°C) --------
    st.subheader("Histogrammes horaires : Observations vs Modèle (barres côte à côte)")

# Bins fixes (-5 à 45°C)
    bins = np.arange(-5, 46, 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_centers_int = bin_centers.astype(int)
    bin_labels = [str(int(x)) for x in bin_centers]

    def count_hours_in_bins(temp_hourly, bins):
        """Histogramme en HEURES (pas de conversion en jours)."""
        counts, _ = np.histogram(temp_hourly, bins=bins)
        return counts

    for mois in range(1, 13):

    # ---- Observations : comptage en heures ----
        obs_hourly = obs_mois_all[mois-1]
        obs_counts = count_hours_in_bins(obs_hourly, bins)

    # ---- Modèle ----
        idx0 = sum(heures_par_mois[:mois-1])
        idx1 = sum(heures_par_mois[:mois])
        mod_hourly = model_values[idx0:idx1]
        mod_counts = count_hours_in_bins(mod_hourly, bins)

    # ---- DataFrame trié ----
        df_plot = pd.DataFrame({
            "Temp_Num": bin_centers_int,     # pour l'ordre
            "Température": bin_labels,
            "Observations": obs_counts,
            "Modèle": mod_counts
        }).sort_values("Temp_Num")

       fig, ax = plt.subplots(figsize=(14, 4))

        ax.bar(
            df_plot["Temp_Num"] - 0.2, df_plot["Observations"],
            width=0.4, label="Observations"
        )
        ax.bar(
            df_plot["Temp_Num"] + 0.2, df_plot["Modèle"],
            width=0.4, label="Modèle"
        )

        ax.set_title(f"Mois {mois} - Nombre de jours par température")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Nombre de jours")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)



    # -------- Graphiques CDF et percentiles --------
    st.subheader("Fonctions de répartition mensuelles (CDF)")
    df_percentiles_all = []

    for mois in range(1, 13):
        obs_mois = obs_mois_all[mois-1]
        mod_mois = model_values[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
        obs_percentiles_100 = np.percentile(obs_mois, np.linspace(0,100,100))
        mod_percentiles_100 = np.percentile(mod_mois, np.linspace(0,100,100))
        df_cdf = pd.DataFrame({"Obs": obs_percentiles_100, "Mod": mod_percentiles_100}).round(2)
        st.write(f"Mois {mois} - Fonction de répartition")
        st.line_chart(df_cdf)

        obs_p = np.percentile(obs_mois, percentiles_list)
        mod_p = np.percentile(mod_mois, percentiles_list)
        df_p = pd.DataFrame({
            "Percentile": [f"P{p}" for p in percentiles_list],
            "Obs": obs_p,
            "Mod": mod_p
        }).round(2)
        st.write(f"Mois {mois} - Percentiles")
        st.dataframe(df_p, hide_index=True)

        for i, p in enumerate(percentiles_list):
            df_percentiles_all.append({
                "Mois": mois,
                "Percentile": f"P{p}",
                "Obs": obs_p[i],
                "Mod": mod_p[i]
            })

    # -------- Tableau bilan chaud/froid --------
    st.subheader("Bilan modèle vs Observations (chaud/froid)")
    df_bilan = pd.DataFrame(df_percentiles_all).round(2)
    df_bilan["Ecart"] = df_bilan["Mod"] - df_bilan["Obs"]
    df_bilan_pivot = df_bilan.pivot(index="Percentile", columns="Mois", values="Ecart").round(2)

    def color_map(val):
        if pd.isna(val): return ""
        if val < 0: return f"background-color: rgba(0,0,255,{min(abs(val)/5,1)})"
        else: return f"background-color: rgba(255,0,0,{min(val/5,1)})"

    st.dataframe(df_bilan_pivot.style.applymap(color_map))

    # -------- Section multi-scénarios pour la ville --------
    st.subheader(f"Comparaison multi-scénarios pour {ville_sel}")
    df_percentiles_scenarios = []

    for scenario in scenarios:
        nc_file = os.path.join(base_folder, scenario, f"{ville_sel}.nc")
        ds = xr.open_dataset(nc_file, decode_times=True)
        temps = ds["T2m"].to_series().values
        start_idx = 0
        for mois, nb_heures in enumerate(heures_par_mois, start=1):
            obs_mois = temps[start_idx:start_idx + nb_heures]
            obs_p = np.percentile(obs_mois, percentiles_list)
            for i, p in enumerate(percentiles_list):
                df_percentiles_scenarios.append({
                    "Scénario": scenario,
                    "Mois": mois,
                    "Percentile": f"P{p}",
                    "Valeur": round(obs_p[i],2)
                })
            start_idx += nb_heures

    df_scenarios = pd.DataFrame(df_percentiles_scenarios)

    # Graphique CDF comparatif par mois
    st.subheader("CDF comparatif par scénario")
    for mois in range(1,13):
        cdf_dict = {}
        for scenario in scenarios:
            nc_file = os.path.join(base_folder, scenario, f"{ville_sel}.nc")
            ds = xr.open_dataset(nc_file, decode_times=True)
            temps = ds["T2m"].to_series().values
            obs_mois = temps[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
            cdf_dict[scenario] = np.percentile(obs_mois, np.linspace(0,100,100))
        df_cdf_scenarios = pd.DataFrame(cdf_dict).round(2)
        st.write(f"Mois {mois}")
        st.line_chart(df_cdf_scenarios)

    # Heatmap des écarts des percentiles par mois et scénario
    st.subheader("Heatmap des écarts des percentiles par mois et scénario (vs modèle)")
    ref_model = {}
    for mois in range(1, 13):
        obs_mois = obs_mois_all[mois-1]
        mod_mois = model_values[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
        for i, p in enumerate(percentiles_list):
            ref_model[(mois, f"P{p}")] = np.percentile(mod_mois, p)

    for p in percentiles_list:
        df_ecart = df_scenarios[df_scenarios["Percentile"] == f"P{p}"].copy()
        df_ecart["Ecart"] = df_ecart.apply(lambda row: row["Valeur"] - ref_model[(row["Mois"], f"P{p}")], axis=1)
        df_pivot = df_ecart.pivot(index="Scénario", columns="Mois", values="Ecart").round(2)
        st.write(f"Percentile {p} - Écart vs modèle")
        st.dataframe(df_pivot.style.background_gradient(cmap="coolwarm"))
