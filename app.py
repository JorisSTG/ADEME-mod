import streamlit as st
import pandas as pd
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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


st.title("Comparaison : Modèle / TRACC")

# -------- Paramètres --------
scenarios = ["2", "2_VC", "2-7", "2-7_VC", "4", "4_VC"]
villes = ["AGEN", "CARPENTRAS", "MACON", "MARIGNAGE", "NANCY", "RENNES", "TOURS", "TRAPPES"]
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
percentiles_list = [5, 25, 50, 75, 95]

# -------- Choix scénario et ville --------
scenario_sel = st.selectbox("Choisir le scénario :", scenarios)
ville_sel = st.selectbox("Choisir la ville :", villes)
base_folder = "ADEME"

# -------- Upload CSV modèle --------
uploaded = st.file_uploader("Déposer le fichier CSV du modèle (colonne unique T°C) :", type=["csv"])

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

   # -------- RMSE et précision sur percentiles --------
    def rmse(a, b):
        min_len = min(len(a), len(b))
        a_sorted = np.sort(a[:min_len])
        b_sorted = np.sort(b[:min_len])
        return np.sqrt(np.nanmean((a_sorted - b_sorted) ** 2))

    def precision_normale(rmse_val, sigma_obs):
        if sigma_obs == 0:
            return 100.0  # pas de variabilité, modèle parfait
        z = rmse_val / sigma_obs
        prob = norm.cdf(z)  # probabilité que X <= z*sigma
        precision = 100 * (1 - prob)
        return round(precision, 2)

    # -------- Boucle sur les mois --------
    results_rmse = []
    obs_mois_all = []
    start_idx_model = 0

    for mois, nb_heures in enumerate(heures_par_mois, start=1):
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
        obs_mois_vals = df_obs[df_obs["month"] == mois]["T2m"].values
        obs_mois_all.append(obs_mois_vals)

    # RMSE sur les distributions
        val_rmse = rmse(mod_mois, obs_mois_vals)

    # écart-type des observations
        sigma_obs = np.std(obs_mois_vals, ddof=1)

    # précision selon loi normale
        pct_precision = precision_normale(val_rmse, sigma_obs)

        results_rmse.append({
            "Mois": mois,
            "RMSE (°C)": round(val_rmse, 2),
            "Sigma_obs (°C)": round(sigma_obs, 2),
            "Précision (%)": pct_precision
        })

        start_idx_model += nb_heures

    # -------- DataFrame final --------
    df_rmse = pd.DataFrame(results_rmse)
    st.subheader("Précision du modèle : RMSE mensuels et précision selon loi normale (%)")
    st.dataframe(df_rmse, hide_index=True)

    # -------- Seuils --------
    t_sup_thresholds = st.text_input("Seuils Tmax supérieur (°C, séparés par des virgules)", "25,30,35")
    t_inf_thresholds = st.text_input("Seuils Tmin inférieur (°C, séparés par des virgules)", "-5,0,5")

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
    st.subheader(f"Histogrammes horaire : Modèle / TRACC +{scenario_sel}/{ville_sel}")

    # Bins fixes
    bins = np.arange(-5, 46, 1)          # -5 à 45
    bin_centers = (bins[:-1] + bins[1:]) / 2   # valeurs numériques
    bin_centers_int = bin_centers.astype(int)  # pour l’ordre
    bin_labels = [str(int(x)) for x in bin_centers]  # labels affichés

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
            width=0.4, label=f"Projection TRACC +{scenario_sel}/{ville_sel}", color="blue"
        )
        ax.bar(
            df_plot["Temp_Num"] + 0.2, df_plot["Modèle"],
            width=0.4, label="Modèle", color="red"
        )

        ax.set_title(f"Mois {mois} - Durée en heure par seuil de température dans le mois")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Durée en heure")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

    # ---- Calcul et affichage ----
    results_temp = []

    def rmse_hours(obs_counts, mod_counts):
        min_len = min(len(obs_counts), len(mod_counts))
        return np.sqrt(np.nanmean((np.array(obs_counts[:min_len]) - np.array(mod_counts[:min_len]))**2))

    for mois in range(1, 13):
        obs_hourly = obs_mois_all[mois-1]
        idx0 = sum(heures_par_mois[:mois-1])
        idx1 = sum(heures_par_mois[:mois])
        mod_hourly = model_values[idx0:idx1]

        # Comptage des heures par créneau
        obs_counts = count_hours_in_bins(obs_hourly, bins)
        mod_counts = count_hours_in_bins(mod_hourly, bins)

        # RMSE et sigma sur les counts
        # ---- Précision simple sur heures par créneau ----
        total_hours = sum(obs_counts)
        hours_error = sum(abs(np.array(obs_counts) - np.array(mod_counts)))
        pct_precision = round(100 * (1 - hours_error / total_hours), 2)

        val_rmse = rmse_hours(obs_counts, mod_counts)
        
        results_temp.append({
            "Mois": mois,
            "RMSE heures": round(val_rmse,2),
            "Précision (%)": pct_precision
        })

    df_temp_precision = pd.DataFrame(results_temp)
    st.subheader(f"Précision du modèle sur la répartition des durées des plages de température (TRACC +{scenario_sel}/{ville_sel})")
    st.dataframe(df_temp_precision, hide_index=True)

    
    # -------- Graphiques CDF et percentiles --------
    st.subheader("Fonctions de répartition mensuelles (CDF)")
    df_percentiles_all = []


    for mois in range(1, 13):
        obs_mois = obs_mois_all[mois-1]
        mod_mois = model_values[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
    
    # Calcul des percentiles pour CDF
        obs_percentiles_100 = np.percentile(obs_mois, np.linspace(0, 100, 100))
        mod_percentiles_100 = np.percentile(mod_mois, np.linspace(0, 100, 100))
    
    # Graphique CDF
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            np.linspace(0, 100, 100),
            mod_percentiles_100,
            label="Modèle",
            color="red"
        )
        ax.plot(
            np.linspace(0, 100, 100),
            obs_percentiles_100,
            label=f"TRACC +{scenario_sel}/{ville_sel}",
            color="blue"  
        )
    
        ax.set_title(f"Mois {mois} - Fonction de répartition", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")  # transparent
    
        st.pyplot(fig)
        plt.close(fig)
    
        # Percentiles clés
        obs_p = np.percentile(obs_mois, percentiles_list)
        mod_p = np.percentile(mod_mois, percentiles_list)
        df_p = pd.DataFrame({
            "Percentile": [f"P{p}" for p in percentiles_list],
            f"TRACC +{scenario_sel}/{ville_sel}": obs_p,
            "Modèle": mod_p
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
    st.subheader(f"Bilan modèle vs TRACC +{scenario_sel}/{ville_sel} (Modèle - TRACC)")
    df_bilan = pd.DataFrame(df_percentiles_all).round(2)
    df_bilan["Ecart"] = df_bilan["Mod"] - df_bilan["Obs"]
    df_bilan_pivot = df_bilan.pivot(index="Percentile", columns="Mois", values="Ecart").round(2)

    def color_map(val):
        if pd.isna(val): return ""
        if val < 0: return f"background-color: rgba(0,0,255,{min(abs(val)/5,1)})"
        else: return f"background-color: rgba(255,0,0,{min(val/5,1)})"

    st.dataframe(df_bilan_pivot.style.applymap(color_map).format("{:.2f}"))

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

    # -------- Graphique CDF comparatif par scénario avec matplotlib --------
    st.subheader("CDF comparatif des scénarios (trait plein / pointillé par paire)")

    # Définir les paires de scénarios et leurs couleurs
    scenario_pairs = [("2", "2_VC"), ("2-7", "2-7_VC"), ("4", "4_VC")]
    colors = ["green", "orange", "indigo"]  # couleur par paire

    for mois in range(1, 13):
        fig, ax = plt.subplots(figsize=(12, 4))
        for i, (sc1, sc2) in enumerate(scenario_pairs):
            color = colors[i]

        # Premier scénario : trait plein
            nc_file = os.path.join(base_folder, sc1, f"{ville_sel}.nc")
            ds = xr.open_dataset(nc_file, decode_times=True)
            temps = ds["T2m"].to_series().values
            obs_mois = temps[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
            cdf_values = np.percentile(obs_mois, np.linspace(0, 100, 100))
            ax.plot(
                np.linspace(0, 100, 100),
                cdf_values,
                label=f"{sc1}",
                color=color,
                linestyle="-"
            )

            # Deuxième scénario : trait pointillé
            nc_file = os.path.join(base_folder, sc2, f"{ville_sel}.nc")
            ds = xr.open_dataset(nc_file, decode_times=True)
            temps = ds["T2m"].to_series().values
            obs_mois = temps[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
            cdf_values = np.percentile(obs_mois, np.linspace(0, 100, 100))
            ax.plot(
                np.linspace(0, 100, 100),
                cdf_values,
                label=f"{sc2}",
                color=color,
                linestyle="--"
            )

        ax.set_title(f"Mois {mois} - CDF comparatif par scénario", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")  # transparent

        st.pyplot(fig)
        plt.close(fig)


    # Heatmap des écarts des percentiles par mois et scénario
    st.subheader(f"Ecarts des percentiles (Modèle - +{scenario_sel}/{ville_sel})")
    ref_model = {}
    for mois in range(1, 13):
        obs_mois = obs_mois_all[mois-1]
        mod_mois = model_values[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
        for i, p in enumerate(percentiles_list):
            ref_model[(mois, f"P{p}")] = np.percentile(mod_mois, p)

    for p in percentiles_list:
        df_ecart = df_scenarios[df_scenarios["Percentile"] == f"P{p}"].copy()
        df_ecart["Ecart"] = -df_ecart.apply(lambda row: row["Valeur"] - ref_model[(row["Mois"], f"P{p}")], axis=1)
         # Conversion en float arrondi à 2 décimales
        df_ecart["Ecart"] = df_ecart["Ecart"].round(2).astype(float)
        df_pivot = df_ecart.pivot(index="Scénario", columns="Mois", values="Ecart").round(2)
        st.write(f"Percentile {p} / Modèle - TRACC +{scenario_sel}/{ville_sel} ")
        st.dataframe(df_pivot.style.background_gradient(cmap="coolwarm").format("{:.2f}"))
