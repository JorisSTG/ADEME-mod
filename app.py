#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 16:59:26 2025

@author: saint-genesj
"""

import streamlit as st
import pandas as pd
import xarray as xr
import glob
import os
import numpy as np

st.title("Comparaison modèle / Observations par scénario et ville")

# -------- Paramètres --------
base_folder = "ADEME"  # dossier racine où sont stockés les sous-dossiers des scénarios
scenarios = ["2", "2-7", "4", "2_VC", "2-7_VC", "4_VC"]  # noms des sous-dossiers
villes = ["AGEN", "CARPENTRAS", "MACON", "MARIGNANE", "NANCY", "RENNES", "TOURS", "TRAPPES"]  # <-- à remplir avec les noms de vos villes (sans .nc)
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]  # année type
percentiles_list = [1, 5, 10, 50, 90, 95, 99]

# -------- Sélection scénario et ville --------
scenario_sel = st.selectbox("Choisir le scénario :", scenarios)
ville_sel = st.selectbox("Choisir la ville :", villes)

# Chemin vers le NetCDF
nc_file_sel = os.path.join(base_folder, scenario_sel, f"{ville_sel}.nc")

# -------- Upload CSV modèle --------
uploaded = st.file_uploader("Dépose ton fichier CSV modèle (colonne unique T) :", type=["csv"])

# -------- Seuils --------
t_sup_thresholds = st.text_input("Seuils Tmax sup (°C, séparés par des virgules)", "25,30,35")
t_inf_thresholds = st.text_input("Seuils Tmin inf (°C, séparés par des virgules)", "-5,0,5")

if uploaded:
    # Lecture CSV modèle
    model_values = pd.read_csv(uploaded, header=0).iloc[:, 0].values

    # Lecture NetCDF
    ds_obs = xr.open_dataset(nc_file_sel, decode_times=True)
    obs_series = ds_obs["T2m"].to_series()
    df_obs = obs_series.reset_index()
    df_obs.rename(columns={"T2m": "T2m", "time": "time"}, inplace=True)
    df_obs["year"] = df_obs["time"].dt.year
    df_obs["month"] = df_obs["time"].dt.month
    df_obs["day"] = df_obs["time"].dt.day

    # Supprimer 29 février
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

        # Observations pour ce mois (1 seule année ici)
        vals = df_obs[df_obs["month"] == mois]["T2m"].values
        obs_mois_all.append(np.sort(vals))

        min_len = min(len(mod_sorted), len(vals))
        val_rmse = rmse(mod_sorted[:min_len], np.sort(vals)[:min_len])
        results_rmse.append({"Mois": mois, "RMSE_percentiles": round(val_rmse, 2)})

        start_idx_model += nb_heures

    df_rmse = pd.DataFrame(results_rmse)
    st.subheader("RMSE sur les percentiles mensuels")
    st.dataframe(df_rmse)

    # -------- Nombre moyen d'heures sup/inf et écart obs-mod --------
    t_sup_thresholds_list = [float(x.strip()) for x in t_sup_thresholds.split(",")]
    t_inf_thresholds_list = [float(x.strip()) for x in t_inf_thresholds.split(",")]
    stats = []

    for mois, nb_heures in enumerate(heures_par_mois, start=1):
        mod_mois = model_values[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]
        obs_mois = obs_mois_all[mois-1]

        # Heures supérieures
        for seuil in t_sup_thresholds_list:
            nb_heures_obs = np.sum(obs_mois > seuil)
            nb_heures_mod = np.sum(mod_mois > seuil)
            ecart = nb_heures_obs - nb_heures_mod
            stats.append({
                "Mois": mois,
                "Seuil": seuil,
                "Type": "Supérieur",
                "Nb_heures_obs": round(nb_heures_obs,2),
                "Nb_heures_mod": round(nb_heures_mod,2),
                "Ecart_obs_mod": round(ecart,2)
            })

        # Heures inférieures
        for seuil in t_inf_thresholds_list:
            nb_heures_obs = np.sum(obs_mois < seuil)
            nb_heures_mod = np.sum(mod_mois < seuil)
            ecart = nb_heures_obs - nb_heures_mod
            stats.append({
                "Mois": mois,
                "Seuil": seuil,
                "Type": "Inférieur",
                "Nb_heures_obs": round(nb_heures_obs,2),
                "Nb_heures_mod": round(nb_heures_mod,2),
                "Ecart_obs_mod": round(ecart,2)
            })

    df_stats = pd.DataFrame(stats)
    st.subheader("Nombre d'heures par seuil et écart obs-mod")
    st.dataframe(df_stats)

    # -------- Export CSV --------
    df_rmse.to_csv("RMSE_percentiles.csv", index=False)
    df_stats.to_csv("Heures_seuils.csv", index=False)
    st.download_button("Télécharger RMSE", "RMSE_percentiles.csv", "text/csv")
    st.download_button("Télécharger stats heures", "Heures_seuils.csv", "text/csv")

    # -------- Graphiques CDF et tableaux percentiles --------
    st.subheader("Fonctions de répartition mensuelles (CDF)")
    df_percentiles_all = []

    for mois in range(1, 13):
        obs_mois = obs_mois_all[mois-1]
        mod_mois = model_values[sum(heures_par_mois[:mois-1]):sum(heures_par_mois[:mois])]

        obs_percentiles_100 = np.percentile(obs_mois, np.linspace(0, 100, 100))
        mod_percentiles_100 = np.percentile(mod_mois, np.linspace(0, 100, 100))

        # Graphique CDF
        df_cdf = pd.DataFrame({"Obs": obs_percentiles_100, "Mod": mod_percentiles_100}).round(2)
        st.write(f"Mois {mois} - Fonction de répartition")
        st.line_chart(df_cdf)

        # Tableau percentiles
        obs_p = np.percentile(obs_mois, percentiles_list)
        mod_p = np.percentile(mod_mois, percentiles_list)
        df_p = pd.DataFrame({
            "Percentile": [f"P{p}" for p in percentiles_list],
            "Obs": obs_p,
            "Mod": mod_p
        }).round(2)
        st.write(f"Mois {mois} - Percentiles")
        st.dataframe(df_p)

        # Stockage pour tableau bilan
        for i, p in enumerate(percentiles_list):
            df_percentiles_all.append({
                "Mois": mois,
                "Percentile": f"P{p}",
                "Obs": obs_p[i],
                "Mod": mod_p[i]
            })

    # -------- Tableau bilan (plus chaud / plus froid) --------
    st.subheader("Bilan modèle vs Observations (chaud/froid)")
    df_bilan = pd.DataFrame(df_percentiles_all).round(2)
    df_bilan["Ecart"] = df_bilan["Mod"] - df_bilan["Obs"]

    df_bilan_pivot = df_bilan.pivot(index="Percentile", columns="Mois", values="Ecart").round(2)

    def color_map(val):
        if pd.isna(val):
            return ""
        if val < 0:
            return f"background-color: rgba(0,0,255,{min(abs(val)/5,1)})"
        else:
            return f"background-color: rgba(255,0,0,{min(val/5,1)})"

    st.dataframe(df_bilan_pivot.style.applymap(color_map))
