import streamlit as st
import pandas as pd
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---- STYLE sombre pour se fondre avec le thème Streamlit ----
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": "#FFFFFF",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#DDDDDD",
    "ytick.color": "#DDDDDD",
    "text.color": "#FFFFFF",
})


st.title("Comparaison : Modèle / TRACC")

st.markdown(
    """
    La caractérisation du modèle vis-à-vis des données issues de la TRACC est basée uniquement sur la température de l'air !
    
    **Note sur les couleurs :**  
    - Les couleurs visent à caractériser le **MODÈLE** (données issues du fichier `.csv`).  
    - Rouge → Modèle plus chaud que TRACC  
    - Bleu → Modèle plus froid que TRACC  
    - Pour les indicateurs de précision : vert → bon résultat, rouge → moins bon résultat
    """,
    unsafe_allow_html=True
)

# -------- Paramètres --------
scenarios = ["2", "2_VC", "2-7", "2-7_VC", "4", "4_VC"]
villes = ["AGEN", "CARPENTRAS", "MACON", "MARIGNAGE", "NANCY", "RENNES", "TOURS", "TRAPPES"]
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
percentiles_list = [10, 25, 50, 75, 90]

vmaxT=5
vminT=-5

vmaxP=100
vminP=0

vmaxH=100
vminH=-100

# -------- Noms des mois --------
mois_noms = {
    1: "01 - Janvier",   2: "02 - Février",  3: "03 - Mars",
    4: "04 - Avril",     5: "05 - Mai",      6: "06 - Juin",
    7: "07 - Juillet",   8: "08 - Août",     9: "09 - Septembre",
    10: "10 - Octobre", 11: "11 - Novembre", 12: "12 - Décembre"
}

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
    df_obs["month_num"] = df_obs["time"].dt.month
    df_obs["month"] = df_obs["month_num"].map(mois_noms)
    df_obs["day"] = df_obs["time"].dt.day

    # -------- RMSE --------
    def rmse(a, b):
        min_len = min(len(a), len(b))
        a_sorted = np.sort(a[:min_len])
        b_sorted = np.sort(b[:min_len])
        return np.sqrt(np.nanmean((a_sorted - b_sorted) ** 2))
    
    # -------- Précision basée sur les écarts de percentiles --------
    def precision_ecarts_percentiles(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.nan
        # Percentiles 1 à 99
        percentiles = np.arange(1, 100)
        pa = np.percentile(a, percentiles)
        pb = np.percentile(b, percentiles)
    
        # Différence moyenne normalisée par l'écart-type
        diff_moyenne = np.mean(np.abs(pa - pb))
        scale = np.std(pb)
        
        if scale == 0:
            return 100.0  # pas de variation dans b, a=b ou pas → score max
    
        score = 100 * (1 - diff_moyenne / (2*scale))
        score = max(0, min(100, score))  # on contraint entre 0 et 100
    
        return round(score, 2)

    # -------- Boucle sur les mois --------
    results_rmse = []
    obs_mois_all = []
    start_idx_model = 0

    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
        obs_mois_vals = df_obs[df_obs["month_num"] == mois_num]["T2m"].values
        obs_mois_all.append(obs_mois_vals)

        val_rmse = rmse(mod_mois, obs_mois_vals)
        pct_precision = precision_ecarts_percentiles(mod_mois, obs_mois_vals)

        results_rmse.append({
            "Mois": mois,
            "RMSE (°C)": round(val_rmse, 2),
            "Précision percentile (%)": pct_precision
        })

        start_idx_model += nb_heures

    # -------- DataFrame final --------
    df_rmse = pd.DataFrame(results_rmse)
    df_rmse_styled = (
        df_rmse.style
        .background_gradient(subset=["Précision percentile (%)"], cmap="RdYlGn", vmin=vminP, vmax=vmaxP, axis=None)
        .format({"Précision percentile (%)": "{:.2f}", "RMSE (°C)": "{:.2f}"})
    )
    st.subheader("Précision du modèle : RMSE et précision via écarts des percentiles")
    
    st.markdown(
        """
        La précision est calculé selon la différence moyenne entre les percentiles (RMSE) et l'écart-type du mois de la TRACC
        """,
        unsafe_allow_html=True
    )
    st.dataframe(df_rmse_styled, hide_index=True)

    # -------- Seuils --------
    t_sup_thresholds = st.text_input("Seuils Tmax supérieur (°C, séparés par des virgules)", "25,30,35")
    t_inf_thresholds = st.text_input("Seuils Tmin inférieur (°C, séparés par des virgules)", "-5,0,5")
    t_sup_thresholds_list = [int(float(x.strip())) for x in t_sup_thresholds.split(",")]
    t_inf_thresholds_list = [int(float(x.strip())) for x in t_inf_thresholds.split(",")]
    
    stats_sup = []
    stats_inf = []
    
    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_mois = model_values[idx0:idx1]
        obs_mois = obs_mois_all[mois_num-1]
    
        # Seuils supérieurs
        for seuil in t_sup_thresholds_list:
            heures_obs = np.sum(obs_mois > seuil)
            nb_heures_mod = np.sum(mod_mois > seuil)
            ecart = nb_heures_mod - heures_obs  # Modèle - TRACC
            stats_sup.append({
                "Mois": mois,
                "Seuil (°C)": f"{seuil}",
                "Heures Modèle": nb_heures_mod,
                "Heures TRACC": heures_obs,
                "Ecart (Modèle - TRACC)": ecart
            })
        
        # Seuils inférieurs
        for seuil in t_inf_thresholds_list:
            heures_obs = np.sum(obs_mois < seuil)
            nb_heures_mod = np.sum(mod_mois < seuil)
            ecart = nb_heures_mod - heures_obs  # Modèle - TRACC
            stats_inf.append({
                "Mois": mois,
                "Seuil (°C)": f"{seuil}",
                "Heures Modèle": nb_heures_mod,
                "Heures TRACC": heures_obs,
                "Ecart (Modèle - TRACC)": ecart
            })
    
    # Création des DataFrames
    df_sup = pd.DataFrame(stats_sup)
    df_inf = pd.DataFrame(stats_inf)
    
    # Conversion en int
    for df in [df_sup, df_inf]:
        df["Heures Modèle"] = df["Heures Modèle"].astype(int)
        df["Heures TRACC"] = df["Heures TRACC"].astype(int)
        df["Ecart (Modèle - TRACC)"] = df["Ecart (Modèle - TRACC)"].astype(int)
    
    # Style : seuils supérieurs → rouge = plus chaud
    df_sup_styled = (
        df_sup.style
        .background_gradient(subset=["Ecart (Modèle - TRACC)"], cmap="bwr", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.subheader("Nombre d'heures supérieur aux seuils")
    st.dataframe(df_sup_styled, hide_index=True)
    
    # Style : seuils inférieurs → rouge = plus froid
    # Pour inverser les couleurs, on peut juste inverser le cmap
    df_inf_styled = (
        df_inf.style
        .background_gradient(subset=["Ecart (Modèle - TRACC)"], cmap="bwr_r", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.subheader("Nombre d'heures inférieur aux seuils")
    st.dataframe(df_inf_styled, hide_index=True)


    # -------- Histogrammes par plage de température --------
    st.subheader(f"Histogrammes horaire : Modèle et TRACC +{scenario_sel}/{ville_sel} [X°C,X+1°C[")
    # Bins correspondant à [X, X+1[ pour chaque température entière
    bin_edges = bins = np.arange(-5, 46, 1)  # bornes des bins
    bin_labels = bin_edges[:-1].astype(int)  # labels = début de l'intervalle
    
    def count_hours_in_bins(temp_hourly, bins):
        counts, _ = np.histogram(temp_hourly, bins=bins)
        return counts
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        
        # Observations
        obs_hourly = obs_mois_all[mois_num-1]
        obs_counts = count_hours_in_bins(obs_hourly, bin_edges)
        
        # Modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_hourly = model_values[idx0:idx1]
        mod_counts = count_hours_in_bins(mod_hourly, bin_edges)
        
        # Préparer le DataFrame pour le plot
        df_plot = pd.DataFrame({
            "Temp_Num": bin_labels,
            "Température": bin_labels.astype(str),
            "TRACC": obs_counts,
            "Modèle": mod_counts
        }).sort_values("Temp_Num")
        
        # Création du plot
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(df_plot["Temp_Num"] - 0.2, df_plot["TRACC"], width=0.4, label=f"TRACC +{scenario_sel}/{ville_sel}", color="blue")
        ax.bar(df_plot["Temp_Num"] + 0.2, df_plot["Modèle"], width=0.4, label="Modèle", color="red")
        ax.set_title(f"{mois} - Durée en heure par seuil de température")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Durée en heure")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


    # -------- Précision par créneau horaire --------
    results_temp = []
    def rmse_hours(obs_counts, mod_counts):
        min_len = min(len(obs_counts), len(mod_counts))
        return np.sqrt(np.nanmean((np.array(obs_counts[:min_len]) - np.array(mod_counts[:min_len]))**2))

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        obs_hourly = obs_mois_all[mois_num-1]
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_hourly = model_values[idx0:idx1]
        obs_counts = count_hours_in_bins(obs_hourly, bins)
        mod_counts = count_hours_in_bins(mod_hourly, bins)
        total_hours = 2*heures_par_mois[mois_num-1]
        hours_error = sum(abs(np.array(obs_counts) - np.array(mod_counts)))
        pct_precision = round(100 * (1 - hours_error / total_hours), 2)
        val_rmse = rmse_hours(obs_counts, mod_counts)
        results_temp.append({
            "Mois": mois,
            "RMSE heures": round(val_rmse,2),
            "Précision (%)": pct_precision
        })

    df_temp_precision = pd.DataFrame(results_temp)
    df_temp_precision_styled = df_temp_precision.style \
        .background_gradient(subset=["Précision (%)"], cmap="RdYlGn", vmin=vminP, vmax=vmaxP, axis=None) \
        .format({"Précision (%)": "{:.2f}", "RMSE heures": "{:.2f}"})

    st.subheader(f"Précision du modèle sur la répartition des durées des plages de température (TRACC +{scenario_sel}/{ville_sel})")
    st.dataframe(df_temp_precision_styled, hide_index=True)

    # ============================
    #   COURBES Tn / Tmoy / Tx
    # ============================
    st.subheader("Évolution mensuelle : Tn / Tmoy / Tx (Modèle vs TRACC)")
    st.markdown(
        """  
        - Les valeurs tracés représente le Tn et Tx absolue du mois (soit P0 et P100) 
        - La Tmoy est une moyenne globale du mois à tous les pas de temps
        - En outre, les températures du mois ne dépassent jamais les bornes Tn et Tx
        """,
        unsafe_allow_html=True
    )
    # Calcul des Tn/Tmoy/Tx pour 12 mois
    results_tstats = []
    for mois_num in range(1, 12+1):
        mois = mois_noms[mois_num]
    
        # Observations
        obs_vals = obs_mois_all[mois_num-1]
        obs_tn = np.min(obs_vals)
        obs_tm = np.mean(obs_vals)
        obs_tx = np.max(obs_vals)
    
        # Modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_vals = model_values[idx0:idx1]
        mod_tn = np.min(mod_vals)
        mod_tm = np.mean(mod_vals)
        mod_tx = np.max(mod_vals)
    
        results_tstats.append({
            "Mois": mois,
            "TRACC_Tn": obs_tn, "Modèle_Tn": mod_tn, "TRACC_Tm": obs_tm, "Modèle_Tm": mod_tm, "TRACC_Tx": obs_tx, "Modèle_Tx": mod_tx
        })
    
    df_tstats = pd.DataFrame(results_tstats)
    
    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14,4))

    ax.plot(df_tstats["Mois"], df_tstats["Modèle_Tx"], color="red", label="Modèle Tx", linestyle="-")
    ax.plot(df_tstats["Mois"], df_tstats["Modèle_Tm"], color="white", label="Modèle Tmoy", linestyle="-")
    ax.plot(df_tstats["Mois"], df_tstats["Modèle_Tn"], color="cyan", label="Modèle Tn", linestyle="-")

    ax.plot(df_tstats["Mois"], df_tstats["TRACC_Tx"], color="red", label="TRACC Tx", linestyle="--")
    ax.plot(df_tstats["Mois"], df_tstats["TRACC_Tm"], color="white", label="TRACC Tmoy", linestyle="--")
    ax.plot(df_tstats["Mois"], df_tstats["TRACC_Tn"], color="cyan", label="TRACC Tn", linestyle="--")

    ax.set_title(f"Tn / Tmoy / Tx – Modèle vs TRACC +{scenario_sel}/{ville_sel}")
    ax.set_ylabel("Température (°C)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(facecolor="black")
    
    st.pyplot(fig)
    plt.close(fig)
    
    # ---- Tableau correspondant ----
    st.write("Tableau Tn / Tmoy / Tx")
    st.dataframe(df_tstats.round(2), hide_index=True)

    # ============================
    #  SECTION: Tn / Tmoy / Tx journaliers
    # ============================
    st.subheader("Tn / Tmoy / Tx journaliers — CDF par mois et tableaux de percentiles")
    
    def daily_stats_from_hourly(hourly):
        """
        Retourne trois tableaux journaliers (min, mean, max).
        Tronque si nécessaire pour avoir des jours complets (24h).
        """
        if len(hourly) < 24:
            return np.array([]), np.array([]), np.array([])
        n_full_days = len(hourly) // 24
        arr = np.array(hourly[: n_full_days * 24]).reshape((n_full_days, 24))
        daily_min = arr.min(axis=1)
        daily_mean = arr.mean(axis=1)
        daily_max = arr.max(axis=1)
        return daily_min, daily_mean, daily_max
    
    # percentiles pour les petits tableaux
    pct_table = percentiles_list  # utilise la liste déjà définie en haut (ex: [10,25,50,75,90])
    pct_for_cdf = np.linspace(0, 100, 100)  # pour tracer les CDF
    
    # boucle mois par mois
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
    
        # ---- extraire hourly pour le mois: TRACC (obs) et modèle (csv) ----
        obs_hourly = obs_mois_all[mois_num - 1] if len(obs_mois_all) >= mois_num else np.array([])
        idx0 = sum(heures_par_mois[:mois_num - 1])
        idx1 = sum(heures_par_mois[:mois_num])
        model_hourly = model_values[idx0:idx1]
    
        # ---- calculer stats journalières ----
        obs_tn, obs_tm, obs_tx = daily_stats_from_hourly(obs_hourly)
        mod_tn, mod_tm, mod_tx = daily_stats_from_hourly(model_hourly)
    
        # Si pas de données, passer
        if obs_tn.size == 0 or mod_tn.size == 0:
            st.write(f"{mois} — données insuffisantes pour calculer les statistiques journalières.")
            continue
    
        # ---- préparer CDFs (percentiles des séries journalières) ----
        obs_tn_cdf = np.percentile(obs_tn, pct_for_cdf)
        mod_tn_cdf = np.percentile(mod_tn, pct_for_cdf)
        obs_tm_cdf = np.percentile(obs_tm, pct_for_cdf)
        mod_tm_cdf = np.percentile(mod_tm, pct_for_cdf)
        obs_tx_cdf = np.percentile(obs_tx, pct_for_cdf)
        mod_tx_cdf = np.percentile(mod_tx, pct_for_cdf)
    
        # ---- tracé : un seul graphique regroupant Tn / Tmoy / Tx ----
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Couleurs cohérentes pour chaque variable
        colors = {
            "Tn": "cyan",
            "Tm": "white",
            "Tx": "red"
        }
        
        # Tn
        ax.plot(pct_for_cdf, mod_tx_cdf, linestyle="-", linewidth=2,
                label="Modèle Tx", color=colors["Tx"])
        ax.plot(pct_for_cdf, mod_tm_cdf, linestyle="-", linewidth=2,
                label="Modèle Tmoy", color=colors["Tm"])
        ax.plot(pct_for_cdf, mod_tn_cdf, linestyle="-", linewidth=2,
                label="Modèle Tn", color=colors["Tn"])

    
        ax.plot(pct_for_cdf, obs_tx_cdf, linestyle="--", linewidth=1.7,
                label="TRACC Tx", color=colors["Tx"])
        ax.plot(pct_for_cdf, obs_tm_cdf, linestyle="--", linewidth=1.7,
                label="TRACC Tmoy", color=colors["Tm"])
        ax.plot(pct_for_cdf, obs_tn_cdf, linestyle="--", linewidth=1.7,
                label="TRACC Tn", color=colors["Tn"])
        
        # Mise en forme
        ax.set_title(f"{mois} — CDF Tn / Tmoy / Tx (Modèle vs TRACC +{scenario_sel}/{ville_sel})", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")
        
        st.pyplot(fig)
        plt.close(fig)

        def pct_table_values(arr, pct_list):
            return [np.percentile(arr, p) for p in pct_list]
    
        tab = pd.DataFrame({
            "Percentile": [f"P{p}" for p in pct_table],
            "TRACC_Tn": np.round(pct_table_values(obs_tn, pct_table), 2),
            "Mod_Tn":   np.round(pct_table_values(mod_tn, pct_table), 2),
            "TRACC_Tm": np.round(pct_table_values(obs_tm, pct_table), 2),
            "Mod_Tm":   np.round(pct_table_values(mod_tm, pct_table), 2),
            "TRACC_Tx": np.round(pct_table_values(obs_tx, pct_table), 2),
            "Mod_Tx":   np.round(pct_table_values(mod_tx, pct_table), 2),
        })
    
        st.write(f"{mois} — Table des percentiles journaliers (Tn / Tmoy / Tx)")
    
        tab = pd.DataFrame(tab)
        num_cols = tab.select_dtypes(include=[np.number]).columns
        tab[num_cols] = tab[num_cols].apply(pd.to_numeric, errors="coerce")
        styler = tab.style.format({col: "{:.2f}" for col in num_cols})
    
        st.dataframe(styler, hide_index=True)


    # ======================================
    #  COURBES DES PERCENTILES PAR MOIS
    # ======================================
    st.subheader("Évolution mensuelle des percentiles (Modèle vs TRACC)")

    df_percentiles_all = []
    percentiles_list2 = [10,50,90]
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
    
        # Observations
        obs_vals = obs_mois_all[mois_num-1]
    
        # Modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_vals = model_values[idx0:idx1]

        
        # Ajout des percentiles
        for p in percentiles_list2:
            df_percentiles_all.append({
                "Mois": mois,
                "Percentile": f"P{p}",
                "Obs": np.percentile(obs_vals, p),
                "Mod": np.percentile(mod_vals, p)
            })

    # Table ordonnée pour faciliter les tracés
    df_percentiles_ordered = (
        pd.DataFrame(df_percentiles_all)
        .assign(Pnum=lambda d: d["Percentile"].str.extract("(\d+)").astype(int))
        .sort_values(["Pnum", "Mois"])
    )
    
    # Construction du graphique par percentile
    fig, ax = plt.subplots(figsize=(14,5))
    colors_perc = ["darkcyan", "khaki", "firebrick"]
    i=0
    for p in percentiles_list2:
        dfp = df_percentiles_ordered[df_percentiles_ordered["Pnum"] == p]
        # TRACC : ligne pointillée
        ax.plot(
            dfp["Mois"], dfp["Obs"],
            linestyle="--", label=f"TRACC P{p}", color=colors_perc[i]
        )
        # Modèle : ligne pleinne
        ax.plot(
            dfp["Mois"], dfp["Mod"],
            linestyle="-", label=f"Modèle P{p}", color=colors_perc[i]
        )
        i+=1
    
    ax.set_title(f"Percentiles {percentiles_list} – Modèle vs TRACC +{scenario_sel}/{ville_sel}")
    ax.set_ylabel("Température (°C)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=2, facecolor="black")
    st.pyplot(fig)
    plt.close(fig)


    # -------- Graphiques CDF et percentiles --------
    st.subheader("Fonctions de répartition mensuelles (CDF)")
    df_percentiles_all = []
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        obs_mois = obs_mois_all[mois_num-1]
        mod_mois = model_values[sum(heures_par_mois[:mois_num-1]):sum(heures_par_mois[:mois_num])]
        obs_percentiles_100 = np.percentile(obs_mois, np.linspace(0, 100, 100))
        mod_percentiles_100 = np.percentile(mod_mois, np.linspace(0, 100, 100))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.linspace(0, 100, 100), mod_percentiles_100, label="Modèle", color="red")
        ax.plot(np.linspace(0, 100, 100), obs_percentiles_100, label=f"TRACC +{scenario_sel}/{ville_sel}", color="blue")
        ax.set_title(f"{mois} - Fonction de répartition", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")
        st.pyplot(fig)
        plt.close(fig)

        obs_p = np.percentile(obs_mois, percentiles_list)
        mod_p = np.percentile(mod_mois, percentiles_list)
        df_p = pd.DataFrame({
            "Percentile": [f"P{p}" for p in percentiles_list],
            f"TRACC +{scenario_sel}/{ville_sel}": obs_p,
            "Modèle": mod_p
        }).round(2)
        st.write(f"{mois} - Percentiles")
        st.dataframe(df_p, hide_index=True)

        for i, p in enumerate(percentiles_list):
            df_percentiles_all.append({
                "Mois": mois,
                "Percentile": f"P{p}",
                "Obs": obs_p[i],
                "Mod": mod_p[i]
            })

    st.subheader(f"Bilan modèle vs TRACC +{scenario_sel}/{ville_sel} (Modèle - TRACC)") 
    # Création du DataFrame
    df_bilan = pd.DataFrame(df_percentiles_all).round(2)
    df_bilan["Ecart"] = df_bilan["Mod"] - df_bilan["Obs"]
    # Extraire le numéro du percentile (5, 25, ...) pour imposer l'ordre
    df_bilan["Percentile_num"] = df_bilan["Percentile"].str.extract("(\d+)").astype(int)
    # Imposer l'ordre des percentiles
    df_bilan["Percentile"] = pd.Categorical(df_bilan["Percentile"], 
                                            categories=[f"P{p}" for p in percentiles_list], 
                                            ordered=True)
    # Pivot pour affichage
    df_bilan_pivot = df_bilan.pivot(index="Percentile", columns="Mois", values="Ecart").round(2)
    # Affichage stylé avec couleurs selon l'écart
    st.dataframe(
        df_bilan_pivot.style
        .background_gradient(cmap="jet", vmin=vminT, vmax=vmaxT)
        .format("{:.2f}")
    )
    # -------- Section multi-scénarios pour la ville --------
    st.subheader(f"Comparaison multi-scénarios pour {ville_sel}")

    
    df_percentiles_scenarios = []
    for scenario in scenarios:
        nc_file = os.path.join(base_folder, scenario, f"{ville_sel}.nc")
        ds = xr.open_dataset(nc_file, decode_times=True)
        temps = ds["T2m"].to_series().values
        start_idx = 0
        for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
            mois = mois_noms[mois_num]
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
    st.subheader("CDF comparatif des 6 scénarios")
    
    scenario_pairs = [("2", "2_VC"), ("2-7", "2-7_VC"), ("4", "4_VC")]
    colors = ["green", "orange", "magenta"]  # couleur par paire
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
    
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_ylim(-5, 45)
    
        # ------------------------------
        # EXTRACTION MODELE CSV (en blanc)
        # ------------------------------
        idx0 = sum(heures_par_mois[:mois_num - 1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_mois_csv = model_values[idx0:idx1]
    
        cdf_model = np.percentile(mod_mois_csv, np.linspace(0, 100, 100))
    
        ax.plot(
            np.linspace(0, 100, 100),
            cdf_model,
            label="Modèle",
            color="white",
            linewidth=2,
            linestyle="-"
        )
    
        # ------------------------------
        # COURBES DES SCÉNARIOS
        # ------------------------------
        for i, (sc1, sc2) in enumerate(scenario_pairs):
            ax.set_ylim(-5, 45)
            color = colors[i]
    
            # ---- Scénario 1 (trait plein) ----
            nc_file = os.path.join(base_folder, sc1, f"{ville_sel}.nc")
            ds = xr.open_dataset(nc_file, decode_times=True)
            temp = ds["T2m"].to_series().values
            mod_mois = temp[idx0:idx1]
            cdf_values = np.percentile(mod_mois, np.linspace(0, 100, 100))
    
            ax.plot(
                np.linspace(0, 100, 100),
                cdf_values,
                label=f"{sc1}",
                color=color,
                linestyle="-"
            )
    
            # ---- Scénario 2 (pointillé) ----
            nc_file = os.path.join(base_folder, sc2, f"{ville_sel}.nc")
            ds = xr.open_dataset(nc_file, decode_times=True)
            temp = ds["T2m"].to_series().values
            mod_mois = temp[idx0:idx1]
            cdf_values = np.percentile(mod_mois, np.linspace(0, 100, 100))
    
            ax.plot(
                np.linspace(0, 100, 100),
                cdf_values,
                label=f"{sc2}",
                color=color,
                linestyle="--"
            )
    
        # ------------------------------
        # Mise en forme
        # ------------------------------
        ax.set_title(f"{mois} - CDF comparatif par scénario", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")
    
        st.pyplot(fig)
        plt.close(fig)

    # -------- Heatmap des écarts des percentiles par mois et scénario --------
    st.subheader(f"Ecarts des percentiles (Modèle - Scénarios TRACC)")
    
    # Création du dictionnaire de référence Modèle
    ref_model = {}
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        obs_mois = obs_mois_all[mois_num-1]
        mod_mois = model_values[sum(heures_par_mois[:mois_num-1]):sum(heures_par_mois[:mois_num])]
        for i, p in enumerate(percentiles_list):
            ref_model[(mois, f"P{p}")] = np.percentile(mod_mois, p)
    
    for p in percentiles_list:
        df_ecart = df_scenarios[df_scenarios["Percentile"] == f"P{p}"].copy()
        df_ecart["Ecart"] = -df_ecart.apply(lambda row: row["Valeur"] - ref_model[(row["Mois"], f"P{p}")], axis=1)
        df_ecart["Ecart"] = df_ecart["Ecart"].round(2).astype(float)
        df_pivot = df_ecart.pivot(index="Scénario", columns="Mois", values="Ecart").round(2)
        st.write(f"Percentile {p} : Modèle - TRACC/{ville_sel}")
        st.dataframe(df_pivot.style.background_gradient(cmap="jet", vmin=vminT, vmax=vmaxT).format("{:.2f}"))

