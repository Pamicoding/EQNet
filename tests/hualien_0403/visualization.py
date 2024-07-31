#%%
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import os
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import h5py
import pygmt

def pack(catalog_list: List[Path], name_list: List[str], station_path_list: List[Path]) -> Dict[int, Dict[str, Any]]:
    pack_dict = {}
    for i, (catalog_path, name, sta_path) in enumerate(zip(catalog_list, name_list, station_path_list)):
        pack_dict[i] = {"catalog": catalog_path, "name": name, "station": sta_path}
    return pack_dict

def plot_distribution(catalog: Dict[int, Dict[str, Any]]) -> None:
    params = {
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth": 3,
        "lines.markersize": 10,
        'image.origin': "lower",
        "figure.figsize": (4 * 2.5, 3 * 2.5),
        # 'axes.facecolor': 'white',
        # 'axes.edgecolor': 'white',
        # "figure.facecolor": "white",
        # "figure.edgecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
    matplotlib.rcParams.update(params)
    ls = LightSource(azdeg=0, altdeg=45)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12 * (max_lat-min_lat)/((max_lon-min_lon)*np.cos(np.deg2rad(min_lat)))), 
                            gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]})
    for i in catalog.keys():
        catalog_df = pd.read_csv(catalog[i]["catalog"])
        station_info = pd.read_csv(catalog[i]["station"])
        equip = catalog[i]["name"]

        # cmap = "viridis"
        catalog_df = catalog_df[(catalog_df["longitude"]>min_lon) & (catalog_df["longitude"]<max_lon) & 
                                (catalog_df["latitude"]>min_lat) & (catalog_df["latitude"]<max_lat) & 
                                (catalog_df["depth_km"] > min_depth) & (catalog_df["depth_km"] < max_depth)]

    # timestamp_das2 = events_das2["time"].apply(lambda x: datetime.fromisoformat(x).replace(tzinfo=timezone.utc).timestamp()).to_numpy()
    # timestamp_catalog2 = events_seis_catalog["time"].apply(lambda x: datetime.fromisoformat(x).timestamp()).to_numpy()
    # diff_time2 = timestamp_das2 - timestamp_catalog2[:, np.newaxis]
    # recall_idx2 = (np.abs(diff_time2) < 5).any(axis=0)
    # events_das2 = events_das2[recall_idx2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12 * (max_lat-min_lat)/((max_lon-min_lon)*np.cos(np.deg2rad(min_lat)))), 
                             gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]})

    region = [min_lon, max_lon, min_lat, max_lat]
    topo = pygmt.datasets.load_earth_relief(resolution="15s", region=region).to_numpy() / 1e3  # km
    x = np.linspace(min_lon, max_lon, topo.shape[1])
    y = np.linspace(min_lat, max_lat, topo.shape[0])
    dx, dy = 1, 1
    xgrid, ygrid = np.meshgrid(x, y)
    axes[0,0].pcolormesh(
        xgrid,
        ygrid,
        ls.hillshade(topo, vert_exag=10, dx=dx, dy=dy),
        vmin=-1,
        shading="gouraud",
        cmap="gray",
        alpha=1.0,
        antialiased=True,
        rasterized=True,
    )

    axes[0,0].scatter(
        catalog_df["longitude"],
        catalog_df["latitude"],
        s=5,
        c="blue",
        alpha=0.5,
        label=f"Seismometer ({catalog_df.shape()[0]})",
        rasterized=True,
    )
    axes[0,0].scatter(catalog_df["longitude"], catalog_df["latitude"], s=5, c="r", alpha=0.5, label=f"DAS ({len(catalog_df)})", rasterized=True)
    axes[0,0].autoscale(tight=True)
    xlim = axes[0,0].get_xlim()
    ylim = axes[0,0].get_ylim()
    # zlim = (0, 21)
    # axes[0,0].set_title(f"Number of events: {len(events_das2)}")
    axes[0, 0].scatter(das_info["longitude"], das_info["latitude"], s=2, c="k", marker=".", label=f"DAS cable ({len(das_info)})", alpha=0.5, rasterized=True)
    # axes[0, 0].axis("scaled")
    axes[0, 0].set_aspect(1.0/np.cos(np.deg2rad(min_latitude)))
    axes[0,0].set_xlim(xlim)
    axes[0,0].set_ylim(ylim)
    axes[0, 0].legend(markerscale=5)
    axes[0,0].set_ylabel("Latitude")

    axes[0,1].scatter(
        events_seis_catalog["depth_km"],
        events_seis_catalog["latitude"],
        s=5,
        c="blue",
        alpha=0.5,
        label="seismometer",
        rasterized=True,
    )
    axes[0,1].scatter(catalog_df["depth_km"], catalog_df["latitude"],  s=5, c="r", alpha=0.5, rasterized=True)
    axes[0,1].autoscale(tight=True)
    axes[0,1].set_ylim(ylim)
    axes[0,1].set_xlim([0, 45])
    axes[0,1].set_xlabel("Depth (km)")

    axes[1, 0].scatter(
        events_seis_catalog["longitude"],
        events_seis_catalog["depth_km"],
        s=5,
        c="blue",
        alpha=0.5,
        label="seismometer",
        rasterized=True,
    )
    axes[1,0].scatter(catalog_df["longitude"], catalog_df["depth_km"], s=5, c="r", alpha=0.5, rasterized=True)
    axes[1,0].autoscale(tight=True)    
    axes[1,0].set_xlim(xlim)
    axes[1,0].set_ylim([0, 45])
    axes[1,0].invert_yaxis()
    axes[1,0].set_ylabel("Depth (km)")
    axes[1,0].set_xlabel("Longitude")
    axes[1, 1].axis('off')

    fig.savefig(figure_path / figure_name)

def plot_diff() -> None:
    events_seis = pd.read_csv(seis_catalog)
    events_das = pd.read_csv(das_catalog)
    
    events_das_catalog = events_das[(events_das["longitude"]>min_lon) & (events_das["longitude"]<max_lon) & 
                             (events_das["latitude"]>min_lat) & (events_das["latitude"]<max_lat) & 
                             (events_das["depth_km"] > min_depth) & (events_das["depth_km"] < max_depth)]
    
    events_seis_catalog = events_seis[(events_seis["longitude"]>min_lon) & (events_seis["longitude"]<max_lon) & 
                                      (events_seis["latitude"]>min_lat) & (events_seis["latitude"]<max_lat) & 
                                      (events_seis["depth_km"] > min_depth) & (events_seis["depth_km"] < max_depth)]
    
    timestamp_das = events_das_catalog["time"].apply(lambda x: datetime.fromisoformat(x).timestamp()).to_numpy() # no need to add tzinfo(utc+8)
    timestamp_catalog = events_seis_catalog["time"].apply(lambda x: datetime.fromisoformat(x).timestamp()).to_numpy()
    diff_time = timestamp_das - timestamp_catalog[:, np.newaxis]
    # degree2km = 111.3
    # diff_lat = (events_das_catalog["latitude"].to_numpy() - events_seis_catalog["latitude"].to_numpy()[:, np.newaxis]) * degree2km
    # diff_lon = (events_das_catalog["longitude"].to_numpy() - events_seis_catalog["longitude"].to_numpy()[:, np.newaxis]) * np.cos(np.deg2rad(lat0)) * degree2km
    # diff_dep = events_das_catalog["depth_km"].to_numpy() - events_seis_catalog["depth_km"].to_numpy()[:, np.newaxis]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0, 0].hist(diff_time[np.abs(diff_time) < 5], range=[-2, 2], bins=31)#, facecolor="b", edgecolor="k", alpha=0.8)
    axes[0, 0].autoscale(enable=True, axis="x", tight=True)
    axes[0, 0].set_title(r"$\Delta$t (s)")
    fig.tight_layout()
    plt.savefig(figure_path / "diff.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    figure_path = Path('/home/patrick/Work/EQNet/tests/hualien_0403/compare')
    figure_path.mkdir(parents=True, exist_ok=True)
    figure_name = "whole.png"
    seis_station_info = Path("")
    all_station_info = Path("/home/patrick/Work/EQNet/tests/hualien_0403/station_all.csv")
    seis_catalog = Path("/home/patrick/Work/AutoQuake/GaMMA/results/Hualien_data/daily/20240403.csv")
    # das_catalog = Path("/home/patrick/Work/EQNet/tests/hualien_0403/gamma_order.csv")
    combined_catalog = Path("")

    catalog_list = [seis_catalog, combined_catalog]
    name_list = ["only seismometer", "seismometer + DAS"]
    station_list = [seis_station_info, all_station_info]
    catalog_dict = pack(catalog_list, name_list, station_list)

    min_lon = 119.7 # 119.7 121.0
    max_lon = 122.5 # 122.5 122.5
    min_lat = 21.7 # 21.7 23.5
    max_lat = 25.4 # 25.4 24.5
    min_depth = -0.1
    max_depth = 41.1

    # plot()
# %%
