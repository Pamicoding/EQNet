#%%
import pandas as pd
from pathlib import Path
import multiprocessing as mp

def process_file(file):
    event_window_id = Path(file).stem
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Modify the 'channel_id' column and rename it to 'station_id'
    df['channel_index'] = df['channel_index'].astype(str).str.zfill(4)
    df['channel_index'] = df['channel_index'].apply(lambda x: f"A{x[1:]}" if x[0] == '0' else f"B{x[1:]}")
    df['station_id'] = df['channel_index'].apply(lambda x: f"{event_window_id}.{x}")
    df = df.drop(columns=['channel_index'])
    
    # Return the modified DataFrame
    return df

def concat_picks(picks_dir, all_picks, num_workers=40):
    files = list(picks_dir.rglob('*.csv'))
    
    # Use multiprocessing to process files in parallel
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_file, files)
    
    # Concatenate all DataFrames
    concatenated_df = pd.concat(results)
    
    # Sort by 'station_id' column
    concatenated_df = concatenated_df.sort_values(by='station_id')
    
    # Write the concatenated DataFrame to a CSV file
    concatenated_df.to_csv(all_picks, index=False)
    
    print(f"Concatenated CSV saved to {all_picks}")
        
def concat_seis_das(seis, das, output_csv_path):
    df_seis = pd.read_csv(seis)
    df_das = pd.read_csv(das)

    df_seis = df_seis.loc[:, ['station_id', 'phase_index', 'phase_time', 'phase_score', 'phase_type']]
    result = pd.concat([df_seis, df_das], axis=0).reset_index(drop=True)
    result.to_csv(output_csv_path, index=False)
    print(f"Concatenated CSV saved to {output_csv_path}")


if __name__ == '__main__':
    raw_das_picks_dir = Path("/home/patrick/Work/Phasenet_DAS/For_Yen/20240403_results/picks_phasenet_das")
    phasenet_picks_dir = Path("/home/patrick/Work/EQNet/tests/hualien_0403/picks_phasenet_das")
    seismometer_picks = phasenet_picks_dir / "bat_cwb_sm_20240403.csv"
    das_picks = phasenet_picks_dir / "MiDAS_20240403_0_86100.csv"
    combined_csv = phasenet_picks_dir / "all_20240403_picks.csv"

    # concat das picks
    concat_picks(raw_das_picks_dir, das_picks)

    # concat das and seis
    concat_seis_das(seismometer_picks, das_picks, combined_csv)
# %%
