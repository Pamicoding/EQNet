import os
import logging
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def convert_utc_datetime(utc_datetime_str):
    """
    Converts a UTC datetime string into a datetime object.
    Modify this function based on the format of your UTCDateTime strings.
    """
    return datetime.strptime(utc_datetime_str, '%Y-%m-%dT%H:%M:%S.%f')

def process_file(file):
    event_window_id = file.stem
    with open(file, 'r') as r:
        reader = csv.reader(r)
        header = next(reader)
        data = []
        for line in reader:
            line.insert(0, event_window_id)
            data.append(line)
    data.sort(key=lambda row: row[-1])
    return header, data
def concat_picks(gamma_picks_dir, all_picks):
    files = list(gamma_picks_dir.rglob('*csv'))
    headers_written = False

    with ThreadPoolExecutor(max_workers=40) as executor:
        results = list(executor.map(process_file, files)) # using map will indeed wait for all threads

    with open(all_picks, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for header, data in results:
            if not headers_written:
                writer.writerow(header)
                headers_written = True
            writer.writerows(data)
    return all_picks

def gamma_reorder(ori_csv, reorder_csv):
    
    with open(ori_csv, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = list(reader)
    data.sort(key=lambda row: convert_utc_datetime(row[1])) # second column is time
    
    with open(reorder_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header) 
        writer.writerows(data) 

def gamma_chunk_split(split_dir, reorder_csv):
    split_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(pd.read_csv(reorder_csv, chunksize=4000)):
        chunk.to_csv(split_dir / f'gamma_events_{i}.csv',index=False)

def transform(index, split_dir, gamma_picks, output_dir):
    gamma_events = split_dir / f'gamma_events_{index}.csv'
    logging.info(f'we are in gamma_events_{index}') 
    with open(gamma_events,'r') as f:
        event_lines = f.readlines()
    with open(gamma_picks,'r') as picks_read:
        picks_lines = picks_read.readlines()
    output_file = output_dir / f'gamma_new_{index}.dat_ch'
    with open(output_file,'a') as r:
        for line in event_lines[1:]:
            item = line.split(',')
            event_window_id = item[0]
            utc_time = datetime.strptime(item[1], '%Y-%m-%dT%H:%M:%S.%f')
            ymd = utc_time.strftime('%Y%m%d')
            hh = utc_time.hour
            mm = utc_time.minute
            ss = round(utc_time.second + utc_time.microsecond / 1000000, 2)
            lon_int = int(float(item[2]))
            lon_deg = (float(item[2]) - lon_int)*60
            lat_int = int(float(item[3]))
            lat_deg = (float(item[3]) - lat_int)*60
            depth = round(float(item[4]),2)
            event_index = item[-1]
            r.write(f"{ymd:>9}{hh:>2}{mm:>2}{ss:>6.2f}{lat_int:2}{lat_deg:0>5.2f}{lon_int:3}{lon_deg:0>5.2f}{depth:>6.2f}\n")
            for p_line in picks_lines[1:]:
                part = p_line.split(',')
                picks_index = part[-1]
                picks_window_id = part[0]
                if event_window_id == picks_window_id and event_index == picks_index:
                    phase = part[-2]
                    sta = part[1].zfill(4)
                    if sta[:1] == '0':
                        sta = f"A{sta[1:]}"
                    elif sta[:1] == '1':
                        sta = f"B{sta[1:]}"
                    pick_time = datetime.strptime(part[3], '%Y-%m-%dT%H:%M:%S.%f')
                    if mm == 59 and pick_time.minute == 0: 
                        wmm = int(60)
                    else:
                        wmm = pick_time.minute
                    wss = round(pick_time.second + pick_time.microsecond / 1000000, 2)
                    weight = '1.00'
                    if phase == 'P':
                        r.write(f"{' ':1}{sta:<4}{'0.0':>6}{'0':>4}{'0':>4}{wmm:>4}{wss:>6.2f}{'0.01':>5}{weight:>5}{'0.00':>6}{'0.00':>5}{'0.00':>5}\n")
                    else:
                        r.write(f"{' ':1}{sta:<4}{'0.0':>6}{'0':>4}{'0':>4}{wmm:>4}{'0.00':>6}{'0.00':>5}{'0.00':>5}{wss:>6.2f}{'0.01':>5}{weight:>5}\n")
    logging.info(f'gamma_event_{index} transform is done')                    

if __name__ == '__main__':
    ori_csv = Path('/home/patrick/Work/EQNet/tests/hualien_0403/catalog_gamma.csv')
    reorder_csv = Path('/home/patrick/Work/EQNet/tests/hualien_0403/gamma_order.csv')
    split_dir = Path('/home/patrick/Work/EQNet/tests/hualien_0403/split_dir')
    gamma_picks_dir = Path('/home/patrick/Work/EQNet/tests/hualien_0403/output/')
    gamma_picks = Path('/home/patrick/Work/EQNet/tests/hualien_0403/gamma_picks.csv')
    output_dir = Path('/home/patrick/Work/EQNet/tests/hualien_0403/h3dd_format')

    logging.basicConfig(filename='trans.log',level=logging.INFO,filemode='w')
    
    gamma_reorder(ori_csv, reorder_csv)
    gamma_chunk_split(split_dir, reorder_csv)
    
    check = True
    if not check:
        gamma_picks = concat_picks(gamma_picks_dir, gamma_picks)

    chunk_num = len(os.listdir(split_dir))
    index_list = np.arange(0, chunk_num)
    
    with ThreadPoolExecutor(max_workers=chunk_num) as executor:
        executor.map(transform, index_list, [split_dir] * chunk_num, [gamma_picks] * chunk_num, [output_dir] * chunk_num)

    





