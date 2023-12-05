#!/usr/bin/env python3
"""Plotting using qlog (QUIC logging format) files. 

This script processes qlog files and generates visualizations based on the data.
"""

import json
import os
import time
import argparse
import concurrent.futures

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import pandas as pd
import seaborn as sns


def packet_direction(log_file):
    log_file_name, log_file_format = os.path.splitext(log_file)
    if log_file_name.split('.')[-1] == 'server':
        direction = 'packet_sent'
    elif log_file_name.split('.')[-1] == 'client':
        direction = 'packet_received'
    else:
        print("Log file name is not correct!")
    return direction

def extract_data(log_dir, log_file):
    log_file_name, log_file_format = os.path.splitext(log_file)

    if log_file_format == '.qlog':
        packets_list, metrics_list, offsets_list, datagram_list = parse_picoquic_log(log_dir, log_file)
    elif log_file_format == '.sqlog':
        packets_list, metrics_list, offsets_list, datagram_list = parse_quiche_log(log_dir, log_file)
    else:
        print("The log file format is not supported!")

    # Convert list of dicts to DataFrame after loop ends
    df_packets = pd.DataFrame(packets_list)
    df_metrics = pd.DataFrame(metrics_list)
    df_offsets = pd.DataFrame(offsets_list)
    df_datagram = pd.DataFrame(datagram_list)
    df_packets['duplicate'] = df_packets.duplicated(subset=['packet_number'])
    df_packets['packet_size_cumsum'] = df_packets.packet_size.cumsum()
    df_offsets['duplicate'] = df_offsets.duplicated(subset=['offset'])
    return df_packets, df_metrics, df_offsets, df_datagram


def parse_quiche_log(log_dir, log_file):
    # parsing quiche log (*.sqlog)
    packets_list = []
    metrics_list = []
    offsets_list = []
    datagram_list = []
    direction = packet_direction(log_file)
    with open(f'{log_dir}{log_file}', 'r') as json_file_read:
        content = json_file_read.read()
        # split the input data into individual JSON texts
        # \u001E is the ASCII Record Separator (RS) character
        json_objects = content.split('\u001E')
        for json_object in json_objects:
            # remove the line feed at the end of the json_text
            json_object = json_object.strip()
            if json_object:  # check the string is not empty
                try:
                    json_seq = json.loads(json_object)
                    if json_seq.get('name') == f'transport:{direction}':
                        packet_dict = {'time': json_seq['time']*1000,
                                        'packet_number': json_seq['data']['header']['packet_number'],
                                        'packet_size': json_seq['data']['raw']['length']}
                        packets_list.append(packet_dict)

                    if json_seq.get('name') == f'transport:{direction}':
                        frames = json_seq['data']['frames']
                        for frame in frames:
                            if frame['frame_type'] == 'stream':
                                offset_dict = {'time': json_seq['time']*1000,
                                                'offset': frame['offset'],
                                                'length': frame['length'],
                                                'goodput': None,
                                                'goodput_no_gaps': None}
                                offsets_list.append(offset_dict)

                    if json_seq.get('name') == 'recovery:metrics_updated':
                        for key in json_seq['data']:
                            if key in ["min_rtt", "smoothed_rtt", "latest_rtt", "rtt_variance"]:
                                value = json_seq['data'][key] * 1000
                            elif key == 'pacing_rate':
                                value = json_seq['data'][key] * 8
                            else:
                                value = json_seq['data'][key]

                                metric_dict = {'time': json_seq['time']*1000,
                                               'key': key,
                                               'value': value
                                               }
                                metrics_list.append(metric_dict)
                    if json_seq.get('name') == f'transport:{direction}':
                        datagram_dict = {'time': json_seq['time']*1000,
                                             'length': json_seq['data']['raw']['length'],
                                             'throughput': None}
                        datagram_list.append(datagram_dict)
                except json.JSONDecodeError:
                    # print the first 100 characters
                    print(
                        f'Skipping malformed JSON object: {json_object}...')
    return packets_list, metrics_list, offsets_list, datagram_list

def parse_picoquic_log(log_dir, log_file):
    # parsing picoqioc log (*.qlog)
    packets_list = []
    metrics_list = []
    offsets_list = []
    datagram_list = []
    direction = packet_direction(log_file)
    with open(f'{log_dir}{log_file}', 'r') as json_file_read:
        json_file_load = json.load(json_file_read)
        events = json_file_load["traces"][0]["events"]
        for event in events:
            if event[1] == 'transport' and event[2] == direction:
                packet_dict = {'time': event[0],
                               'packet_number': event[3]['header']['packet_number'],
                               'packet_size': event[3]['header']['packet_size']}
                packets_list.append(packet_dict)
                frames = event[3]['frames']
                for frame in frames:
                    if frame['frame_type'] == 'stream':
                        offset_dict = {'time': event[0],
                                       'offset': frame['offset'],
                                       'length': frame['length'],
                                       'goodput': None,
                                       'goodput_no_gaps': None}
                        offsets_list.append(offset_dict)
            if event[1] == 'recovery' and event[2] == 'metrics_updated':
                for key in event[3]:
                    metric_dict = {'time': event[0],
                                   'key': key,
                                   'value': event[3][key]}
                    metrics_list.append(metric_dict)

            if event[1] == 'transport' and event[2] == direction:
                datagram_dict = {'time': event[0],
                                 'length': event[3]['header']['packet_size'],
                                 'throughput': None}
                datagram_list.append(datagram_dict)
    return packets_list, metrics_list, offsets_list, datagram_list

def get_time_window_size(last_time):
    return last_time / 25


def calculate_throughput_goodput(df, metric):
    if metric == 'goodput':
        # print('Starting goodput calculation...')
        df = df[df['duplicate'] == False]
    else:
        print('Starting throughput calculation...')
    last = df['time'].iloc[-1]
    time_window_size = get_time_window_size(float(last))
    # time_window_size = 10000000  # 1 sec
    interval_start = df['time'].iloc[0]
    interval_end = interval_start + time_window_size
    while interval_end <= last:
        # interval = df[(df['time'] >= interval_start) & (df['time'] < interval_end)]
        interval = df[(df['time'] >= interval_start)
                      & (df['time'] < interval_end)]
        if not interval['time'].empty:
            interval_sum = interval['length'].sum()
            metric_val = interval_sum / time_window_size
            metric_val = megabyte_per_sec_to_megabit_per_sec(metric_val)
            df.loc[df["time"] == interval['time'].iloc[0], metric] = metric_val
            # updated the next interval:
            interval_start = interval['time'].iloc[0] + 1
            interval_end = interval_start + time_window_size
        else:
            interval_start = interval_start + time_window_size
            interval_end = interval_start + time_window_size
    return df

def megabyte_per_sec_to_megabit_per_sec(mb_per_sec):
    return mb_per_sec * 8


def plot_figures(df_packets, df_metrics, df_offsets, df_datagram, log_file):
    sns.set()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [
        'Times New Roman'] + plt.rcParams['font.serif']
    # This will change the default font size for all text
    plt.rcParams['font.size'] = 12
    log_file_name, log_file_format = os.path.splitext(log_file)
    font_size = 12
    MB = 1000**2
    fig, ax = plt.subplots(5, 1, figsize=(4, 10), sharex=True)

    # generic
    for axis in ax:
        axis.grid(True)

    formatter1 = EngFormatter(
        places=0,
        sep="\N{THIN SPACE}")  # U+2009
    # time-offset/data
    ax[0].yaxis.set_major_formatter(formatter1)
    line_0_off, = ax[0].plot(df_offsets[df_offsets['duplicate'] == False]['time'] / 1e6,
                             df_offsets[df_offsets['duplicate']
                                        == False]['offset'] / MB,
                             '.', markersize=1, label="offset")
    line_0_re, = ax[0].plot(df_offsets[df_offsets['duplicate']]['time'] / 1e6,
                            df_offsets[df_offsets['duplicate']]['offset'] / MB,
                            '.', markersize=1, label="offset retransmitted")
    line_0_pkt, = ax[0].plot(df_packets['time'] / 1e6,
                             df_packets['packet_size_cumsum'] / MB,
                             '.', markersize=1, label="cumulative data size")
    ax[0].legend(handles=[line_0_off, line_0_re, line_0_pkt],
                 markerscale=10,  fontsize=font_size)
    ax[0].set_ylabel('offset [MB]', fontsize=font_size)

    

    # ax[0].set_title(f"Data / Offset vs. Time \ns:{server} | c:{client} | {cc} | {obj_size_str} | {itr}")

    # pacing rate
    pacing_rate_data = df_metrics[df_metrics['key'] == "pacing_rate"]
    line_1_pacing, = ax[1].plot(pacing_rate_data['time'] / 1e6, pacing_rate_data['value'] / MB,
                                '.', markersize=1, label="pacing_rate")
    ax[1].legend(handles=[line_1_pacing], markerscale=10, fontsize=font_size)
    ax[1].set_ylabel("pacing rate [Mbps]", fontsize=font_size)
    # pacing_rate: defined in picoquic in the below file line #1075
    # https://github.com/private-octopus/picoquic/blob/8a127902f11699542ae9bed10d2cffa7b02348e5/picoquic/picoquic_internal.h

    # cwnd, bytes_in_flight
    # cwnd_data = df_metrics.query('key == "cwnd"')
    cwnd_data = df_metrics[df_metrics['key'].isin(
        ['cwnd', 'congestion_window'])]
    # bytes_in_flight_data = df_metrics.query('key == "bytes_in_flight"')
    bytes_in_flight_data = df_metrics[df_metrics['key'] == "bytes_in_flight"]
    line_2_cwnd, = ax[2].plot(cwnd_data['time'] / 1e6,
                              cwnd_data['value'] / MB,
                              '.', markersize=1, label="cwnd")
    line_2_flight, = ax[2].plot(bytes_in_flight_data['time'] / 1e6,
                                bytes_in_flight_data['value'] / MB,
                                '.', markersize=1, label="bytes_in_flight")
    ax[2].legend(handles=[line_2_cwnd, line_2_flight],
                 markerscale=10, fontsize=font_size)
    ax[2].set_ylabel("metrics [MB]", fontsize=font_size)

    # rtt stuff
    smoothed_rtt_data = df_metrics[df_metrics['key'] == "smoothed_rtt"]
    latest_rtt_data = df_metrics[df_metrics['key'] == "latest_rtt"]
    min_rtt_data = df_metrics[df_metrics['key'] == "min_rtt"]
    line_3_smoothedrtt, = ax[3].plot(smoothed_rtt_data['time'] / 1e6,
                                     smoothed_rtt_data['value'] / 1e3,
                                     '.', markersize=1, label="smoothed_rtt")
    line_3_latestrtt, = ax[3].plot(latest_rtt_data['time'] / 1e6,
                                   latest_rtt_data['value'] / 1e3,
                                   '.', markersize=1, label="latest_rtt")
    line_3_minrtt, = ax[3].plot(min_rtt_data['time'] / 1e6,
                                min_rtt_data['value'] / 1e3,
                                '.', markersize=1, label="min_rtt")
    ax[3].legend(
        handles=[
            line_3_smoothedrtt,
            line_3_latestrtt,
            line_3_minrtt],
        markerscale=10, fontsize=font_size)
    ax[3].set_ylabel("RTT [ms]", fontsize=font_size)

    # throughput/goodput
    line_4_throughput, = ax[4].plot(df_datagram['time'] / 1e6,
                                    df_datagram['throughput'],
                                    'o-', markersize=1, label="throughput")
    line_4_goodput, = ax[4].plot(df_offsets[df_offsets['duplicate'] == False]['time'] / 1e6,
                                 df_offsets[df_offsets['duplicate']
                                            == False]['goodput'],
                                 '-.', markersize=1, label="goodput")
    ax[4].legend(handles=[line_4_throughput, line_4_goodput],
                 markerscale=10, fontsize=font_size)
    ax[4].set_ylabel("data rate [Mbps]", fontsize=font_size)
    ax[4].set_xlabel("Time [s]", fontsize=font_size)
    fig.align_ylabels(ax[:])
    
    fig.suptitle(
        f"Data / Offset vs. Time \n{log_file_name}", fontsize=font_size)
    fig.tight_layout()
    return fig


def save_data_and_figures(df_packets, df_metrics, df_offsets, df_datagram, fig, log_file, log_dir):
    log_file_name, log_file_format = os.path.splitext(log_file)
    df_packets.to_csv(f"{log_dir}/{log_file_name}_packets.csv", index=False)
    df_metrics.to_csv(f"{log_dir}/{log_file_name}_metrics.csv", index=False)
    df_datagram.to_csv(f'{log_dir}/{log_file_name}_datagram.csv', index=False)
    df_offsets.to_csv(f'{log_dir}/{log_file_name}_offsets.csv', index=False)
    plt.savefig(f"{log_dir}/{log_file_name}.pdf")
    plt.savefig(f"{log_dir}/{log_file_name}.png", dpi=600)
    pass


def process_file(log_dir, log_file):
    try:
        print(f"Processing file: {log_file}")
        start_time = time.time()
        df_packets, df_metrics, df_offsets, df_datagram = extract_data(
            log_dir, log_file)

        df_datagram = calculate_throughput_goodput(df_datagram, 'throughput')
        df_offsets[df_offsets['duplicate'] ==
                   False] = calculate_throughput_goodput(df_offsets, 'goodput')

        fig = plot_figures(df_packets, df_metrics,
                           df_offsets, df_datagram, log_file)
        save_data_and_figures(df_packets, df_metrics, df_offsets,
                              df_datagram, fig, log_file, log_dir)

        elapsed_time = time.time() - start_time
        return log_file, elapsed_time
    except json.JSONDecodeError as e:
        print(f"Error processing {log_file}: {e}")
        return log_file, None


def main():
    parser = argparse.ArgumentParser(description='Process qlog files and generate visualizations.')
    parser.add_argument('log_dir', type=str, help='Directory containing qlog files')
    parser.add_argument('--file', type=str, help='Specific qlog file to process', required=False)
    args = parser.parse_args()

    print("Expected file format: filename.[server|client].[sqlog|qlog]")
    start_time_total = time.time()

    if args.file:
        # Process only the specified file
        if is_valid_file(args.file):
            file_path = os.path.join(args.log_dir, args.file)
            if os.path.exists(file_path):
                print(f"Processing file: {file_path}")
                process_file(args.log_dir, args.file)
            else:
                print(f"File not found: {file_path}")
        else:
            print(f"Invalid file format: {args.file}")

    else:
        # Process all files in the directory
        list_log_files_in_dir = [f for f in os.listdir(args.log_dir) if is_valid_file(f)]
        total_files = len(list_log_files_in_dir)

        if total_files == 0:
            print("No valid files found in directory.")
        else:
            processed_files = 0
            print(f"Processing all valid files in directory: {args.log_dir}")

        with concurrent.futures.ProcessPoolExecutor() as executor:
        # The map method blocks until all results are returned
            for log_file, elapsed_time in executor.map(process_file, [args.log_dir]*total_files, list_log_files_in_dir):
                if elapsed_time is None:
                    print(f"Error processing {log_file}")
                else:
                    processed_files += 1
                    print(f"Processed {log_file} in {elapsed_time} seconds. ({processed_files}/{total_files} files completed.)")
    print(f"\nTotal run time: {time.time() - start_time_total}")


def is_valid_file(filename):
    parts = filename.split('.')
    return len(parts) >= 3 and parts[-2] in ['server', 'client'] and parts[-1] in ['sqlog', 'qlog']


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
