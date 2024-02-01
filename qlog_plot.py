#!/usr/bin/env python3
"""Plotting using qlog (QUIC logging format) files.

This script processes qlog files and generates visualizations based on the data.
"""

import json
import os
import time
import logging
import traceback
import argparse
import multiprocessing
from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import pandas as pd
import seaborn as sns


class QlogFormat(Enum):
    SQLOG = '.sqlog'
    QLOG = '.qlog'


def extract_data(qlog_file):
    """
    Extract data from a qlog file and convert it to DataFrames.

    :param qlog_file: Name of the qlog file to be processed.
    :return: A tuple of DataFrames (df_packets, df_metrics, df_offsets, df_datagram).
    """
    if qlog_file.endswith(QlogFormat.QLOG.value):
        packets_list, metrics_list, offsets_list, datagram_list = parse_picoquic_log(qlog_file)
    elif qlog_file.endswith(QlogFormat.SQLOG.value):
        packets_list, metrics_list, offsets_list, datagram_list = parse_quiche_log(qlog_file)
    else:
        logging.error("The qlog file format is not supported!")
        return None, None, None, None

    return create_dataframes(packets_list, metrics_list, offsets_list, datagram_list)


def create_dataframes(packets_list, metrics_list, offsets_list, datagram_list):
    """
    Convert lists of dictionaries to Pandas DataFrames.

    :param packets_list: List of packet dictionaries.
    :param metrics_list: List of metrics dictionaries.
    :param offsets_list: List of offsets dictionaries.
    :param datagram_list: List of datagram dictionaries.
    :return: A tuple of DataFrames (df_packets, df_metrics, df_offsets, df_datagram).
    """
    try:
        df_packets = pd.DataFrame(packets_list)
        df_metrics = pd.DataFrame(metrics_list)
        df_offsets = pd.DataFrame(offsets_list)
        df_datagram = pd.DataFrame(datagram_list)

        if 'packet_number' in df_packets.columns:
            df_packets['duplicate'] = df_packets.duplicated(subset=['packet_number'])
            df_packets['packet_size_cumsum'] = df_packets.packet_size.cumsum()

        if 'offset' in df_offsets.columns:
            df_offsets['duplicate'] = df_offsets.duplicated(subset=['offset'])

    except Exception as e:
        logging.error(f"Error creating DataFrames: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    return df_packets, df_metrics, df_offsets, df_datagram


def parse_quiche_log(qlog_file):
    """
    Parse a quiche log file and extract relevant data.

    :param qlog_file: Name of the qlog file to be processed.
    :return: Tuple containing lists for packets, metrics, offsets, and datagrams.
    """

    packets_list = []
    metrics_list = []
    offsets_list = []
    datagram_list = []

    try:
        with open(qlog_file, 'r') as json_file:
            content = json_file.read()

            # split the input data into individual JSON texts
            # \u001E is the ASCII Record Separator (RS) character
            json_objects = content.split('\u001E')

            # Extract role from the first JSON object
            first_json_object = json_objects[1]
            if first_json_object:
                first_json_data = json.loads(first_json_object)
                role = first_json_data.get('title', None)

            packet_direction = get_packet_direction(role)

            # Process remaining JSON objects
            for json_object in json_objects:
                json_object = json_object.strip()   # remove the line feed at the end of the json_text
                if json_object:  # check the string is not empty
                    process_quiche_json_object(json_object, packet_direction, packets_list,
                                               metrics_list, offsets_list, datagram_list)

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON for file {qlog_file}: {e}")
        return [], [], [], []

    return packets_list, metrics_list, offsets_list, datagram_list


def process_quiche_json_object(json_object, packet_direction, packets_list, metrics_list, offsets_list, datagram_list):
    """
    Process a single JSON object from the quiche qlog file.

    :param json_object: Single JSON object from the qlog file.
    :param packets_list: List to store packet data.
    :param metrics_list: List to store metrics data.
    :param offsets_list: List to store offsets data.
    :param datagram_list: List to store datagram data.
    """
    try:
        json_seq = json.loads(json_object)

        if json_seq.get('name') == f'transport:{packet_direction}':
            packet_dict = {'time': json_seq['time'] * 1000,
                           'packet_number': json_seq['data']['header']['packet_number'],
                           'packet_size': json_seq['data']['raw']['length']}
            packets_list.append(packet_dict)

            frames = json_seq['data']['frames']
            for frame in frames:
                if frame['frame_type'] == 'stream':
                    offset_dict = {'time': json_seq['time'] * 1000,
                                   'offset': frame['offset'],
                                   'length': frame['length'],
                                   'goodput': None,
                                   'goodput_no_gaps': None}
                    offsets_list.append(offset_dict)

            datagram_dict = {'time': json_seq['time'] * 1000,
                             'length': json_seq['data']['raw']['length'],
                             'throughput': None}
            datagram_list.append(datagram_dict)

        if json_seq.get('name') == 'recovery:metrics_updated':
            for key in json_seq['data']:
                if key in ["min_rtt",
                           "smoothed_rtt",
                           "latest_rtt",
                           "rtt_variance"]:
                    value = json_seq['data'][key] * 1000
                elif key == 'pacing_rate':
                    value = json_seq['data'][key] * 8
                else:
                    value = json_seq['data'][key]

                metric_dict = {'time': json_seq['time'] * 1000,
                                   'key': key,
                                   'value': value}
                metrics_list.append(metric_dict)
    except json.JSONDecodeError:
        logging.warning(f"Skipping malformed JSON object: {json_object[:100]}...")


def parse_picoquic_log(qlog_file):
    """
    Parse a picoquic log file and extract relevant data.

    :param qlog_file: Name of the qlog file to be processed.
    :return: Tuple containing lists for packets, metrics, offsets, and datagrams.
    """
    packets_list = []
    metrics_list = []
    offsets_list = []
    datagram_list = []

    try:
        with open(os.path.join(qlog_file), 'r') as json_file:
            json_file_load = json.load(json_file)
            events = json_file_load["traces"][0]["events"]
            role = json_file_load['traces'][0]['vantage_point']['type']
            packet_direction = get_packet_direction(role)

            for event in events:
                process_picoquic_event(event, packet_direction, packets_list, metrics_list, offsets_list, datagram_list)

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON for file {qlog_file}: {e}")
        return [], [], [], []

    return packets_list, metrics_list, offsets_list, datagram_list


def get_packet_direction(role):
    """
    Determine the packet direction based on the role.

    :param role: The role (server or client) from the qlog file.
    :return: The packet direction as a string.
    """
    if role == 'server' or role == "quiche-server qlog":
        return 'packet_sent'
    elif role == 'client' or role == "quiche-client qlog":
        return 'packet_received'
    else:
        logging.warning(f"Role for qlog file is not realized: {role}")
        return None


def process_picoquic_event(event, packet_direction, packets_list, metrics_list, offsets_list, datagram_list):
    """
    Process a single event from the qlog file.

    :param event: Single qlog event.
    :param packet_direction: The packet direction (sent or received).
    :param packets_list: List to store packet data.
    :param metrics_list: List to store metrics data.
    :param offsets_list: List to store offsets data.
    :param datagram_list: List to store datagram data.
    """
    if event[1] == 'transport' and event[2] == packet_direction:
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

        datagram_dict = {'time': event[0],
                         'length': event[3]['header']['packet_size'],
                         'throughput': None}
        datagram_list.append(datagram_dict)

    if event[1] == 'recovery' and event[2] == 'metrics_updated':
        for key in event[3]:
            metric_dict = {'time': event[0],
                           'key': key,
                           'value': event[3][key]}
            metrics_list.append(metric_dict)


def get_time_window_size(last_time):
    return last_time / 25


def calculate_throughput_goodput(df, metric):
    try:
        if df.empty:
            raise ValueError("DataFrame is empty")

        if metric == 'goodput':
            df = df[df['duplicate'] == False]

        last = df['time'].iloc[-1]
        time_window_size = get_time_window_size(float(last))

        interval_start = df['time'].iloc[0]
        interval_end = interval_start + time_window_size
        while interval_end <= last:
            interval = df[(df['time'] >= interval_start)
                          & (df['time'] < interval_end)]
            if not interval.empty:
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

    except Exception as e:
        logging.error(f"Error in calculate_throughput_goodput: {e}")
        return pd.DataFrame()


def megabyte_per_sec_to_megabit_per_sec(mb_per_sec):
    return mb_per_sec * 8


def plot_figures(df_packets, df_metrics, df_offsets, df_datagram, qlog_file):
    sns.set()
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = [
    #     'Times New Roman'] + plt.rcParams['font.serif']
    # This will change the default font size for all text
    plt.rcParams['font.size'] = 10
    qlog_file, _ = os.path.splitext(qlog_file)
    
    # Extract the base name without the extension
    qlog_file_name = os.path.basename(qlog_file)
    
    font_size = 10
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
    line_0_off, = ax[0].plot(df_offsets[df_offsets['duplicate'] == False]['time'] /
                             1e6, df_offsets[df_offsets['duplicate'] == False]['offset'] /
                             MB, '.', markersize=1, label="offset")
    line_0_re, = ax[0].plot(df_offsets[df_offsets['duplicate']]['time'] / 1e6,
                            df_offsets[df_offsets['duplicate']]['offset'] / MB,
                            '.', markersize=1, label="offset retransmitted")
    line_0_pkt, = ax[0].plot(df_packets['time'] / 1e6,
                             df_packets['packet_size_cumsum'] / MB,
                             '.', markersize=1, label="cumulative data size")
    ax[0].legend(handles=[line_0_off, line_0_re, line_0_pkt],
                 markerscale=10, fontsize=font_size)
    ax[0].set_ylabel('offset [MB]', fontsize=font_size)

    # pacing rate
    pacing_rate_data = df_metrics[df_metrics['key'] == "pacing_rate"]
    line_1_pacing, = ax[1].plot(pacing_rate_data['time'] /
                                1e6, pacing_rate_data['value'] /
                                MB, '.', markersize=1, label="pacing_rate")
    ax[1].legend(handles=[line_1_pacing], markerscale=10, fontsize=font_size)
    ax[1].set_ylabel("pacing rate [Mbps]", fontsize=font_size)

    # cwnd, bytes_in_flight
    cwnd_data = df_metrics[df_metrics['key'].isin(
        ['cwnd', 'congestion_window'])]
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
        f"Data / Offset vs. Time \n{qlog_file_name}", fontsize=font_size)
    fig.tight_layout()
    return fig


def save_data_and_figures(
        df_packets,
        df_metrics,
        df_offsets,
        df_datagram,
        fig,
        qlog_file):
    qlog_file_withoutSuffix = qlog_file.removesuffix('.qlog').removesuffix('.sqlog')
    df_packets.to_csv(f"{qlog_file_withoutSuffix}_packets.csv", index=False)
    df_metrics.to_csv(f"{qlog_file_withoutSuffix}_metrics.csv", index=False)
    df_datagram.to_csv(f'{qlog_file_withoutSuffix}_datagram.csv', index=False)
    df_offsets.to_csv(f'{qlog_file_withoutSuffix}_offsets.csv', index=False)
    plt.savefig(f"{qlog_file_withoutSuffix}.pdf")
    plt.savefig(f"{qlog_file_withoutSuffix}.png", dpi=600)
    pass


def process_single_file(qlog_file):
    """
    Process a single log file by extracting data, calculating metrics,
    plotting figures, and saving the results.

    :param qlog_file: Path and name of the log file to be processed.
    :return: Tuple containing the log file name and processing time or None in case of error.
    """
    try:
        logging.info(f"Processing file: {qlog_file}")
        start_time = time.time()
        df_packets, df_metrics, df_offsets, df_datagram = extract_data(qlog_file)
        df_datagram, df_offsets = process_data(df_datagram, df_offsets)

        fig = plot_figures(
            df_packets,
            df_metrics,
            df_offsets,
            df_datagram,
            qlog_file)
        save_data_and_figures(df_packets, df_metrics, df_offsets, df_datagram, fig, qlog_file)

        elapsed_time = time.time() - start_time
        return qlog_file, elapsed_time
    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        logging.error(f"Error processing {qlog_file}: {e}")
        return qlog_file, None


def process_data(df_datagram, df_offsets):
    """
    Process the data frames to calculate throughput and goodput.

    :param df_datagram: Data frame containing datagram information.
    :param df_offsets: Data frame containing offsets information.
    :return: Processed data frames.
    """
    try:
        df_datagram_processed = calculate_throughput_goodput(df_datagram, 'throughput')
        df_offsets[df_offsets['duplicate'] == False] = calculate_throughput_goodput(df_offsets, 'goodput')

        return df_datagram_processed, df_offsets

    except Exception as e:
        logging.error(f"Error in processing data frames: {e}")
        # Return original data frames in case of error
        return df_datagram, df_offsets


def main():
    """
    Processes qlog files in a specified directory.
    Can process a single file or all files in a directory.
    """
    parser = argparse.ArgumentParser(
        description='Process qlog files and generate visualizations.')
    parser.add_argument(
        'file',
        nargs='+',
        type=str,
        help='List of qlog files to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Configure logging based on the debug mode
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Expected file format: filename.[QUIC logging format]")

    start_time_total = time.time()

    process_files(args.file)

    logging.info(f"Total run time: {time.time() - start_time_total}")


def process_files(qlog_files):
    """
    Process all valid qlog files in the specified directory.

    :param qlog_files: List of qlog files, not checked for validity yet.
    """
    valid_qlog_files = [f for f in qlog_files if is_valid_file(f)]

    if not valid_qlog_files:
        logging.info("No valid qlog files found in the directory.")
        return

    logging.info(f"Processing following valid qlog files: {valid_qlog_files}")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(process_single_file, valid_qlog_files)
    logging.info(f"Processed {len(results)} files.")


def is_valid_file(filename):
    """
    Check if the file exists and has a valid qlog file suffix (qlog or sqlog).

    :param filename: The name of the file to check.
    :return: True if the file has a valid qlog format, False otherwise.
    """
    if not os.path.exists(filename):
        logging.error(f"File not found: {filename}")
        return False

    return any(filename.endswith(fmt.value) for fmt in QlogFormat)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.debug(traceback.format_exc())  # Logs the full stack trace at debug level
