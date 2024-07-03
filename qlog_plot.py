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
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import pandas as pd
import seaborn as sns


class QlogFormat(Enum):
    SQLOG = '.sqlog'
    QLOG = '.qlog'


class BaseQlogFileParser:
    def __init__(self, qlog_file: str):
        self.qlog_file = qlog_file

    def extract_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("Must be implemented by subclasses")

    @staticmethod
    def create_dataframes(packets_list: List[dict], metrics_list: List[dict], offsets_list: List[dict], datagram_list: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            df_packets = pd.DataFrame(packets_list)
            df_metrics = pd.DataFrame(metrics_list)
            df_offsets = pd.DataFrame(offsets_list)
            df_datagram = pd.DataFrame(datagram_list)

            if 'packet_number' in df_packets.columns:
                df_packets['duplicate'] = df_packets.duplicated(
                    subset=['packet_number'])
                df_packets['packet_size_cumsum'] = df_packets['packet_size'].cumsum()

            if 'offset' in df_offsets.columns:
                df_offsets['duplicate'] = df_offsets.duplicated(subset=[
                                                                'offset'])

            return df_packets, df_metrics, df_offsets, df_datagram

        except Exception as e:
            logging.error(f"Error creating DataFrames: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


class QlogFileParser(BaseQlogFileParser):
    def extract_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        packets_list, metrics_list, offsets_list, datagram_list = [], [], [], []

        try:
            with open(self.qlog_file, 'r') as json_file:
                json_file_load = json.load(json_file)
                events = json_file_load["traces"][0]["events"]
                role = json_file_load['traces'][0]['vantage_point']['type']
                packet_direction = self.get_packet_direction(role)

                for event in events:
                    self.process_picoquic_event(
                        event, packet_direction, packets_list, metrics_list, offsets_list, datagram_list)

            return self.create_dataframes(packets_list, metrics_list, offsets_list, datagram_list)

        except json.JSONDecodeError as e:
            logging.error(
                f"Error decoding JSON for file {self.qlog_file}: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def process_picoquic_event(self, event: list, packet_direction: str, packets_list: List[dict], metrics_list: List[dict], offsets_list: List[dict], datagram_list: List[dict]):
        if event[1] == 'transport' and event[2] == packet_direction:
            packet_dict = {
                'time': event[0],
                'packet_number': event[3]['header']['packet_number'],
                'packet_size': event[3]['header']['packet_size']
            }
            packets_list.append(packet_dict)

            frames = event[3]['frames']
            for frame in frames:
                if frame['frame_type'] == 'stream':
                    offset_dict = {
                        'time': event[0],
                        'offset': frame['offset'],
                        'length': frame['length']
                    }
                    offsets_list.append(offset_dict)

            datagram_dict = {
                'time': event[0],
                'length': event[3]['header']['packet_size'],
                'throughput': None
            }
            datagram_list.append(datagram_dict)

        if event[1] == 'recovery' and event[2] == 'metrics_updated':
            for key in event[3]:
                metric_dict = {
                    'time': event[0],
                    'key': key,
                    'value': event[3][key]
                }
                metrics_list.append(metric_dict)

    def get_packet_direction(self, role: str) -> str:
        if role == 'server' or role == "quiche-server qlog":
            return 'packet_sent'
        elif role == 'client' or role == "quiche-client qlog":
            return 'packet_received'
        else:
            logging.warning(f"Role for qlog file is not realized: {role}")
            return None


class SQlogFileParser(BaseQlogFileParser):
    def extract_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        packets_list, metrics_list, offsets_list, datagram_list = [], [], [], []

        try:
            with open(self.qlog_file, 'r') as json_file:
                content = json_file.read()
                json_objects = content.split('\u001E')
                role = json.loads(json_objects[1]).get('title', None)
                packet_direction = self.get_packet_direction(role)

                for json_object in json_objects:
                    json_object = json_object.strip()
                    if json_object:
                        self.process_quiche_json_object(
                            json_object, packet_direction, packets_list, metrics_list, offsets_list, datagram_list)

            return self.create_dataframes(packets_list, metrics_list, offsets_list, datagram_list)

        except json.JSONDecodeError as e:
            logging.error(
                f"Error decoding JSON for file {self.qlog_file}: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def process_quiche_json_object(self, json_object: str, packet_direction: str, packets_list: List[dict], metrics_list: List[dict], offsets_list: List[dict], datagram_list: List[dict]):
        try:
            json_seq = json.loads(json_object)

            if json_seq.get('name') == f'transport:{packet_direction}':
                packet_dict = {
                    'time': json_seq['time'] * 1000,
                    'packet_number': json_seq['data']['header']['packet_number'],
                    'packet_size': json_seq['data']['raw']['length']
                }
                packets_list.append(packet_dict)

                frames = json_seq['data']['frames']
                for frame in frames:
                    if frame['frame_type'] == 'stream':
                        offset_dict = {
                            'time': json_seq['time'] * 1000,
                            'offset': frame['offset'],
                            'length': frame['length']
                        }
                        offsets_list.append(offset_dict)

                datagram_dict = {
                    'time': json_seq['time'] * 1000,
                    'length': json_seq['data']['raw']['length'],
                    'throughput': None
                }
                datagram_list.append(datagram_dict)

            if json_seq.get('name') == 'recovery:metrics_updated':
                for key in json_seq['data']:
                    value = json_seq['data'][key] * 1000 if key in [
                        "min_rtt", "smoothed_rtt", "latest_rtt", "rtt_variance"] else json_seq['data'][key]
                    if key == 'pacing_rate':
                        value *= 8

                    metric_dict = {
                        'time': json_seq['time'] * 1000,
                        'key': key,
                        'value': value
                    }
                    metrics_list.append(metric_dict)
        except json.JSONDecodeError:
            logging.warning(
                f"Skipping malformed JSON object: {json_object[:100]}...")

    def get_packet_direction(self, role: str) -> str:
        if role == 'server' or role == "quiche-server qlog":
            return 'packet_sent'
        elif role == 'client' or role == "quiche-client qlog":
            return 'packet_received'
        else:
            logging.warning(f"Role for qlog file is not realized: {role}")
            return None


class QlogDataProcessor:
    def __init__(self, df_packets: pd.DataFrame, df_metrics: pd.DataFrame, df_offsets: pd.DataFrame, df_datagram: pd.DataFrame, time_interval: float):
        self.df_packets = df_packets
        self.df_metrics = df_metrics
        self.df_offsets = df_offsets
        self.df_datagram = df_datagram
        self.time_interval = time_interval
        self.data_rate_df = pd.DataFrame()

    def calculate_throughput_and_goodput(self):
        try:
            # Convert time to datetime
            self.df_datagram['datetime'] = pd.to_datetime(
                self.df_datagram['time'], unit='us')
            self.df_offsets['datetime'] = pd.to_datetime(
                self.df_offsets['time'], unit='us')

            # Set datetime as index
            self.df_datagram.set_index('datetime', inplace=True)
            self.df_offsets.set_index('datetime', inplace=True)

            # Calculate rolling sum for throughput over 100ms window
            self.df_datagram['throughput'] = self.df_datagram['length'].rolling(
                f'{int(self.time_interval/1e3)}ms').sum().fillna(0).apply(self.byte_per_sec_to_bits_per_sec)

            # Calculate rolling sum for goodput over 100ms window, considering non-duplicate offsets
            df_goodput_non_dup = self.df_offsets[self.df_offsets['duplicate'] == False].copy(
            )
            df_goodput_non_dup.loc[:, 'goodput'] = df_goodput_non_dup['length'].rolling(
                f'{int(self.time_interval/1e3)}ms').sum().fillna(0).apply(self.byte_per_sec_to_bits_per_sec)

            # Resample to the specified time interval
            df_throughput_resampled = self.df_datagram['throughput'].resample(
                f'{int(self.time_interval/1e3)}ms').mean().fillna(0)
            df_goodput_resampled = df_goodput_non_dup['goodput'].resample(
                f'{int(self.time_interval/1e3)}ms').mean().fillna(0)

            self.data_rate_df = pd.concat(
                [df_throughput_resampled, df_goodput_resampled], axis=1).reset_index()
            self.data_rate_df.columns = ['datetime', 'throughput', 'goodput']

            # Convert datetime to seconds for plotting
            self.data_rate_df['start_interval'] = (
                self.data_rate_df['datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            self.data_rate_df['end_interval'] = self.data_rate_df['start_interval'] + \
                self.time_interval / 1e6
            self.data_rate_df.drop(columns=['datetime'], inplace=True)

            # Reorder columns
            self.data_rate_df = self.data_rate_df[[
                'start_interval', 'end_interval', 'throughput', 'goodput']]

        except Exception as e:
            logging.error(f"Error in calculate_throughput_and_goodput: {e}")

    @staticmethod
    def byte_per_sec_to_bits_per_sec(mb_per_sec: float) -> float:
        return mb_per_sec * 8  # Convert B/s to bits per second


class QlogPlotter:
    def __init__(self, df_packets: pd.DataFrame, df_metrics: pd.DataFrame, df_offsets: pd.DataFrame, data_rate_df: pd.DataFrame, qlog_file: str):
        self.df_packets = df_packets
        self.df_metrics = df_metrics
        self.df_offsets = df_offsets
        self.data_rate_df = data_rate_df
        self.qlog_file = qlog_file

    def plot_figures(self):
        sns.set()
        plt.rcParams['font.size'] = 10
        qlog_file_name = os.path.basename(os.path.splitext(self.qlog_file)[0])

        font_size = 10
        MB = 1000**2
        fig, ax = plt.subplots(5, 1, figsize=(4, 10), sharex=True)

        for axis in ax:
            axis.grid(True)

        formatter1 = EngFormatter(places=0, sep="\N{THIN SPACE}")  # U+2009
        ax[0].yaxis.set_major_formatter(formatter1)
        line_0_off, = ax[0].plot(self.df_offsets[self.df_offsets['duplicate'] == False]['time'] / 1e6,
                                 self.df_offsets[self.df_offsets['duplicate']
                                                 == False]['offset'] / MB,
                                 '.', markersize=1, label="offset")
        line_0_re, = ax[0].plot(self.df_offsets[self.df_offsets['duplicate']]['time'] / 1e6,
                                self.df_offsets[self.df_offsets['duplicate']
                                                ]['offset'] / MB,
                                '.', markersize=1, label="offset retransmitted")
        line_0_pkt, = ax[0].plot(self.df_packets['time'] / 1e6,
                                 self.df_packets['packet_size_cumsum'] /
                                 MB,
                                 '.', markersize=1, label="cumulative data size")
        ax[0].legend(handles=[line_0_off, line_0_re, line_0_pkt],
                     markerscale=10, fontsize=font_size)
        ax[0].set_ylabel('offset [MB]', fontsize=font_size)

        pacing_rate_data = self.df_metrics[self.df_metrics['key']
                                           == "pacing_rate"]
        line_1_pacing, = ax[1].plot(pacing_rate_data['time'] / 1e6,
                                    pacing_rate_data['value'] / MB,
                                    '.', markersize=1, label="pacing_rate")
        ax[1].legend(handles=[line_1_pacing],
                     markerscale=10, fontsize=font_size)
        ax[1].set_ylabel("pacing rate [Mbps]", fontsize=font_size)

        cwnd_data = self.df_metrics[self.df_metrics['key'].isin(
            ['cwnd', 'congestion_window'])]
        bytes_in_flight_data = self.df_metrics[self.df_metrics['key']
                                               == "bytes_in_flight"]
        line_2_cwnd, = ax[2].plot(cwnd_data['time'] / 1e6,
                                  cwnd_data['value'] / MB,
                                  '.', markersize=1, label="cwnd")
        line_2_flight, = ax[2].plot(bytes_in_flight_data['time'] / 1e6,
                                    bytes_in_flight_data['value'] / MB,
                                    '.', markersize=1, label="bytes_in_flight")
        ax[2].legend(handles=[line_2_cwnd, line_2_flight],
                     markerscale=10, fontsize=font_size)
        ax[2].set_ylabel("metrics [MB]", fontsize=font_size)

        smoothed_rtt_data = self.df_metrics[self.df_metrics['key']
                                            == "smoothed_rtt"]
        latest_rtt_data = self.df_metrics[self.df_metrics['key']
                                          == "latest_rtt"]
        min_rtt_data = self.df_metrics[self.df_metrics['key'] == "min_rtt"]
        line_3_smoothedrtt, = ax[3].plot(smoothed_rtt_data['time'] / 1e6,
                                         smoothed_rtt_data['value'] / 1e3,
                                         '.', markersize=1, label="smoothed_rtt")
        line_3_latestrtt, = ax[3].plot(latest_rtt_data['time'] / 1e6,
                                       latest_rtt_data['value'] / 1e3,
                                       '.', markersize=1, label="latest_rtt")
        line_3_minrtt, = ax[3].plot(min_rtt_data['time'] / 1e6,
                                    min_rtt_data['value'] / 1e3,
                                    '.', markersize=1, label="min_rtt")
        ax[3].legend(handles=[line_3_smoothedrtt, line_3_latestrtt, line_3_minrtt],
                     markerscale=10, fontsize=font_size)
        ax[3].set_ylabel("RTT [ms]", fontsize=font_size)

        line_4_throughput, = ax[4].plot(self.data_rate_df['start_interval'],
                                        # Convert bits to megabits
                                        self.data_rate_df['throughput'] / MB,
                                        '-', markersize=1, label="throughput")
        line_4_goodput, = ax[4].plot(self.data_rate_df['start_interval'],
                                     # Convert bits to megabits
                                     self.data_rate_df['goodput'] / MB,
                                     '--', markersize=1, label="goodput")
        ax[4].legend(handles=[line_4_throughput, line_4_goodput],
                     markerscale=10, fontsize=font_size)
        ax[4].set_ylabel("data rate [Mbps]", fontsize=font_size)
        ax[4].set_xlabel("Time [s]", fontsize=font_size)
        fig.align_ylabels(ax[:])

        fig.suptitle(
            f"Data / Offset vs. Time \n{qlog_file_name}", fontsize=font_size)
        fig.tight_layout()
        return fig

    def save_figures(self, fig):
        plt.savefig(f"{self.qlog_file}.pdf")
        plt.savefig(f"{self.qlog_file}.png", dpi=600)


class QlogProcessor:
    def __init__(self, qlog_file: str, time_interval: float):
        self.qlog_file = qlog_file
        self.time_interval = time_interval
        self.df_packets = pd.DataFrame()
        self.df_metrics = pd.DataFrame()
        self.df_offsets = pd.DataFrame()
        self.df_datagram = pd.DataFrame()
        self.data_rate_df = pd.DataFrame()

    def process_file(self) -> Tuple[str, Union[float, None]]:
        logging.info(f"Processing file: {self.qlog_file}")
        start_time = time.time()

        parser = self.get_parser()
        self.df_packets, self.df_metrics, self.df_offsets, self.df_datagram = parser.extract_data()

        if self.df_packets.empty and self.df_metrics.empty and self.df_offsets.empty and self.df_datagram.empty:
            return self.qlog_file, None

        data_processor = QlogDataProcessor(
            self.df_packets, self.df_metrics, self.df_offsets, self.df_datagram, self.time_interval)
        data_processor.calculate_throughput_and_goodput()
        self.data_rate_df = data_processor.data_rate_df

        plotter = QlogPlotter(self.df_packets, self.df_metrics,
                              self.df_offsets, self.data_rate_df, self.qlog_file)
        fig = plotter.plot_figures()
        self.save_data()
        plotter.save_figures(fig)

        elapsed_time = time.time() - start_time
        return self.qlog_file, elapsed_time

    def get_parser(self) -> BaseQlogFileParser:
        if self.qlog_file.endswith(QlogFormat.QLOG.value):
            return QlogFileParser(self.qlog_file)
        elif self.qlog_file.endswith(QlogFormat.SQLOG.value):
            return SQlogFileParser(self.qlog_file)
        else:
            raise ValueError(f"Unsupported qlog file format: {self.qlog_file}")

    def save_data(self):
        self.df_packets.to_csv(
            f"{self.qlog_file}.packets.csv", index=False)
        self.df_metrics.to_csv(
            f"{self.qlog_file}.metrics.csv", index=False)
        self.df_datagram.to_csv(
            f'{self.qlog_file}.datagram.csv', index=False)
        self.df_offsets.to_csv(
            f'{self.qlog_file}.offsets.csv', index=False)
        self.data_rate_df.columns = [
            'start_interval (s)', 'end_interval (s)', 'throughput (bps)', 'goodput (bps)']
        self.data_rate_df.to_csv(
            f'{self.qlog_file}.data_rate.csv', index=False)


def process_files(qlog_files: List[str], time_interval: float):
    """
    Process all valid qlog files in the specified directory.

    :param qlog_files: List of qlog files, not checked for validity yet.
    :param time_interval: Time window interval in microseconds.
    """
    valid_qlog_files = [f for f in qlog_files if is_valid_file(f)]

    if not valid_qlog_files:
        logging.info("No valid qlog files found in the directory.")
        return

    logging.info(f"Processing following valid qlog files: {valid_qlog_files}")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.starmap(process_single_file, [(
        f, time_interval) for f in valid_qlog_files])
    logging.info(f"Processed {len(results)} files.")


def process_single_file(qlog_file: str, time_interval: float) -> Tuple[str, Union[float, None]]:
    processor = QlogProcessor(qlog_file, time_interval)
    return processor.process_file()


def is_valid_file(filename: str) -> bool:
    """
    Check if the file exists and has a valid qlog file suffix (qlog, sqlog, or any other future format).

    :param filename: The name of the file to check.
    :return: True if the file has a valid qlog format, False otherwise.
    """
    if not os.path.exists(filename):
        logging.error(f"File not found: {filename}")
        return False

    return any(filename.endswith(fmt.value) for fmt in QlogFormat)


def main():
    """
    Processes qlog files in a specified directory.
    Can process a single file or all files in a directory.
    """
    parser = argparse.ArgumentParser(
        description='Process qlog files and generate visualizations.')
    parser.add_argument('file', nargs='+', type=str,
                        help='List of qlog files to process')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--interval', type=float, default=1000000.0,
                        help='Time window interval in microseconds (default: 1000000.0)')
    args = parser.parse_args()

    # Configure logging based on the debug mode
    if args.debug:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Expected file format: filename.[QUIC logging format]")

    start_time_total = time.time()

    process_files(args.file, args.interval)

    logging.info(f"Total run time: {time.time() - start_time_total} sec")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.debug(traceback.format_exc())
