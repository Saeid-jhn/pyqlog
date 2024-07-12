import pyshark
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os
import logging
import time
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from matplotlib.ticker import FuncFormatter


class PcapFileProcessor:
    def __init__(self, pcap_file: str, plot_seq: bool = False, filter_src_ip: Optional[str] = None,
                 filter_src_port: Optional[int] = None, filter_dst_ip: Optional[str] = None,
                 filter_dst_port: Optional[int] = None, stream_index: Optional[int] = None,
                 filter_tcp: bool = False, filter_quic: bool = False):
        self.pcap_file = pcap_file
        self.plot_seq = plot_seq
        self.filter_src_ip = filter_src_ip
        self.filter_src_port = filter_src_port
        self.filter_dst_ip = filter_dst_ip
        self.filter_dst_port = filter_dst_port
        self.stream_index = stream_index
        self.filter_tcp = filter_tcp
        self.filter_quic = filter_quic
        self.packets: List[Tuple[float, int, str, int]] = []
        self.tcp_seq_data: List[Tuple[float, int, str, str, int, int]] = []

    def read_pcap(self) -> Tuple[List[Tuple[float, int, str, int]], List[Tuple[float, int, str, str, int, int]]]:
        logging.info(f"Starting to read pcap file: {self.pcap_file}")
        start_time_read = time.time()
        try:
            capture = pyshark.FileCapture(self.pcap_file)
            start_time = None

            for packet in capture:
                try:
                    timestamp = float(packet.sniff_timestamp)
                    length = int(packet.length)
                    protocol = 'Other'
                    src_port = int(packet[packet.transport_layer].srcport) if hasattr(
                        packet[packet.transport_layer], 'srcport') else 0

                    if start_time is None:
                        start_time = timestamp

                    relative_timestamp = timestamp - start_time

                    if 'TCP' in packet:
                        protocol = 'TCP'
                        self.process_tcp_packet(packet, relative_timestamp)
                    elif 'QUIC' in packet:
                        protocol = 'QUIC'

                    self.packets.append(
                        (relative_timestamp, length, protocol, src_port))
                except AttributeError:
                    pass

            capture.close()
            end_time_read = time.time()
            logging.info(
                f"Finished reading pcap file: {self.pcap_file}. Total packets read: {len(self.packets)} in {end_time_read - start_time_read:.2f} seconds")
        except Exception as e:
            logging.error(
                f"Error reading pcap file: {self.pcap_file}. Exception: {e}")
        return self.packets, self.tcp_seq_data

    def process_tcp_packet(self, packet: pyshark.packet.packet.Packet, relative_timestamp: float) -> None:
        try:
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst
            src_port = int(packet.tcp.srcport)
            dst_port = int(packet.tcp.dstport)
            seq_num = int(packet.tcp.seq)

            if (self.filter_src_ip and self.filter_dst_ip and self.filter_src_port and self.filter_dst_port):
                if (src_ip == self.filter_src_ip and dst_ip == self.filter_dst_ip and
                        src_port == self.filter_src_port and dst_port == self.filter_dst_port):
                    self.tcp_seq_data.append(
                        (relative_timestamp, seq_num, src_ip, dst_ip, src_port, dst_port))
            else:
                self.tcp_seq_data.append(
                    (relative_timestamp, seq_num, src_ip, dst_ip, src_port, dst_port))
        except Exception as e:
            logging.error(f"Error processing TCP packet. Exception: {e}")


class ThroughputCalculator:
    def __init__(self, interval: float):
        self.interval = interval

    def calculate(self, packets: List[Tuple[float, int, str, int]]) -> pd.DataFrame:
        logging.info("Calculating throughput.")
        try:
            df = pd.DataFrame(packets, columns=[
                              'Timestamp', 'Length', 'Protocol', 'SrcPort'])
            df['Time_Bin'] = (df['Timestamp'] // self.interval) * self.interval

            min_time_bin = df['Time_Bin'].min()
            max_time_bin = df['Time_Bin'].max()
            complete_time_bins = pd.Series(
                np.arange(min_time_bin, max_time_bin + self.interval, self.interval))

            throughput = df.groupby(['Time_Bin', 'Protocol'])[
                'Length'].sum().unstack(fill_value=0) * 8 / self.interval
            throughput_df = throughput.reset_index()
            throughput_df['end_interval'] = throughput_df['Time_Bin'] + \
                self.interval
            throughput_df['throughput (bps)'] = throughput_df.sum(
                axis=1) - throughput_df['Time_Bin'] - throughput_df['end_interval']
            throughput_df.columns.name = None
            throughput_df = throughput_df[[
                'Time_Bin', 'end_interval', 'throughput (bps)', 'TCP', 'QUIC']]
            throughput_df.columns = [
                'start_interval (s)', 'end_interval (s)', 'throughput (bps)', 'tcp_throughput (bps)', 'quic_throughput (bps)']

            logging.info("Throughput calculation completed.")
            return throughput_df
        except Exception as e:
            logging.error(f"Error calculating throughput. Exception: {e}")
            return pd.DataFrame(columns=['start_interval (s)', 'end_interval (s)', 'throughput (bps)', 'tcp_throughput (bps)', 'quic_throughput (bps)'])


class Plotter:
    def __init__(self, base_name: str):
        self.base_name = f"{base_name}.pcap"

    def plot_throughput(self, throughput: pd.DataFrame, plot_tcp: bool, plot_quic: bool) -> None:
        sns.set()

        if throughput.empty:
            logging.info("No throughput data available to plot.")
            return

        logging.info("Plotting throughput.")
        try:
            # Convert zero throughputs to np.nan to create gaps in the plot
            throughput['throughput (bps)'] = throughput['throughput (bps)'].replace(
                0, np.nan)
            throughput['tcp_throughput (bps)'] = throughput['tcp_throughput (bps)'].replace(
                0, np.nan)
            throughput['quic_throughput (bps)'] = throughput['quic_throughput (bps)'].replace(
                0, np.nan)

            plt.figure(figsize=(12, 6))

            if plot_tcp and 'tcp_throughput (bps)' in throughput.columns:
                sns.lineplot(x='start_interval (s)', y='tcp_throughput (bps)',
                             data=throughput, label='TCP', linestyle='-', marker=None)

            if plot_quic and 'quic_throughput (bps)' in throughput.columns:
                sns.lineplot(x='start_interval (s)', y='quic_throughput (bps)',
                             data=throughput, label='QUIC', linestyle='-', marker=None)

            sns.lineplot(x='start_interval (s)', y='throughput (bps)',
                         data=throughput, label='Total', linestyle='--', marker=None)

            plt.title('Throughput')
            plt.xlabel('Time (s)')
            plt.ylabel('Data rate (bps)')
            plt.legend()
            plt.grid(True)

            plt.gca().yaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f'{x * 1e-6:.1f} Mbps'))

            save_path_png = f"{self.base_name}.data_rate.png"
            save_path_pdf = f"{self.base_name}.data_rate.pdf"

            plt.savefig(save_path_png)
            plt.savefig(save_path_pdf)
            plt.close()
            logging.info(
                f"Throughput plot saved to {save_path_png} and {save_path_pdf}")
        except Exception as e:
            logging.error(f"Error plotting throughput. Exception: {e}")

    def plot_time_sequence(self, tcp_seq_data: List[Tuple[float, int, str, str, int, int]]) -> None:
        if not tcp_seq_data:
            logging.info("No TCP sequence data available to plot.")
            return

        logging.info("Plotting TCP sequence numbers over time.")
        try:
            seq_df = pd.DataFrame(tcp_seq_data, columns=[
                                  'Timestamp', 'SequenceNumber', 'SrcIP', 'DstIP', 'SrcPort', 'DstPort'])

            plt.figure(figsize=(12, 6))

            for (src_ip, dst_ip, src_port, dst_port), group in seq_df.groupby(['SrcIP', 'DstIP', 'SrcPort', 'DstPort']):
                plt.scatter(group['Timestamp'], group['SequenceNumber'],
                            label=f"{src_ip}:{src_port} -> {dst_ip}:{dst_port}", s=5)

            plt.title('TCP Sequence Number Over Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Sequence Number')
            plt.legend()
            plt.grid(True)

            save_path_seq_png = f"{self.base_name}.seq.png"
            save_path_seq_pdf = f"{self.base_name}.seq.pdf"

            plt.savefig(save_path_seq_png)
            plt.savefig(save_path_seq_pdf)
            plt.close()
            logging.info(
                f"Sequence plot saved to {save_path_seq_png} and {save_path_seq_pdf}")
        except Exception as e:
            logging.error(
                f"Error plotting TCP sequence numbers. Exception: {e}")


class PcapAnalyzer:
    def __init__(self, pcap_file: str, interval: float, plot_seq: bool, filter_src_ip: Optional[str],
                 filter_src_port: Optional[int], filter_dst_ip: Optional[str], filter_dst_port: Optional[int],
                 stream_index: Optional[int] = None, filter_tcp: bool = False, filter_quic: bool = False):
        self.pcap_file = pcap_file
        self.processor = PcapFileProcessor(
            pcap_file, plot_seq, filter_src_ip, filter_src_port, filter_dst_ip, filter_dst_port, stream_index, filter_tcp, filter_quic)
        self.calculator = ThroughputCalculator(interval)
        self.plotter = Plotter(os.path.splitext(pcap_file)[0])
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self) -> None:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger.info("Logging is set up.")

    def run_analysis(self, plot_tcp: bool, plot_quic: bool) -> None:
        start_time = time.time()
        self.logger.info(f"Starting analysis for file: {self.pcap_file}")

        packets, tcp_seq_data = self.processor.read_pcap()
        throughput = self.calculator.calculate(packets)

        self.plotter.plot_throughput(throughput, plot_tcp, plot_quic)
        throughput.to_csv(
            f"{self.plotter.base_name}.data_rate.csv", index=False)
        self.logger.info(
            f"Throughput data saved to {self.plotter.base_name}.data_rate.csv")

        if tcp_seq_data:
            self.plotter.plot_time_sequence(tcp_seq_data)
            seq_df = pd.DataFrame(tcp_seq_data, columns=[
                                  'Timestamp', 'SequenceNumber', 'SrcIP', 'DstIP', 'SrcPort', 'DstPort'])
            seq_df.to_csv(f"{self.plotter.base_name}.seq.csv", index=False)
            self.logger.info(
                f"Sequence data saved to {self.plotter.base_name}.seq.csv")

        end_time = time.time()
        self.logger.info(f"Total runtime: {end_time - start_time:.2f} seconds")


def analyze_pcap_file(pcap_file: str, interval: float, plot_seq: bool, filter_src_ip: Optional[str],
                      filter_src_port: Optional[int], filter_dst_ip: Optional[str], filter_dst_port: Optional[int],
                      stream_index: Optional[int], filter_tcp: bool, filter_quic: bool) -> None:
    if not os.path.exists(pcap_file):
        logging.error(f"Pcap file does not exist: {pcap_file}")
        return

    analyzer = PcapAnalyzer(pcap_file=pcap_file, interval=interval, plot_seq=plot_seq,
                            filter_src_ip=filter_src_ip, filter_src_port=filter_src_port,
                            filter_dst_ip=filter_dst_ip, filter_dst_port=filter_dst_port,
                            stream_index=stream_index, filter_tcp=filter_tcp, filter_quic=filter_quic)
    analyzer.run_analysis(filter_tcp, filter_quic)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and plot throughput from pcap files.")
    parser.add_argument("pcap_files", type=str, nargs='+',
                        help="Paths to the pcap files")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Time interval in seconds for calculating throughput")
    parser.add_argument("--plot-seq", action="store_true",
                        help="Plot TCP sequence number over time (only for TCP)")
    parser.add_argument("--filter-src-ip", type=str,
                        help="Source IP address to filter")
    parser.add_argument("--filter-src-port", type=int,
                        help="Source port to filter")
    parser.add_argument("--filter-dst-ip", type=str,
                        help="Destination IP address to filter")
    parser.add_argument("--filter-dst-port", type=int,
                        help="Destination port to filter")
    parser.add_argument("--stream-index", type=int,
                        help="Filter for a specific TCP stream index")
    parser.add_argument("--tcp", action="store_true",
                        help="Filter TCP connections")
    parser.add_argument("--quic", action="store_true",
                        help="Filter QUIC connections")
    args = parser.parse_args()

    with Pool(cpu_count()) as pool:
        pool.starmap(analyze_pcap_file, [(pcap_file, args.interval, args.plot_seq, args.filter_src_ip,
                                          args.filter_src_port, args.filter_dst_ip, args.filter_dst_port,
                                          args.stream_index, args.tcp, args.quic) for pcap_file in args.pcap_files])


if __name__ == "__main__":
    main()
