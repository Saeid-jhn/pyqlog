import pyshark
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os
import logging
import time
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count
from matplotlib.ticker import FuncFormatter


class PcapFileProcessor:
    def __init__(self, pcap_file: str, plot_seq: bool = False, stream_index: Optional[int] = None,
                 filter_tcp: bool = False, filter_quic: bool = False):
        self.pcap_file = pcap_file
        self.plot_seq = plot_seq
        self.stream_index = stream_index
        self.filter_tcp = filter_tcp
        self.filter_quic = filter_quic
        # Include src_port and dst_port
        self.packets: List[Tuple[float, int, str, int, int]] = []
        self.tcp_seq_data: List[Tuple[float, int, str, str, int, int]] = []

    def read_pcap(self) -> Tuple[List[Tuple[float, int, str, int, int]], List[Tuple[float, int, str, str, int, int]]]:
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

                    if start_time is None:
                        start_time = timestamp

                    relative_timestamp = timestamp - start_time

                    src_port = None
                    dst_port = None

                    if 'TCP' in packet:
                        protocol = 'TCP'
                        src_port = int(packet.tcp.srcport)
                        dst_port = int(packet.tcp.dstport)
                        self.process_tcp_packet(packet, relative_timestamp)
                    elif 'QUIC' in packet:
                        protocol = 'QUIC'
                        src_port = int(packet.udp.srcport)
                        dst_port = int(packet.udp.dstport)
                    elif 'UDP' in packet:
                        protocol = 'UDP'
                        src_port = int(packet.udp.srcport)
                        dst_port = int(packet.udp.dstport)

                    self.packets.append(
                        (relative_timestamp, length, protocol, src_port, dst_port))
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

            self.tcp_seq_data.append(
                (relative_timestamp, seq_num, src_ip, dst_ip, src_port, dst_port))
        except Exception as e:
            logging.error(f"Error processing TCP packet. Exception: {e}")


class ThroughputCalculator:
    def __init__(self, interval: float, include_tcp: bool, include_quic: bool, ports: Optional[List[int]] = None):
        self.interval = interval
        self.include_tcp = include_tcp
        self.include_quic = include_quic
        self.ports = ports  # List of ports to calculate throughput for

    def calculate(self, packets: List[Tuple[float, int, str, int, int]]) -> pd.DataFrame:
        logging.info("Calculating throughput.")
        try:
            df = pd.DataFrame(packets, columns=[
                              'Timestamp', 'Length', 'Protocol', 'SrcPort', 'DstPort'])
            df['Time_Bin'] = (df['Timestamp'] // self.interval) * self.interval

            # Calculate total throughput
            total_throughput = df.groupby(
                'Time_Bin')['Length'].sum().reset_index()
            # bits per second
            total_throughput['Throughput'] = total_throughput['Length'] * \
                8 / self.interval
            total_throughput = total_throughput[['Time_Bin', 'Throughput']]
            total_throughput.rename(
                columns={'Throughput': 'TotalThroughput'}, inplace=True)

            # Consider only Destination Ports to avoid double counting
            port_df = df[['Time_Bin', 'DstPort', 'Length']].rename(columns={'DstPort': 'Port'})


            # If ports are specified, filter only those ports for calculation
            if self.ports:
                port_df = port_df[port_df['Port'].isin(self.ports)]

            # Group by Time_Bin and Port, sum the lengths, and calculate throughput
            port_throughput = port_df.groupby(['Time_Bin', 'Port'])[
                'Length'].sum().reset_index()
            # bits per second
            port_throughput['Throughput'] = port_throughput['Length'] * \
                8 / self.interval

            # Merge total throughput with port throughput
            throughput = pd.merge(
                port_throughput, total_throughput, on='Time_Bin', how='left')

            logging.info("Throughput calculation completed.")
            return throughput
        except Exception as e:
            logging.error(f"Error calculating throughput. Exception: {e}")
            return pd.DataFrame()


class Plotter:
    def __init__(self, base_name: str, plot_total: bool = False, port_legends: Optional[Dict[int, str]] = None):
        self.base_name = f"{base_name}.pcap"
        self.plot_total = plot_total  # Flag to indicate whether to plot total throughput
        self.port_legends = port_legends or {}  # Mapping from port to legend

    def plot_throughput(self, throughput: pd.DataFrame, ports: List[int]) -> None:
        sns.set()

        if throughput.empty:
            logging.info("No throughput data available to plot.")
            return

        logging.info("Plotting throughput.")
        try:
            plt.figure(figsize=(12, 6))

            if ports:
                for port in ports:
                    port_data = throughput[throughput['Port'] == port]
                    if port_data.empty:
                        logging.info(f"No data available for port {port}")
                        continue
                    legend = self.port_legends.get(
                        port, f'Port {port}')  # Use legend if available
                    sns.lineplot(x='Time_Bin', y='Throughput',
                                 data=port_data, label=legend, linestyle='-', marker=None)

            if self.plot_total:
                total_data = throughput[['Time_Bin',
                                         'TotalThroughput']].drop_duplicates()
                sns.lineplot(x='Time_Bin', y='TotalThroughput',
                             data=total_data, label='Total', linestyle='--', marker=None, color='black')

            plt.title('Throughput by Port')
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

    def plot_time_sequence(self, tcp_seq_data: List[Tuple[float, int, str, str, int, int]], ports: List[int]) -> None:
        if not tcp_seq_data:
            logging.info("No TCP sequence data available to plot.")
            return

        logging.info("Plotting TCP sequence numbers over time.")
        try:
            seq_df = pd.DataFrame(tcp_seq_data, columns=[
                                  'Timestamp', 'SequenceNumber', 'SrcIP', 'DstIP', 'SrcPort', 'DstPort'])

            plt.figure(figsize=(12, 6))

            if ports:
                for port in ports:
                    port_data = seq_df[(seq_df['SrcPort'] == port) | (
                        seq_df['DstPort'] == port)]
                    if port_data.empty:
                        logging.info(
                            f"No sequence data available for port {port}")
                        continue
                    legend = self.port_legends.get(
                        port, f'Port {port}')  # Use legend if available
                    plt.scatter(port_data['Timestamp'], port_data['SequenceNumber'],
                                label=legend, s=5)
            else:
                plt.scatter(seq_df['Timestamp'], seq_df['SequenceNumber'],
                            label='TCP Sequence', s=5)

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
    def __init__(self, pcap_file: str, interval: float, plot_seq: bool, stream_index: Optional[int] = None,
                 filter_tcp: bool = False, filter_quic: bool = False, ports: Optional[List[int]] = None,
                 plot_total: bool = False, port_legends: Optional[Dict[int, str]] = None, output_dir: str = "output"):
        self.pcap_file = pcap_file
        self.processor = PcapFileProcessor(
            pcap_file, plot_seq, stream_index, filter_tcp, filter_quic)
        self.calculator = ThroughputCalculator(
            interval, filter_tcp, filter_quic, ports)
        base_name = os.path.splitext(os.path.basename(pcap_file))[0]
        self.plotter = Plotter(os.path.join(
            output_dir, base_name), plot_total, port_legends)
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.ports = ports  # Ports for plotting

    def setup_logging(self) -> None:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger.info("Logging is set up.")

    def run_analysis(self, plot_tcp: bool, plot_quic: bool) -> None:
        start_time = time.time()
        self.logger.info(f"Starting analysis for file: {self.pcap_file}")

        packets, tcp_seq_data = self.processor.read_pcap()
        throughput = self.calculator.calculate(packets)

        self.plotter.plot_throughput(throughput, self.ports)
        throughput.to_csv(
            f"{self.plotter.base_name}.data_rate.csv", index=False)
        self.logger.info(
            f"Throughput data saved to {self.plotter.base_name}.data_rate.csv")

        if tcp_seq_data:
            self.plotter.plot_time_sequence(tcp_seq_data, self.ports)
            seq_df = pd.DataFrame(tcp_seq_data, columns=[
                                  'Timestamp', 'SequenceNumber', 'SrcIP', 'DstIP', 'SrcPort', 'DstPort'])
            seq_df.to_csv(f"{self.plotter.base_name}.seq.csv", index=False)
            self.logger.info(
                f"Sequence data saved to {self.plotter.base_name}.seq.csv")

        end_time = time.time()
        self.logger.info(f"Total runtime: {end_time - start_time:.2f} seconds")


def analyze_pcap_file(pcap_file: str, interval: float, plot_seq: bool,
                      stream_index: Optional[int], filter_tcp: bool, filter_quic: bool, ports: Optional[List[int]],
                      plot_total: bool, port_legends: Optional[Dict[int, str]], output_dir: Optional[str] = None) -> None:
    if not os.path.exists(pcap_file):
        logging.error(f"Pcap file does not exist: {pcap_file}")
        return

    # Set the output directory to the directory of the pcap file
    output_dir = os.path.dirname(pcap_file)

    analyzer = PcapAnalyzer(
        pcap_file=pcap_file,
        interval=interval,
        plot_seq=plot_seq,
        stream_index=stream_index,
        filter_tcp=filter_tcp,
        filter_quic=filter_quic,
        ports=ports,
        plot_total=plot_total,
        port_legends=port_legends,
        output_dir=output_dir
    )
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
    parser.add_argument("--stream-index", type=int,
                        help="Filter for a specific TCP stream index")
    parser.add_argument("--tcp", action="store_true",
                        help="Filter TCP connections")
    parser.add_argument("--quic", action="store_true",
                        help="Filter QUIC connections")
    parser.add_argument("--port", type=int, nargs='+',
                        help="List of ports to plot separately")  # Ports argument
    parser.add_argument("--port-legend", type=str, nargs='+',
                        help="List of legends corresponding to the ports specified in --port")
    parser.add_argument("--total", action="store_true",
                        help="Plot total throughput alongside per-port throughput")  # Added --total argument
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save output files")  # Added --output-dir argument
    args = parser.parse_args()

    # Validate port and port-legend arguments
    port_legends_mapping = {}
    if args.port_legend:
        if not args.port:
            parser.error(
                "--port-legend requires --port to be specified with corresponding ports.")
        if len(args.port) != len(args.port_legend):
            parser.error(
                f"The number of --port-legend arguments ({len(args.port_legend)}) does not match the number of --port arguments ({len(args.port)}).")
        # Create a mapping from port to legend
        port_legends_mapping = dict(zip(args.port, args.port_legend))
    elif args.port:
        # If no legends provided, use port numbers as legends
        port_legends_mapping = {port: f"Port {port}" for port in args.port}

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    with Pool(cpu_count()) as pool:
        pool.starmap(analyze_pcap_file, [(pcap_file, args.interval, args.plot_seq,
                                          args.stream_index, args.tcp, args.quic, args.port,
                                          args.total, port_legends_mapping, args.output_dir) for pcap_file in args.pcap_files])


if __name__ == "__main__":
    main()
