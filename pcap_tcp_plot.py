import pyshark
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import logging
from multiprocessing import Pool, cpu_count
from matplotlib.ticker import FuncFormatter


class PcapAnalyzer:
    def __init__(self, pcap_file, interval=1.0, plot_seq=False, filter_src_ip=None, filter_src_port=None, filter_dst_ip=None, filter_dst_port=None):
        self.pcap_file = pcap_file
        self.interval = interval
        self.plot_seq = plot_seq
        self.filter_src_ip = filter_src_ip
        self.filter_src_port = filter_src_port
        self.filter_dst_ip = filter_dst_ip
        self.filter_dst_port = filter_dst_port
        self.packets = []
        self.throughput = None
        self.goodput = None
        self.tcp_seq_data = []
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger.info("Logging is set up.")

    def read_pcap(self):
        self.logger.info(f"Reading pcap file: {self.pcap_file}")
        capture = pyshark.FileCapture(self.pcap_file)
        start_time = None

        for packet in capture:
            try:
                # Extract timestamp, packet length, and IP payload length
                timestamp = float(packet.sniff_timestamp)
                length = int(packet.length)
                if 'IP' in packet:
                    ip_payload_length = int(
                        packet.ip.len) - (int(packet.ip.hdr_len) * 4)
                else:
                    ip_payload_length = length  # If not an IP packet, consider full length

                if start_time is None:
                    start_time = timestamp

                relative_timestamp = timestamp - start_time
                self.packets.append(
                    (relative_timestamp, length, ip_payload_length))

                # Extract TCP sequence number data if needed
                if self.plot_seq and 'TCP' in packet:
                    src_ip = packet.ip.src
                    dst_ip = packet.ip.dst
                    src_port = int(packet.tcp.srcport)
                    dst_port = int(packet.tcp.dstport)

                    # Apply filtering if filter parameters are provided
                    if self.filter_src_ip and self.filter_dst_ip and self.filter_src_port and self.filter_dst_port:
                        if (src_ip == self.filter_src_ip and dst_ip == self.filter_dst_ip and
                                src_port == self.filter_src_port and dst_port == self.filter_dst_port):
                            seq_num = int(packet.tcp.seq)
                            self.tcp_seq_data.append(
                                (relative_timestamp, seq_num, src_ip, dst_ip, src_port, dst_port))
                    else:
                        seq_num = int(packet.tcp.seq)
                        self.tcp_seq_data.append(
                            (relative_timestamp, seq_num, src_ip, dst_ip, src_port, dst_port))

            except AttributeError:
                # Skip packets that don't have the required attributes
                continue
        capture.close()
        self.logger.info(f"Total packets read: {len(self.packets)}")

    def calculate_throughput_goodput(self):
        self.logger.info("Calculating throughput and goodput.")
        # Convert to DataFrame
        df = pd.DataFrame(self.packets, columns=[
                          'Timestamp', 'Length', 'IP_Payload_Length'])

        # Create time bins
        df['Time_Bin'] = (df['Timestamp'] // self.interval) * self.interval

        # Calculate throughput and goodput per interval
        self.throughput = df.groupby('Time_Bin')['Length'].sum(
        ) * 8 / self.interval  # bits per second
        self.goodput = df.groupby('Time_Bin')['IP_Payload_Length'].sum(
        ) * 8 / self.interval  # bits per second

        self.logger.info("Calculation completed.")

    def plot_throughput_goodput(self):
        self.logger.info("Plotting throughput and goodput.")
        sns.set(style="whitegrid")  # Use whitegrid Seaborn style

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=self.throughput.index, y=self.throughput.values,
                     label='Throughput', marker='o', linestyle='-')
        sns.lineplot(x=self.goodput.index, y=self.goodput.values,
                     label='Goodput', marker='x', linestyle='-')

        plt.title('Throughput and Goodput Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Bits per Second (bps)')
        plt.legend()
        plt.grid(True)

        # Format y-axis labels to Mbps
        def bits_to_mbps(x, pos):
            return f'{x * 1e-6:.1f} Mbps'

        plt.gca().yaxis.set_major_formatter(FuncFormatter(bits_to_mbps))

        # Create the output file paths with .png and .pdf extensions
        base_name = os.path.basename(self.pcap_file)
        save_path_png = os.path.join(os.path.dirname(
            self.pcap_file), f"{base_name}.data_rate.png")
        save_path_pdf = os.path.join(os.path.dirname(
            self.pcap_file), f"{base_name}.data_rate.pdf")

        plt.savefig(save_path_png)
        plt.savefig(save_path_pdf)
        self.logger.info(f"Plot saved to {save_path_png} and {save_path_pdf}")
        plt.close()

        # Combine throughput and goodput into a single DataFrame
        data_rate_df = pd.DataFrame({
            'Time_Bin': self.throughput.index,
            'Throughput (bps)': self.throughput.values,
            'Goodput (bps)': self.goodput.values
        })

        # Save throughput and goodput to CSV
        save_path_csv = os.path.join(os.path.dirname(
            self.pcap_file), f"{base_name}.data_rate.csv")
        data_rate_df.to_csv(save_path_csv, index=False)
        self.logger.info(f"Data rates saved to {save_path_csv}")

    def plot_time_sequence(self):
        if not self.plot_seq:
            return

        self.logger.info("Plotting time/sequence graph.")
        sns.set(style="whitegrid")  # Use whitegrid Seaborn style

        seq_df = pd.DataFrame(self.tcp_seq_data, columns=[
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

        # Create the output file paths with .png and .pdf extensions
        base_name = os.path.basename(self.pcap_file)
        save_path_seq_png = os.path.join(os.path.dirname(
            self.pcap_file), f"{base_name}.seq.png")
        save_path_seq_pdf = os.path.join(os.path.dirname(
            self.pcap_file), f"{base_name}.seq.pdf")

        plt.savefig(save_path_seq_png)
        plt.savefig(save_path_seq_pdf)
        self.logger.info(
            f"Time/sequence plot saved to {save_path_seq_png} and {save_path_seq_pdf}")
        plt.close()

        # Save sequence data to CSV
        save_path_seq_csv = os.path.join(os.path.dirname(
            self.pcap_file), f"{base_name}.seq.csv")
        seq_df.to_csv(save_path_seq_csv, index=False)
        self.logger.info(f"Sequence data saved to {save_path_seq_csv}")

    def run_analysis(self):
        self.read_pcap()
        self.calculate_throughput_goodput()
        self.plot_throughput_goodput()
        self.plot_time_sequence()


def analyze_pcap_file(pcap_file, interval, plot_seq, filter_src_ip, filter_src_port, filter_dst_ip, filter_dst_port):
    analyzer = PcapAnalyzer(pcap_file=pcap_file, interval=interval, plot_seq=plot_seq,
                            filter_src_ip=filter_src_ip, filter_src_port=filter_src_port,
                            filter_dst_ip=filter_dst_ip, filter_dst_port=filter_dst_port)
    analyzer.run_analysis()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot throughput and goodput from pcap files.")
    parser.add_argument("pcap_files", type=str, nargs='+',
                        help="Paths to the pcap files")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Time interval in seconds for calculating throughput and goodput")
    parser.add_argument("--plot-seq", action="store_true",
                        help="Plot TCP sequence number over time")
    parser.add_argument("--filter-src-ip", type=str,
                        help="Source IP address to filter")
    parser.add_argument("--filter-src-port", type=int,
                        help="Source port to filter")
    parser.add_argument("--filter-dst-ip", type=str,
                        help="Destination IP address to filter")
    parser.add_argument("--filter-dst-port", type=int,
                        help="Destination port to filter")
    args = parser.parse_args()

    # Use all available CPU cores for processing
    with Pool(cpu_count()) as pool:
        pool.starmap(analyze_pcap_file, [(pcap_file, args.interval, args.plot_seq, args.filter_src_ip,
                     args.filter_src_port, args.filter_dst_ip, args.filter_dst_port) for pcap_file in args.pcap_files])


if __name__ == "__main__":
    main()
