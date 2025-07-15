#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Configure plot styling without Seaborn
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (5, 12)
})


def plot_all(csv_prefix: str, formats: list):
    # Load CSVs
    df_data_rate = pd.read_csv(f"{csv_prefix}.data_rate.csv")
    df_offsets = pd.read_csv(f"{csv_prefix}.offsets.csv")
    df_metrics = pd.read_csv(f"{csv_prefix}.metrics.csv")
    df_packets = pd.read_csv(f"{csv_prefix}.packets.csv")

    # Precompute required columns
    df_offsets['time_s'] = df_offsets['time'] / 1e6
    df_offsets['offset_MB'] = df_offsets['offset'] / 1e6
    df_packets['time_s'] = df_packets['time'] / 1e6

    if 'packet_size_cumsum' not in df_packets.columns:
        df_packets['packet_size_cumsum'] = df_packets['packet_size'].cumsum()
    df_packets['packet_size_cumsum_MB'] = df_packets['packet_size_cumsum'] / 1e6

    df_metrics['time_s'] = df_metrics['time'] / 1e6

    # Set up Seaborn and layout
    sns.set()
    sns.set_theme(style='whitegrid')
    MB = 1e6
    plot_order = [0, 2, 3, 4, 1]

    fig, ax = plt.subplots(5, 1, sharex=False)
    csv_name = os.path.basename(csv_prefix)

    for axis in ax:
        axis.grid(True)

    # Plot 1: Offset / Retransmissions / Cumulative Data Size
    off = df_offsets['duplicate'] == False
    re_off = df_offsets['duplicate']
    line_offset = ax[plot_order[0]].plot(
        df_offsets.loc[off, 'time_s'],
        df_offsets.loc[off, 'offset_MB'],
        '.', markersize=1, label="Offset")[0]
    line_retx = ax[plot_order[0]].plot(
        df_offsets.loc[re_off, 'time_s'],
        df_offsets.loc[re_off, 'offset_MB'],
        '.', markersize=1, label="Retransmitted Offset")[0]
    line_cum = ax[plot_order[0]].plot(
        df_packets['time_s'],
        df_packets['packet_size_cumsum_MB'],
        '.', markersize=1, label="Cumulative Data Size")[0]
    ax[plot_order[0]].legend(
        handles=[line_offset, line_retx, line_cum], markerscale=8)
    ax[plot_order[0]].set_ylabel("offset [MB]")
    ax[plot_order[0]].set_xlabel("Time [s]")

    # Plot 2: Pacing Rate
    pacing_df = df_metrics[df_metrics['key'] == 'pacing_rate']
    line_pace = ax[plot_order[1]].plot(
        pacing_df['time_s'], pacing_df['value'] / MB,
        '.', markersize=1, label='Pacing Rate')[0]
    ax[plot_order[1]].legend(handles=[line_pace], markerscale=8)
    ax[plot_order[1]].set_ylabel("pacing rate [Mbps]")
    ax[plot_order[1]].set_xlabel("Time [s]")

    # Plot 3: CWND / Bytes in Flight
    cwnd_df = df_metrics[df_metrics['key'].isin(['cwnd', 'congestion_window'])]
    flight_df = df_metrics[df_metrics['key'] == 'bytes_in_flight']
    line_cwnd = ax[plot_order[2]].plot(
        cwnd_df['time_s'], cwnd_df['value'] / MB,
        '.', markersize=1, label='CWND')[0]
    line_flight = ax[plot_order[2]].plot(
        flight_df['time_s'], flight_df['value'] / MB,
        '.', markersize=1, label='Bytes in Flight')[0]
    ax[plot_order[2]].legend(handles=[line_cwnd, line_flight], markerscale=8)
    ax[plot_order[2]].set_ylabel("Metrics [MB]")
    ax[plot_order[2]].set_xlabel("Time [s]")

    # Plot 4: RTTs
    smoothed = df_metrics[df_metrics['key'] == 'smoothed_rtt']
    latest = df_metrics[df_metrics['key'] == 'latest_rtt']
    min_rtt = df_metrics[df_metrics['key'] == 'min_rtt']
    line_sm = ax[plot_order[3]].plot(
        smoothed['time_s'], smoothed['value'] / 1e3, '.', markersize=1, label='Smoothed RTT')[0]
    line_lt = ax[plot_order[3]].plot(
        latest['time_s'], latest['value'] / 1e3, '.', markersize=1, label='Latest RTT')[0]
    line_mn = ax[plot_order[3]].plot(
        min_rtt['time_s'], min_rtt['value'] / 1e3, '.', markersize=1, label='Min RTT')[0]
    ax[plot_order[3]].legend(
        handles=[line_sm, line_lt, line_mn], markerscale=8)
    ax[plot_order[3]].set_ylabel("RTT [ms]")
    ax[plot_order[3]].set_xlabel("Time [s]")

    # 99th percentile clipping
    p_sm = smoothed['value'].quantile(0.99)
    p_la = latest['value'].quantile(0.99)
    p_mn = min_rtt['value'].quantile(0.99)
    rtt99 = max(p_sm, p_la, p_mn)
    rtt_min = min(smoothed['value'].min(),
                  latest['value'].min(), min_rtt['value'].min())
    ax[plot_order[3]].set_ylim((rtt_min * 0.9) / 1e3, (rtt99 * 1.2) / 1e3)

    # Plot 5: Throughput & Goodput
    tput = ax[plot_order[4]].plot(
        df_data_rate['start_interval (s)'], df_data_rate['throughput (bps)'] / MB,
        '-', markersize=1, label='Throughput')[0]
    gput = ax[plot_order[4]].plot(
        df_data_rate['start_interval (s)'], df_data_rate['goodput (bps)'] / MB,
        '--', markersize=1, label='Goodput')[0]
    ax[plot_order[4]].legend(handles=[tput, gput], markerscale=8)
    ax[plot_order[4]].set_ylabel("data rate [Mbps]")
    ax[plot_order[4]].set_xlabel("Time [s]")

    fig.align_ylabels(ax[:])
    fig.tight_layout()

    # Save in requested formats
    for ext in formats:
        save_path = f"{csv_prefix}.{ext}"
        if ext == 'png':
            plt.savefig(save_path, dpi=900)
        else:
            plt.savefig(save_path)


def main():
    parser = argparse.ArgumentParser(
        description='Plot QUIC CSV metrics in original style without Seaborn.')
    parser.add_argument(
        'csv_prefix', help='Prefix of CSV files (no extension)')
    parser.add_argument(
        '--formats', '-f', nargs='+', choices=['png', 'svg', 'pdf'],
        default=['png'], help='Output file formats (default: png).')
    args = parser.parse_args()
    plot_all(args.csv_prefix, args.formats)


if __name__ == '__main__':
    main()
