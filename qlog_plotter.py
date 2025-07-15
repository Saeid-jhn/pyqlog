#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


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
    plt.rcParams['font.size'] = 10
    font_size = 10
    MB = 1e6
    plot_order = [0, 2, 3, 4, 1]

    fig, ax = plt.subplots(5, 1, figsize=(5, 12), sharex=False)
    csv_name = os.path.basename(csv_prefix)

    for axis in ax:
        axis.grid(True)

    # Plot 1: Offset / Retransmissions / Cumulative Data Size
    line_0_off = ax[plot_order[0]].plot(
        df_offsets[df_offsets['duplicate'] == False]['time_s'],
        df_offsets[df_offsets['duplicate'] == False]['offset_MB'],
        '.', markersize=1, label="offset")[0]
    line_0_re = ax[plot_order[0]].plot(
        df_offsets[df_offsets['duplicate']]['time_s'],
        df_offsets[df_offsets['duplicate']]['offset_MB'],
        '.', markersize=1, label="offset retransmitted")[0]
    line_0_pkt = ax[plot_order[0]].plot(
        df_packets['time_s'],
        df_packets['packet_size_cumsum_MB'],
        '.', markersize=1, label="cumulative data size")[0]
    ax[plot_order[0]].legend(
        handles=[line_0_off, line_0_re, line_0_pkt], markerscale=8, fontsize=font_size)
    ax[plot_order[0]].set_ylabel("offset [MB]", fontsize=font_size)
    ax[plot_order[0]].set_xlabel("Time [s]", fontsize=font_size)

    # Plot 2: Pacing Rate
    pacing_df = df_metrics[df_metrics['key'] == 'pacing_rate']
    line_1 = ax[plot_order[1]].plot(
        pacing_df['time_s'], pacing_df['value'] / MB,
        '.', markersize=1, label='pacing_rate')[0]
    ax[plot_order[1]].legend(
        handles=[line_1], markerscale=8, fontsize=font_size)
    ax[plot_order[1]].set_ylabel("pacing rate [Mbps]", fontsize=font_size)
    ax[plot_order[1]].set_xlabel("Time [s]", fontsize=font_size)

    # Plot 3: CWND / Bytes in Flight
    cwnd_df = df_metrics[df_metrics['key'].isin(['cwnd', 'congestion_window'])]
    flight_df = df_metrics[df_metrics['key'] == 'bytes_in_flight']
    line_2a = ax[plot_order[2]].plot(
        cwnd_df['time_s'], cwnd_df['value'] / MB,
        '.', markersize=1, label='cwnd')[0]
    line_2b = ax[plot_order[2]].plot(
        flight_df['time_s'], flight_df['value'] / MB,
        '.', markersize=1, label='bytes_in_flight')[0]
    ax[plot_order[2]].legend(handles=[line_2a, line_2b],
                             markerscale=8, fontsize=font_size)
    ax[plot_order[2]].set_ylabel("metrics [MB]", fontsize=font_size)
    ax[plot_order[2]].set_xlabel("Time [s]", fontsize=font_size)

    # Plot 4: RTTs
    smoothed = df_metrics[df_metrics['key'] == 'smoothed_rtt']
    latest = df_metrics[df_metrics['key'] == 'latest_rtt']
    min_rtt = df_metrics[df_metrics['key'] == 'min_rtt']
    l1 = ax[plot_order[3]].plot(
        smoothed['time_s'], smoothed['value'] / 1e3, '.', markersize=1, label='smoothed_rtt')[0]
    l2 = ax[plot_order[3]].plot(
        latest['time_s'], latest['value'] / 1e3, '.', markersize=1, label='latest_rtt')[0]
    l3 = ax[plot_order[3]].plot(
        min_rtt['time_s'], min_rtt['value'] / 1e3, '.', markersize=1, label='min_rtt')[0]
    ax[plot_order[3]].legend(handles=[l1, l2, l3],
                             markerscale=8, fontsize=font_size)
    ax[plot_order[3]].set_ylabel("RTT [ms]", fontsize=font_size)
    ax[plot_order[3]].set_xlabel("Time [s]", fontsize=font_size)

    # find the 99th‑pct maximum to avoid outleir data
    p_sm = smoothed['value'].quantile(0.99)
    p_la = latest['value'].quantile(0.99)
    p_mn = min_rtt['value'].quantile(0.99)
    rtt99 = max(p_sm, p_la, p_mn)

    # now compute the absolute minimum across the three series
    rtt_min = min(
        smoothed['value'].min(),
        latest['value'].min(),
        min_rtt['value'].min()
    )

    # set your y‑limits in milliseconds with margin
    ax[plot_order[3]].set_ylim(
        ymin=(rtt_min * 0.9) / 1e3,
        ymax=(rtt99 * 1.2) / 1e3
    )

    # Plot 5: Throughput & Goodput
    t1 = ax[plot_order[4]].plot(
        df_data_rate['start_interval (s)'],
        df_data_rate['throughput (bps)'] / MB,
        '-', markersize=1, label='throughput')[0]
    t2 = ax[plot_order[4]].plot(
        df_data_rate['start_interval (s)'],
        df_data_rate['goodput (bps)'] / MB,
        '--', markersize=1, label='goodput')[0]
    ax[plot_order[4]].legend(
        handles=[t1, t2], markerscale=8, fontsize=font_size)
    ax[plot_order[4]].set_ylabel("data rate [Mbps]", fontsize=font_size)
    ax[plot_order[4]].set_xlabel("Time [s]", fontsize=font_size)

    fig.align_ylabels(ax[:])
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    # Save in requested formats
    for ext in formats:
        if ext == 'png':
            plt.savefig(f"{csv_prefix}.{ext}", dpi=900)
        else:
            plt.savefig(f"{csv_prefix}.{ext}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot QUIC CSV metrics in original style.')
    parser.add_argument(
        'csv_prefix',
        help='Prefix of CSV files (no extension)')
    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        choices=['png', 'svg', 'pdf'],
        default=['png'],
        help='Output file formats (default: png). Warning PDF and SVG takes very long!')
    args = parser.parse_args()
    plot_all(args.csv_prefix, args.formats)


if __name__ == '__main__':
    main()
