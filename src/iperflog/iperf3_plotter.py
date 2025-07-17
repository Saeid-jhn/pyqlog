import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch
import numpy as np


def plot_csv_file(file_path, add_title=False):
    # Use Seaborn's default theme
    sns.set_theme()  # Using default Seaborn style

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert RTT from microseconds to milliseconds
    if 'RTT (microsecond)' in df.columns:
        df['RTT (ms)'] = df['RTT (microsecond)'] / 1000.0
        rtt_column = 'RTT (ms)'
    else:
        rtt_column = None

    # Extract base name and directory
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    dir_name = os.path.dirname(file_path)

    # Create a figure and primary axis for the main plot
    fig, ax1 = plt.subplots(figsize=(25, 6))

    # Plot Goodput on the primary y-axis
    sns.lineplot(
        x='start time (sec)', y='goodput (bits/sec)', data=df,
        ax=ax1, color='tab:blue', label='Goodput'
    )
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Goodput (bits/sec)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Set plot title if add_title is True
    if add_title:
        ax1.set_title(base_name)

    # Initialize lists for legends
    lines, labels = [], []

    # Get handles and labels from ax1
    handle1, label1 = ax1.get_legend_handles_labels()
    lines.extend(handle1)
    labels.extend(label1)

    # Remove default legend from ax1
    if ax1.get_legend():
        ax1.get_legend().remove()

    # Plot Retransmissions if available
    if 'Retransmissions' in df.columns:
        ax2 = ax1.twinx()
        sns.barplot(
            x='start time (sec)', y='Retransmissions', data=df,
            ax=ax2, color='tab:red', alpha=0.3
        )
        ax2.set_ylabel('Retransmissions', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.spines['right'].set_position(('outward', 60))
        ax2.grid(False)  # Turn off grid lines for ax2

        # Create a custom legend handle for Retransmissions
        patch = Patch(facecolor='tab:red', alpha=0.3, label='Retransmissions')
        lines.append(patch)
        labels.append('Retransmissions')

    # Plot cwnd if available
    if 'cwnd (K)' in df.columns:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 120))
        sns.lineplot(
            x='start time (sec)', y='cwnd (K)', data=df,
            ax=ax3, color='tab:green', label='cwnd (K)'
        )
        ax3.set_ylabel('cwnd (K)', color='tab:green')
        ax3.tick_params(axis='y', labelcolor='tab:green')
        ax3.grid(False)  # Turn off grid lines for ax3

        # Get handles and labels from ax3
        handle3, label3 = ax3.get_legend_handles_labels()
        lines.extend(handle3)
        labels.extend(label3)

        # Remove default legend from ax3
        if ax3.get_legend():
            ax3.get_legend().remove()

    # Plot RTT if available
    if rtt_column:
        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 180))
        sns.lineplot(
            x='start time (sec)', y=rtt_column, data=df,
            ax=ax4, color='tab:purple', linestyle='--', label='RTT'
        )
        ax4.set_ylabel('RTT (ms)', color='tab:purple')
        ax4.tick_params(axis='y', labelcolor='tab:purple')
        ax4.grid(False)  # Turn off grid lines for ax4

        # Get handles and labels from ax4
        handle4, label4 = ax4.get_legend_handles_labels()
        lines.extend(handle4)
        labels.extend(label4)

        # Remove default legend from ax4
        if ax4.get_legend():
            ax4.get_legend().remove()

    # Remove duplicate labels
    legend_dict = dict()
    for h, l in zip(lines, labels):
        if l not in legend_dict:
            legend_dict[l] = h
    labels = list(legend_dict.keys())
    lines = list(legend_dict.values())

    # Add combined legend
    ax1.legend(lines, labels, loc='upper left')

    # Adjust x-axis ticks based on 'start time (sec)'
    min_time = df['start time (sec)'].min()
    max_time = df['start time (sec)'].max()
    x_ticks = np.arange(0, max_time + 10, 10)
    ax1.set_xticks(x_ticks)
    ax1.set_xlim(left=0)  # Ensure the x-axis starts at 0

    # Enable grid lines only on the x-axis
    ax1.grid(True, axis='x')

    # Adjust layout
    fig.tight_layout()

    # Save the main plot
    main_plot_png = os.path.join(dir_name, f"{base_name}.png")
    main_plot_pdf = os.path.join(dir_name, f"{base_name}.pdf")
    plt.savefig(main_plot_png)
    plt.savefig(main_plot_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot data from CSV files.')
    parser.add_argument('files', nargs='+', help='CSV file(s) to process')
    parser.add_argument('--add-title', action='store_true',
                        help='Add title to the plot')
    args = parser.parse_args()

    for file_path in args.files:
        plot_csv_file(file_path, add_title=args.add_title)
        print(f"Plots saved for {file_path}")


if __name__ == '__main__':
    main()
