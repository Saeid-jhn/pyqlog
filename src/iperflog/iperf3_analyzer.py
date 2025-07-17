import argparse
import json
import csv
import os


def process_file(input_file):
    # Mandatory fields in desired order
    fieldnames = ['start time (sec)', 'end time (sec)', 'goodput (bits/sec)']
    # Optional fields in desired order
    optional_fields_order = [
        'Retransmissions', 'cwnd (K)', 'RTT (microsecond)', 'RTT_var (microsecond)']
    # Keep track of which optional fields are present
    optional_fields_present = set()
    # List to store data rows
    data_rows = []

    with open(input_file, 'r') as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip lines that are not JSON

            if data.get('event') == 'interval':
                interval_data = data['data']
                stream = interval_data['streams'][0]

                # Initialize a dictionary for the current row
                row = {}

                # Mandatory fields with one decimal place
                start_time = round(stream.get('start', 0), 1)
                end_time = round(stream.get('end', 0), 1)
                row['start time (sec)'] = start_time
                row['end time (sec)'] = end_time
                row['goodput (bits/sec)'] = stream.get('bits_per_second', 0)

                # Optional fields
                if 'retransmits' in stream:
                    row['Retransmissions'] = stream['retransmits']
                    optional_fields_present.add('Retransmissions')
                if 'snd_cwnd' in stream:
                    row['cwnd (K)'] = stream['snd_cwnd'] / \
                        1000  # Convert to Kilobytes
                    optional_fields_present.add('cwnd (K)')
                if 'rtt' in stream:
                    row['RTT (microsecond)'] = stream['rtt']
                    optional_fields_present.add('RTT (microsecond)')
                if 'rttvar' in stream:
                    row['RTT_var (microsecond)'] = stream['rttvar']
                    optional_fields_present.add('RTT_var (microsecond)')

                # Append the row to the data list
                data_rows.append(row)

    # Check if the last row's interval is less than 1 second
    if data_rows:
        last_row = data_rows[-1]
        interval_duration = last_row['end time (sec)'] - \
            last_row['start time (sec)']
        if interval_duration < 1:
            # Remove the last row from data_rows
            print(
                f"Ignoring last row with interval duration {interval_duration} seconds (less than 1 second).")
            data_rows.pop()

    # Build the final fieldnames list, including only present optional fields in the desired order
    for field in optional_fields_order:
        if field in optional_fields_present:
            fieldnames.append(field)

    # Generate output file name
    output_file = f"{input_file}.csv"

    # Write data to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description='Extract data from iperf3 log files and save to CSV.')
    parser.add_argument('log_files', nargs='+',
                        help='One or more iperf3 log files to process.')
    args = parser.parse_args()

    for input_file in args.log_files:
        if os.path.isfile(input_file):
            print(f'Processing file: {input_file}')
            process_file(input_file)
            print(f'CSV file saved as: {input_file}.csv\n')
        else:
            print(f'File not found: {input_file}\n')


if __name__ == '__main__':
    main()
