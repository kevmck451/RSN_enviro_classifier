# Functions for utilities

from pathlib import Path
import pandas as pd
import time
import csv
import os



# Checks to see if directory exists and if it does it creates it
def create_directory_if_not_exists(file_path):
    path = Path(file_path)
    if not path.exists():
        # Create the file and any necessary parent directories
        path.mkdir(parents=True)
    else:
        pass

# Checks to see if file exists
def check_file_exists(file_path):
    path = Path(file_path)

    if path.exists() and path.is_file():
        return True
    else:
        return False

# Copy directory structure
def copy_directory_structure(src_dir, dest_dir):

    # Make sure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for dirpath, dirnames, filenames in os.walk(src_dir):
        # construct the destination directory path
        dest_path = dirpath.replace(src_dir, dest_dir)
        # create directory if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
            print('New Directory Structure Created')


class CSVFile:
    def __init__(self, file_path):
        self.file_path = str(file_path)
        self.header, self.data = self._read_csv_file()

    def _read_csv_file(self):
        with open(self.file_path, 'r', encoding='utf-8-sig') as csvfile:
            csvreader = csv.reader(csvfile)
            data = list(csvreader)
        header = [column.strip('\ufeff') for column in data[0]]  # Remove the BOM character from the first column
        return header, data[1:]

    def print_entries(self):
        for row in self.data:
            print(', '.join(row))

    def get_column(self, column_name):
        column_index = self.header.index(column_name)
        column_data = [row[column_index] for row in self.data]
        return column_data

    def filter_rows(self, condition):
        filtered_data = [row for row in self.data if condition(row)]
        return filtered_data

    def csv_entries(self):
        csv_list = []
        with open(self.file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                csv_list.append(', '.join(row))
                # print(', '.join(row))
        return csv_list

    def sorted_csv_entries(self, sort_column):
        sorted_data = []
        with open(self.file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)  # Read the header row
            header = [column.strip('\ufeff') for column in header]
            column_index = header.index(sort_column)
            sorted_rows = sorted(csvreader, key=lambda row: row[column_index].lstrip('\ufeff'))

            for row in sorted_rows:
                sorted_data.append([', '.join(row)])

        return sorted_data

    def get_value(self, sample_name, header_name):
        sample_index = self.header.index(header_name)
        for row in self.data:
            if row[0] == sample_name:
                return row[sample_index]
        return None

    def update_value(self, sample_name, header_name, value):
        sample_index = self.header.index(header_name)
        for row in self.data:
            if row[0] == sample_name:
                row[sample_index] = value
                break

    def add_column(self, header_name, column_data):
        self.header.append(header_name)
        for i in range(len(self.data)):
            self.data[i].append(column_data[i] if i < len(column_data) else "")

    def replace_column(self, header_name, new_column_data):
        column_index = self.header.index(header_name)
        for i in range(len(self.data)):
            self.data[i][column_index] = new_column_data[i] if i < len(new_column_data) else ""

    def rename_headers(self, new_headers):
        if len(new_headers) != len(self.header):
            raise ValueError("The number of new headers must match the number of existing headers.")
        self.header = new_headers

        # Write the modified data back to the CSV file
        with open(self.file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.header)
            csvwriter.writerows(self.data)

    def save_changes(self):
        with open(self.file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.header)
            csvwriter.writerows(self.data)




# DATA LOGGING CLASS
class record_class:
    def __init__(self):

        self.record_location_base = 'Records/'
        self.record_location_ad = self.record_location_base + 'Anomaly Detectors/'

        # File Name, Detector, AD Score, AD Eff, Time, AD Parameters, Preprocesses, Image Stats
        self.fields = ['File Name', 'Detector', 'Score', 'Efficiency', 'Run Time', 'Parameters', 'Image Stats']

        if not os.path.isdir(self.record_location_ad):
            os.makedirs(self.record_location_ad)

    def record(self, filename, detector, score, eff, runtime, params, imgstats):
        import csv
        self.filename = self.record_location_ad + detector + '.csv'
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fields)
                writer.writeheader()

        with open(self.filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writerow({self.fields[0] : filename,
                             self.fields[1] : detector,
                             self.fields[2] : score,
                             self.fields[3] : eff,
                             self.fields[4] : runtime,
                             self.fields[5] : params,
                             self.fields[6] : imgstats})

    def compile_detectors(self):

        try:

            if os.path.exists(f'{self.record_location_ad}Detector Data Main.csv'):
                os.remove(f'{self.record_location_ad}Detector Data Main.csv')

            # Get the list of all csv files in the folder
            csv_files = [f'{self.record_location_ad}{f}' for f in os.listdir(self.record_location_ad) if f.endswith('.csv')]

            # Create an empty DataFrame to store the data from all csv files
            df = pd.DataFrame(
                columns=['File Name', 'Detector', 'Score', 'Efficiency', 'Run Time', 'Parameters', 'Image Stats'])

            # Loop through all csv files in the folder
            for file in csv_files:
                # Read the csv file into a DataFrame
                file_df = pd.read_csv(file)

                # Extract the specified columns from the csv file
                data = file_df[['File Name', 'Detector', 'Score', 'Efficiency', 'Run Time', 'Parameters', 'Image Stats']]

                # Append the data from the csv file to the DataFrame
                df = df.append(data)

            # Write the DataFrame to a new csv file
            df.to_csv(f'{self.record_location_ad}Detector Data Main.csv', index=False)
        except:
            print('DETECTOR DATA MAIN COULD NOT BE COMPILED. CHECK FILES.')


# TIME CLASS TO GIVE STATS ABOUT HOW LONG FUNCTION TAKES
class time_class:
    def __init__(self, name):
        self.start_time = time.time()
        self.name = name

    def stats(self):
        total_time = round((time.time() - self.start_time), 1)

        if total_time < 60:
            print(f'{self.name} Time: {total_time} secs')
        else:
            total_time_min = round((total_time / 60), 1)
            print(f'{self.name} Time: {total_time_min} mins')

        return total_time