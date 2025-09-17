import csv
import random

# Function to read the dataset from a CSV file
def read_dataset(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        reader = csv.reader(file)
        # Skip the header if present
        lines = [row for row in reader if row]  # Exclude empty rows
    return lines

# Function to split dataset into train and test
def split_dataset(lines, train_ratio=0.8):
    random.shuffle(lines)  # Shuffle the rows for randomness
    split_index = int(len(lines) * train_ratio)
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]
    return train_lines, test_lines

# Function to write data to a CSV file
def write_to_csv(lines, file_path):
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        # Write the rows
        writer.writerows(lines)

# Main function
def process_dataset(input_file, train_file, test_file):
    # Step 1: Read dataset
    lines = read_dataset(input_file)

    # Step 2: Split dataset into train and test
    train_lines, test_lines = split_dataset(lines)

    # Step 3: Write to train and test files
    write_to_csv(train_lines, train_file)
    write_to_csv(test_lines, test_file)

# Specify file paths
input_file = './MICM-main/IC9600.csv'  # Input CSV file
train_file = './MICM-main/train.csv'  # Output train CSV file
test_file = './MICM-main/test.csv'    # Output test CSV file

# Run the process
process_dataset(input_file, train_file, test_file)
