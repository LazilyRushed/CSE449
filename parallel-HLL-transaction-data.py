# import os
import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperloglog import HyperLogLog
from sklearn.preprocessing import MinMaxScaler


def compute_hll_partition(data):
    hll = HyperLogLog(error_rate=0.01)
    for item in data:
        hll.add(item)
    return hll


def partition_data(data, num_partitions):
    # Partition data into chunks
    chunk_size = len(data) // num_partitions
    partitions = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    return partitions


def merge_partial_hlls(partial_hlls):
    merged_hll = HyperLogLog(error_rate=0.01)
    for hll in partial_hlls:
        merged_hll.update(hll)
    return merged_hll


def parallel_hll(data, num_processes):
    pool = multiprocessing.Pool(num_processes)
    partitions = partition_data(data, num_processes)
    partial_hlls = pool.map(compute_hll_partition, partitions)
    merged_hll = merge_partial_hlls(partial_hlls)
    return merged_hll

if __name__ == '__main__':
    # Print current working directory for import
    # print("Current working directory:", os.getcwd())

    # Load your dataset
    df = pd.read_csv("./data.csv", encoding="ISO-8859-1")

    # Standardize Quantity and UnitPrice columns in the original DataFrame df
    df['Quantity'] = (df['Quantity'] - df['Quantity'].mean()) / df['Quantity'].std()
    df['UnitPrice'] = (df['UnitPrice'] - df['UnitPrice'].mean()) / df['UnitPrice'].std()

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Create a new DataFrame with standardized values and no duplicate or missing rows
    updated_df = df.copy()

    # Define the file path where you want to save the CSV file
    file_path = "./updated_dataset.csv"

    # Save the new dataset as a CSV file
    updated_df.to_csv(file_path, index=False)

    print("New dataset saved successfully as CSV file.")

    # Display dataset information
    print("Dataset Info:")
    print(updated_df.info())
    print()

    # Summary statistics
    print("Summary Statistics:")
    print(updated_df.describe())
    print()

    # Data visualization
    top_n = 12  # Adjust the number of top countries to display
    plt.figure(figsize=(10, 6))
    updated_df['Country'].value_counts().nlargest(top_n).plot(kind='bar')
    plt.xlabel('Country')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} Most Frequent Countries')
    plt.show()

    # Aggregate the data by CustomerID
    customer_agg = updated_df.groupby('CustomerID').agg({'StockCode': 'count', 'Quantity': 'sum', 'UnitPrice': 'mean'})
    customer_agg.reset_index(inplace=True)

    # Plot aggregated data
    plt.figure(figsize=(12, 8))

    # Plot mean or median of each feature against CustomerID
    plt.plot(customer_agg['CustomerID'], customer_agg['StockCode'], label='Mean StockCode', marker='o')
    plt.xlabel('CustomerID')
    plt.ylabel('Aggregated Value')
    plt.title('Aggregated Plot of CustomerID vs. StockCode')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(customer_agg['CustomerID'], customer_agg['Quantity'], label='Total Quantity', marker='o')
    plt.xlabel('CustomerID')
    plt.ylabel('Aggregated Value')
    plt.title('Aggregated Plot of CustomerID vs. Quantity')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(customer_agg['CustomerID'], customer_agg['UnitPrice'], label='Mean UnitPrice', marker='o')
    plt.xlabel('CustomerID')
    plt.ylabel('Aggregated Value')
    plt.title('Aggregated Plot of CustomerID vs. Unit Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # List of columns for which to estimate cardinality using HLL
    columns_to_estimate = ['InvoiceNo', 'StockCode', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']

    # Display all unique elements of each column
    unique_datasets, num_unique_elem = {}, []

    for column in columns_to_estimate:
        unique_values = updated_df[column].unique()
        unique_datasets[column] = pd.DataFrame(unique_values, columns=[column])
        num_unique_elem.append(len(unique_datasets[column]))
        print(f"Display unique values of {column}: ")
        print(unique_datasets[column])

    estimated_cardinality, avg_times = [], []

    num_processes = [1, 2, 4, 6, 8]  # [1, 2, 4, 6, 8, 10, 12]

    for num_process in num_processes:  # Change the number of processes as needed
        estimated_cardinality = []

        print(f"\nNumber of processes: {num_process}\nCalculating...")

        start_time = time.perf_counter()

        # Initialize HLL sketches for each column
        hll_sketches = {}

        # Parallel HLL computation for each column with the current number of processes
        for column in columns_to_estimate:
            column_data = updated_df[column].astype(str).tolist()
            hll_sketches[column] = parallel_hll(column_data, num_process)
            estimated_cardinality.append(len(hll_sketches[column]))

        end_time = time.perf_counter()
        avg_time = (end_time - start_time)
        avg_times.append(avg_time)

        print("Time taken:", avg_time, "seconds\n")

    print("In summary...")

    for num_process, avg_time in zip(num_processes, avg_times):
        print(f"Number of processes: {num_process}")
        print("Time taken:", avg_time, "seconds\n")

    # Plotting the results
    plt.plot([1, 2, 4, 6, 8], avg_times, marker='o')  # [1, 2, 4, 6, 8, 10, 12]
    plt.xlabel('Number of Processes')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Time vs. Number of Processes')
    plt.grid(True)
    plt.show()

    for i, column in enumerate(columns_to_estimate):
        # Display estimated cardinalities for each column
        print(f"Actual number of unique elements in '{column}': {num_unique_elem[i]}")
        print(f"Estimated cardinality of unique elements in '{column}': {estimated_cardinality[i]}")
        print()

    # Create an array of indices for the columns
    index = np.arange(len(columns_to_estimate))

    # Set the width of the bars
    bar_width = 0.35

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot actual unique elements
    plt.bar(index, num_unique_elem, bar_width, label='Number of Unique Elements', color='blue')

    # Plot estimated cardinality
    plt.bar(index + bar_width, estimated_cardinality, bar_width, label='Estimated Cardinality', color='orange')

    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('Count')
    plt.title('Comparison of Number of Unique Elements vs. Estimated Cardinality by Column')
    plt.xticks(index + bar_width / 2, columns_to_estimate, rotation=45)
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calculate the absolute difference between actual and estimated cardinality for each column
    cardinality_difference = [abs(actual - estimated) for actual, estimated in
                              zip(num_unique_elem, estimated_cardinality)]

    # Scale the absolute differences using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_cardinality_difference = scaler.fit_transform(np.array(cardinality_difference).reshape(-1, 1)).flatten()

    # Plotting the scaled absolute difference as a line graph
    plt.figure(figsize=(10, 6))
    plt.plot(columns_to_estimate, scaled_cardinality_difference, marker='o', color='red')
    plt.xlabel('Columns')
    plt.ylabel('Scaled Absolute Difference')
    plt.title('Scaled Absolute Difference between Actual and Estimated Cardinality by Column')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
