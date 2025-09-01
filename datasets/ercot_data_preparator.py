import pandas as pd
import os


def process_ercot_data(file_path: str, settlement_point_name: str) -> pd.DataFrame:
    """
    Reads ERCOT RT data from an Excel file (potentially multiple sheets),
    filters by settlement point, creates a 5-minute datetime index,
    handles potential duplicate timestamps, and interpolates the price
    to a 5-minute frequency.

    Args:
        file_path (str): The path to the ERCOT Excel file (e.g., 'ERCOT_RT_2024.xlsx').
        settlement_point_name (str): The exact name of the settlement point
                                     to filter for (e.g., 'HB_BUSAVG').

    Returns:
        pd.DataFrame: A DataFrame with a 5-minute datetime index and a single
                      column named after the settlement_point_name containing
                      the interpolated prices. Returns an empty DataFrame
                      if the file or settlement point data is not found or processed.
    """

    try:
        # Read all sheets from the Excel file
        # Use decimal=',' based on the screenshot showing comma as decimal separator
        all_sheets = pd.read_excel(file_path, sheet_name=None, decimal=",")

        # Combine data from all sheets
        # Filter out potentially empty or non-data sheets if necessary
        # Here we assume all sheets contain data in the expected format
        # Check if all sheets are DataFrames before concatenating
        list_of_dfs = [df for df in all_sheets.values() if isinstance(df, pd.DataFrame)]
        if not list_of_dfs:
            print(f"Error: No valid data found in any sheets of {file_path}")
            return pd.DataFrame()

        combined_df = pd.concat(list_of_dfs, ignore_index=True)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading or combining Excel sheets: {e}")
        return pd.DataFrame()

    # Filter for the specific settlement point name
    # Ensure column exists before filtering
    if "Settlement Point Name" not in combined_df.columns:
        print(f"Error: 'Settlement Point Name' column not found in the data.")
        return pd.DataFrame()

    df_filtered = combined_df[
        combined_df["Settlement Point Name"] == settlement_point_name
    ].copy()

    if df_filtered.empty:
        print(f"Warning: No data found for settlement point '{settlement_point_name}'")
        return pd.DataFrame()

    # Ensure required columns exist for datetime creation and price
    required_cols = [
        "Delivery Date",
        "Delivery Hour",
        "Delivery Interval",
        "Repeated Hour Flag",
        "Settlement Point Price",
    ]
    if not all(col in df_filtered.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_filtered.columns]
        print(f"Error: Missing required columns for processing: {missing}")
        return pd.DataFrame()

    # --- Create the Datetime Index ---
    # Delivery Hour is 1-24, Interval 1-4 (for 15-min intervals)
    # Hour 1 Interval 1 starts at 00:00, Interval 2 at 00:15, etc.
    # Hour 2 Interval 1 starts at 01:00, etc.
    # Correcting 1-based hour to 0-based hour for datetime construction
    try:
        df_filtered["Hour_0based"] = df_filtered["Delivery Hour"].astype(int) - 1

        # Calculate minutes offset within the hour (0, 15, 30, 45)
        df_filtered["Minutes_offset"] = (
            df_filtered["Delivery Interval"].astype(int) - 1
        ) * 15

        # Combine Date, Hour, and Minutes
        # pd.to_datetime can combine date and time components
        # Ensure Delivery Date is in a parsable format, handle potential errors
        df_filtered["Delivery Date"] = pd.to_datetime(
            df_filtered["Delivery Date"], errors="coerce"
        )
        df_filtered.dropna(
            subset=["Delivery Date"], inplace=True
        )  # Drop rows where date conversion failed

        df_filtered["datetime"] = (
            df_filtered["Delivery Date"]
            + pd.to_timedelta(df_filtered["Hour_0based"], unit="h")
            + pd.to_timedelta(df_filtered["Minutes_offset"], unit="m")
        )

        # Handle the 'Repeated Hour Flag' ('R') - typically adds an hour for the second occurrence
        # This is common during DST transitions (fall back)
        repeated_hour_mask = df_filtered["Repeated Hour Flag"] == "R"
        df_filtered.loc[repeated_hour_mask, "datetime"] = df_filtered.loc[
            repeated_hour_mask, "datetime"
        ] + pd.Timedelta(hours=1)

    except Exception as e:
        print(f"Error creating datetime index or handling repeated hour: {e}")
        return pd.DataFrame()

    # Select only the price column and set the datetime column as the index
    # Do this BEFORE the groupby mean step
    df_price = df_filtered[["Settlement Point Price", "datetime"]].copy()
    df_price.set_index("datetime", inplace=True)

    # Sort the index
    df_price.sort_index(inplace=True)

    # Handle potential duplicate timestamps after repeated hour adjustment
    # (e.g., if original data has issues) - take the mean price
    # Apply groupby mean ONLY to the 'Settlement Point Price' column
    df_price = (
        df_price.groupby(df_price.index)["Settlement Point Price"].mean().to_frame()
    )

    # --- Reindex and Interpolate ---
    # Create a full 5-minute index covering the data range
    if not df_price.empty:
        start_time = df_price.index.min()
        end_time = df_price.index.max()
        # Ensure end_time is at least start_time if only one data point exists
        if start_time > end_time:
            end_time = start_time

        full_5min_index = pd.date_range(start=start_time, end=end_time, freq="5T")

        # Reindex the price data to the full 5-minute index
        df_5min = df_price.reindex(full_5min_index)

        # Interpolate the missing values (which were introduced by reindexing)
        # Ensure the column exists before interpolating
        if "Settlement Point Price" not in df_5min.columns:
            print("Error: 'Settlement Point Price' column not found after reindexing.")
            return pd.DataFrame()

        df_5min[settlement_point_name] = df_5min["Settlement Point Price"].interpolate(
            method="linear"
        )

        # Drop the original 15-minute price column
        df_5min.drop(columns=["Settlement Point Price"], inplace=True)

        return df_5min.dropna()  # Drop any NaNs that might occur at the very start/end if interpolation couldn't reach

    else:
        return pd.DataFrame()  # Should have been caught earlier, but just in case


def concatenate_ercot_years(
    file_paths: list[str], settlement_point_name: str
) -> pd.DataFrame:
    """
    Processes and concatenates ERCOT RT data for a specific settlement point
    from multiple annual Excel files.

    Args:
        file_paths (list[str]): A list of paths to the ERCOT Excel files
                                (e.g., ['ERCOT_RT_2023.xlsx', 'ERCOT_RT_2024.xlsx']).
        settlement_point_name (str): The exact name of the settlement point
                                     to filter for (e.g., 'HB_BUSAVG').

    Returns:
        pd.DataFrame: A single DataFrame with a continuous 5-minute datetime index
                      and a single column named after the settlement_point_name
                      containing the interpolated prices from all files.
                      Returns an empty DataFrame if no data is successfully processed.
    """
    processed_dfs = []

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        if not os.path.exists(file_path):
            print(f"Warning: File not found at '{file_path}'. Skipping.")
            continue  # Skip to the next file if not found

        df = process_ercot_data(file_path, settlement_point_name)

        if not df.empty:
            processed_dfs.append(df)
            print(f"Successfully processed data from {file_path}")
        else:
            print(f"No data processed from {file_path} for {settlement_point_name}.")

    if not processed_dfs:
        print(
            f"Error: No data successfully processed from any of the provided files for {settlement_point_name}."
        )
        return pd.DataFrame()

    # Concatenate all successfully processed DataFrames
    # Pandas handles concatenation correctly based on the datetime index
    # The indices from different years will naturally align
    concatenated_df = pd.concat(processed_dfs)

    # Sort the index of the final concatenated DataFrame
    # This ensures the time series is strictly chronological, which is
    # important if files were processed out of chronological order, though
    # pd.concat on sorted indices usually handles this. Redundant but safe.
    concatenated_df.sort_index(inplace=True)

    # Handle potential duplicate timestamps that might arise from concatenation
    # (highly unlikely with correct processing, but safe)
    if not concatenated_df.index.is_unique:
        print(
            "Warning: Duplicate timestamps found after concatenation. Averaging prices for duplicates."
        )
        concatenated_df = concatenated_df.groupby(concatenated_df.index).mean()

    print("\nConcatenation complete.")
    print(f"Total data points: {len(concatenated_df)}")
    print(f"Start time: {concatenated_df.index.min()}")
    print(f"End time: {concatenated_df.index.max()}")

    return concatenated_df


if __name__ == "__main__":
    # Define the file paths for the years to concatenate
    year_files = [
        "ERCOT_RT_2023.xlsx",
        "ERCOT_RT_2024.xlsx",
    ]
    settlement_point = "HB_BUSAVG"

    # Process and concatenate the data
    combined_data = concatenate_ercot_years(year_files, settlement_point)

    output_filename = f"ERCOT_{settlement_point}_2023_2024_5min_interpolated.parquet"

    combined_data.to_parquet(output_filename, compression="snappy")
    print("Successful !")
