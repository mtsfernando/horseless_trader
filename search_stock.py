import csv

def csv_to_data_arr(csv_file):
    result_array = []
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader) # Skip the header row
            for row in reader:
                if len(row) >= 2:  # Ensure there are at least two columns
                    symbol = row[0]
                    security_name = row[1]
                    result_array.append(f"[{symbol}] {security_name}")

        return result_array

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def search_data(searchterm: str) -> list:
    if not searchterm:
        return data 
    searchterm = searchterm.lower()
    return [item for item in data if searchterm in item.lower()]

data = csv_to_data_arr("assets/stock_data.csv")