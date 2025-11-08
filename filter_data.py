import pandas as pd

HOME_DIR = "data"
filenames = ['Fall 2022.csv', 'Fall 2023.csv', 'Fall 2024.csv', 'Spring 2022.csv', 'Spring 2023.csv', 'Spring 2024.csv']

def filter_data_from_csv(file):
    data = pd.read_csv(f"{HOME_DIR}/{file}")
    
    # Convert Catalog Nbr to numeric
    
    if file != 'Spring 2022.csv':
        data['Catalog Nbr'] = pd.to_numeric(data['Catalog Nbr'], errors='coerce')
        filtered_data = data[
            (data['Subject'] == 'CS') &
            (data['Catalog Nbr'].between(6000, 7000))
        ]
    else:
        data['Catalog Number'] = pd.to_numeric(data['Catalog Number'], errors='coerce')
        filtered_data = data[
            (data['Subject'] == 'CS') &
            (data['Catalog Number'].between(6000, 7000))
        ]


    filtered_data.to_csv(f"{HOME_DIR}/filtered_{file}", index=False)
    print(f"Filtered {file} — saved to filtered_{file}")

for filename in filenames:
    print("Processing file:", filename)
    filter_data_from_csv(filename)
