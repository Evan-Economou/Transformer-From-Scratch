import json
from pathlib import Path


folder_path = Path('speeches')
data = ''
with open("data/pres_data.txt", "w") as outfile:
    for file_path in folder_path.iterdir():
        if file_path.is_file():
            with open(file_path, 'r') as infile:
                data = json.load(infile)
                speeches = data["transcript"]
                infile.close()
                outfile.write(" " + speeches)
    outfile.close() 

            
                   

# for key in data.keys():
#     print(key)

# print(data["transcript"])

# file_path = ''


# with open(file_path, 'w') as f:
#     f.writelines(strings_list