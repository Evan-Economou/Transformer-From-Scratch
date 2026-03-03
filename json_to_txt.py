import json
from pathlib import Path


folder_path = Path('speeches')
data = ''
with open("data/pres_data.txt", "w") as outfile:
    for file_path in folder_path.iterdir():
        if file_path.is_file():
            try:
                with open(file_path, 'r') as infile:
                    data = json.load(infile)
                    speeches = data["transcript"]
                    infile.close()
                    outfile.write(" " + speeches)
            except Exception as e:
                print(e)
                continue

    outfile.close() 


# try:
#     # Code that may raise an exception
#     result = 10 / 0 
#     print(result)
# except ZeroDivisionError:
#     # Code to run if a specific exception (ZeroDivisionError) occurs
#     print("Error: Cannot divide by zero!")
# except Exception as e:
#     # Code to handle other, more general exceptions and access the error message
#     print(f"An unexpected error occurred: {e}")   

# for key in data.keys():
#     print(key)

# print(data["transcript"])

# file_path = ''


# with open(file_path, 'w') as f:
#     f.writelines(strings_list