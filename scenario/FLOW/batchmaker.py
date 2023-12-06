import os

scn_folder = 'observation'

folders = ['conflict', 'intrusion']
batches = {folder:[] for folder in folders}

for case_folder in folders:

    # get list of files
    files_in_dir = os.listdir(scn_folder + '/' + case_folder)
    
    # step 1 go through one folder
    for file_name in files_in_dir:

        lines = f'00:00:00>SCEN {case_folder}_{file_name[:-4]}_CROFF\n' + \
                f'00:00:00>PCALL FLOW/{scn_folder}/{case_folder}/{file_name}\n'
        
        batches[case_folder].append(lines)


# create a file
general_lines = []
for folder in folders:
    lines = '\n'.join(batches[folder])
    # Open the file in write mode and write each string from the list
    with open(f'{folder}batch.scn', 'w') as file:
        file.write(lines + '\n')  # Add a newline after each string
    
    general_lines.append(lines)

# create a general batch
lines = '\n'.join(general_lines)
with open(f'batch.scn', 'w') as file:
    file.write(lines + '\n')  # Add a newline after each string

