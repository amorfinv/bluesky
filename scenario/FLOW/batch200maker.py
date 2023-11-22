import os

folders = ['conflict', 'intrusion', 'live']
batches = {folder:[] for folder in folders}


# get list of files
files_in_dir = os.listdir('live')

live_lines = []
# step 1 go through one folder
for file_name in files_in_dir:

    if 'traf200_' in file_name or 'clust1500_' in file_name:
        pass
    else:
        continue

    lines = f'00:00:00>SCEN live_{file_name[:-4]}\n' + \
            f'00:00:00>PCALL FLOW/live/{file_name}\n'
    
    live_lines.append(lines)

# create a file
lines = '\n'.join(live_lines)
# Open the file in write mode and write each string from the list
with open(f'trialbatch.scn', 'w') as file:
    file.write(lines + '\n')  # Add a newline after each string
