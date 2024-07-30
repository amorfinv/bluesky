from itertools import product

scen_dir = 'conflict'

experiment_cases = {
    'concepts' : {
        'conflict'  : 'CONFLICTCLUSTERING'
    },
    'densities' : {
        '100',
        '200',
        '300',
        '400',
        '500',
    },
    'clusters' : {
        '1000',
        '2500',
        '4000',
        '5000',
    },
    'geotime' : {
        '30',
        '60',
        '120',
        '240',
        '480',
    },
    'seeds': [
        '748180',
        '825078',
        '102890',
        '824289',
        '466213'
    ],
}

exp_cases = list(
    product(
        experiment_cases['concepts'].keys(),
        experiment_cases['densities'],
        experiment_cases['clusters'],
        experiment_cases['geotime'],
        experiment_cases['seeds'],
    )
)

# LOG processing for indivdual experiments
filenames = []
for concept, density, cluster, geotime, seed in exp_cases:

    cluster_plugin = experiment_cases['concepts'][concept]

    # add seeds
    lines_seed = [f'SEED {seed}']

    flowseed = int(seed) + 1
    lines_seed.append(f'FLOWSEED {flowseed}')

    if concept == 'random':
        clustseed = int(seed) + 2
        lines_seed.append(f'CLUSTSEED {clustseed}')
    
    # make a file for this case
    lines = [
        'ENABLEFLOWCONTROL',
        'streetsenable',
        'STOPSIMT 7200',
        'ASAS ON',
        'CDMETHOD M2CD',
        'RESO M2CR',
        'STARTLOGS',
        'STARTCDRLOGS',
        'STARTCLUSTERLOG',
        'STARTFLOWLOG',
        f'trafficnumber {density}',
        f'GEOTIME {geotime}',
        f'SETCLUSTERDISTANCE {cluster}',
        'CASMACHTHR 0',
    ]

    if concept in ['conflict', 'intrusion']:
        lines.append('SETOBSERVATIONTIME 600')

    # add the fast forward
    lines.append('FF')

    # comvine seeds lines
    lines = lines_seed + lines

    # add the prefix to all lines
    lines = ['00:00:00.00>' + line for line in lines]

    # join the lines with a new line charachter
    lines = '\n'.join(lines)

    # create a filename
    filename = f'{concept}_traf{density}_clust{cluster}_geotime{geotime}_seed{seed}_CRON.scn'

    # write the file
    with open(f'{scen_dir}/{filename}', 'w') as file:
        file.write(lines + '\n') 

    
    filenames.append(filename)


# at the end make a batch file for all of this
batch_lines = []
for filename in filenames:

    lines = f'00:00:00>SCEN {filename[:-4]}\n' + \
            f'00:00:00>PCALL geovectoring/{scen_dir}/{filename}\n'
    
    batch_lines.append(lines)

# create a general batch
lines = '\n'.join(batch_lines)
with open(f'conflictbatch.scn', 'w') as file:
    file.write(lines + '\n')  # Add a newline after each string

