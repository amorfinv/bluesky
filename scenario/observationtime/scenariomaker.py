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
        '4000',
    },
    'observationtime' : {
        '60',
        '300',
        '600',
        '900',
        '1200',
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
        experiment_cases['observationtime'],
        experiment_cases['seeds'],
    )
)

# LOG processing for indivdual experiments
filenames = []
for concept, density, cluster, observationtime, seed in exp_cases:

    cluster_plugin = experiment_cases['concepts'][concept]

    # add seeds
    lines_seed = [f'SEED {seed}']

    # make a file for this case
    lines = [
        f'SEED {seed}',
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
        f'SETCLUSTERDISTANCE {cluster}',
        f'SETOBSERVATIONTIME {observationtime}',
        'CASMACHTHR 0',
    ]

    # add the fast forward
    lines.append('FF')

    # add the prefix to all lines
    lines = ['00:00:00.00>' + line for line in lines]

    # join the lines with a new line charachter
    lines = '\n'.join(lines)

    # create a filename
    filename = f'{concept}_traf{density}_clust{cluster}_observationtime{observationtime}_seed{seed}_CRON.scn'

    # write the file
    with open(f'{scen_dir}/{filename}', 'w') as file:
        file.write(lines + '\n') 

    
    filenames.append(filename)


# at the end make a batch file for all of this
batch_lines = []
for filename in filenames:

    lines = f'00:00:00>SCEN {filename[:-4]}\n' + \
            f'00:00:00>PCALL observationtime/{scen_dir}/{filename}\n'
    
    batch_lines.append(lines)

# create a general batch
lines = '\n'.join(batch_lines)
with open(f'conflictbatch.scn', 'w') as file:
    file.write(lines + '\n')  # Add a newline after each string

