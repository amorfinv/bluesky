from itertools import product

scen_dir = 'parameter_tuning'

experiment_cases = {
    'concepts' : {
        'conflict'  : 'CONFLICTCLUSTERING',
        'intrusion' : 'INTRUSIONCLUSTERING',
        'live'      : 'LIVECLUSTERING',
        # 'random'    : 'RANDOMCLUSTERING'
    },
    'densities' : {
        '300',
    },
    'clusters' : {
        '2500',
    },
    'replanlimit':[
        '0',
        '15',
        '30',
        '60',
        '120',
        '360',
    ],
    'replanratio':[
        '0.25',
        '0.5',
        '0.75',
        '1',
    ],
    'seeds': [
        '748180',
        '825078',
        '102890',
        '824289',
        '466213'
    ]
}

exp_cases = list(
    product(
        experiment_cases['concepts'].keys(),
        experiment_cases['densities'],
        experiment_cases['clusters'],
        experiment_cases['replanlimit'],
        experiment_cases['replanratio'],
        experiment_cases['seeds'],
    )
)

# LOG processing for indivdual experiments
filenames = []
for concept, density, cluster, replanlimit, replanratio, seed in exp_cases:

    cluster_plugin = experiment_cases['concepts'][concept]

    # different flow control
    flow_control = 'RANDOMFLOWCONTROL' if concept == 'random' else 'FLOWCONTROL'
    
    # make a file for this case
    lines = [
        f'PLUGIN LOAD {cluster_plugin}',
        f'PLUGIN LOAD {flow_control}',
        'FF',
        'ENABLEFLOWCONTROL',
        'streetsenable',
        'STOPSIMT 7200',
        'ASAS ON',
        'CDMETHOD M2CD',
        f'REPLANLIMIT {replanlimit}',
        f'REPLANRATIO {replanratio}',
        'STARTLOGS',
        'STARTCDRLOGS',
        'STARTCLUSTERLOG',
        'STARTFLOWLOG',
        f'trafficnumber {density}',
        f'SEED {seed}',
        f'SETCLUSTERDISTANCE {cluster}',
        'CASMACHTHR 0',
    ]

    if concept in ['conflict', 'intrusion']:
        lines.append('SETOBSERVATIONTIME 600')

    # add the prefix to all lines
    lines = ['00:00:00.00>' + line for line in lines]

    # join the lines with a new line charachter
    lines = '\n'.join(lines)

    # create a filename
    filename = f'{concept}_traf{density}_clust{cluster}_replanlimit{replanlimit}_replanratio{replanratio}_seed{seed}.scn'

    # write the file
    with open(f'{scen_dir}/{filename}', 'w') as file:
        file.write(lines + '\n') 

    
    filenames.append(filename)


# at the end make a batch file for all of this
batch_lines = []
for filename in filenames:

    lines = f'00:00:00>SCEN {filename[:-4]}_CROFF\n' + \
            f'00:00:00>PCALL FLOW/{scen_dir}/{filename}\n'
    
    batch_lines.append(lines)

# create a general batch
lines = '\n'.join(batch_lines)
with open(f'batch.scn', 'w') as file:
    file.write(lines + '\n')  # Add a newline after each string

