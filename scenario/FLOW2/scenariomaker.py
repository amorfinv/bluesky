from itertools import product

scen_dir = 'flowtime'

experiment_cases = {
    'concepts' : {
        'conflict'  : 'CONFLICTCLUSTERING',
        # 'intrusion' : 'INTRUSIONCLUSTERING',
        # 'live'      : 'LIVECLUSTERING',
    },
    'densities' : {
        '100',
        '200',
        '300',
        '400',
        '500',
    },
    'clusters' : {
        '4000',
    },
    'replanlimit':[
        '0',
    ],
    'replanratio':[
        '1',
    ],
    'graphweights':[
        '1-2-2',
    ],
    'flowupdaterates':[
        '10',
        '30',
        '60',
        '90',
    ],
    'seeds': [
        '748180',
        '825078',
        '102890',
        '824289',
        '466213',
        # '2730111717',
        # '2206979456',
        # '3283131972',
        # '2250510208',
        # '1752397453',
        # '1317504816',
        # '2000344349',
        # '2113055274',
        # '1205330928',
        # '1246124471',
        # '3241510684',
        # '2417284863',
        # '334020136',
        # '2744972216',
        # '2372088387',
    ],
}

exp_cases = list(
    product(
        experiment_cases['concepts'].keys(),
        experiment_cases['densities'],
        experiment_cases['clusters'],
        experiment_cases['replanlimit'],
        experiment_cases['replanratio'],
        experiment_cases['graphweights'],
        experiment_cases['flowupdaterates'],
        experiment_cases['seeds'],
    )
)

# LOG processing for indivdual experiments
filenames = []
for concept, density, cluster, replanlimit, replanratio, graph_weights, flowupdaterate, seed in exp_cases:

    cluster_plugin = experiment_cases['concepts'][concept]
    
    low_weight, medium_weight, high_weight = graph_weights.split('-')
    
    # different flow control
    flow_control = 'RANDOMFLOWCONTROL' if concept == 'random' else 'FLOWCONTROL'

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
        f'SETGRAPHWEIGHTS {low_weight},{medium_weight},{high_weight}',
        'ASAS ON',
        'CDMETHOD M2CD',
        'RESO M2CR',
        f'REPLANLIMIT {replanlimit}',
        f'REPLANRATIO {replanratio}',
        f'FLOWUPDATERATE {flowupdaterate}',
        'STARTLOGS',
        'STARTCDRLOGS',
        'STARTCLUSTERLOG',
        'STARTFLOWLOG',
        f'trafficnumber {density}',
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
    filename = f'{concept}_traf{density}_clust{cluster}_replanlimit{replanlimit}_replanratio{replanratio}_graphweights{graph_weights}_flowupdaterate{flowupdaterate}_seed{seed}_CRON.scn'

    # write the file
    with open(f'{scen_dir}/{filename}', 'w') as file:
        file.write(lines + '\n') 

    
    filenames.append(filename)


# at the end make a batch file for all of this
batch_lines = []
for filename in filenames:

    lines = f'00:00:00>SCEN {filename[:-4]}\n' + \
            f'00:00:00>PCALL FLOW2/{scen_dir}/{filename}\n'
    
    batch_lines.append(lines)

# create a general batch
lines = '\n'.join(batch_lines)
with open(f'conflictbatch.scn', 'w') as file:
    file.write(lines + '\n')  # Add a newline after each string

