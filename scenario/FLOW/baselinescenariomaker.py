from itertools import product

scen_dir = 'city_wide'

experiment_cases = {
    'concepts' : {
        'baseline'    : ''

    },
    'densities' : {
        # '50',
        '100',
        # '150',
        '200',
        # '250',
        '300',
        # '350',
        '400',
        # '450',
        '500',
    },
    'clusters' : {
        # '1000',
        # '2500',
        '4000',
    },
    'replanlimit':[
        # '0',
        # '15',
        '30',
        # '60',
        # '120',
        # '360',
    ],
    'replanratio':[
        # '0.1',
        # '0.25',
        '0.5',
        #'0.75',
        #'1',
    ],
    'graphweights':[
        # '1-1.1-1.2',
        # '1-1.25-1.5',
        '1-2-2',
        # '1-2-4',
        # '1-3-9',
        # '1-10-100',
    ],
    'densitycutoff':[
        '0.25-0.5',
    ],
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
        experiment_cases['replanlimit'],
        experiment_cases['replanratio'],
        experiment_cases['graphweights'],
        experiment_cases['seeds'],
    )
)

# LOG processing for indivdual experiments
filenames = []
for concept, density, cluster, replanlimit, replanratio, graph_weights, seed in exp_cases:

    cluster_plugin = experiment_cases['concepts'][concept]
    
    low_weight, medium_weight, high_weight = graph_weights.split('-')
    
    
    # make a file for this case
    lines = [
        f'SEED {seed}',
        'streetsenable',
        'STOPSIMT 7200',
        'ASAS ON',
        'CDMETHOD M2CD',
        'RESO M2CR',
        'STARTLOGS',
        'STARTCDRLOGS',
        f'trafficnumber {density}',
        'CASMACHTHR 0',
        'FF'
    ]

    # add the prefix to all lines
    lines = ['00:00:00.00>' + line for line in lines]

    # join the lines with a new line charachter
    lines = '\n'.join(lines)

    # create a filename
    filename = f'{concept}_traf{density}_clust{cluster}_replanlimit{replanlimit}_replanratio{replanratio}_graphweights{graph_weights}_seed{seed}_CROFF.scn'

    # write the file
    with open(f'{scen_dir}/{filename}', 'w') as file:
        file.write(lines + '\n') 

    
    filenames.append(filename)


# at the end make a batch file for all of this
batch_lines = []
for filename in filenames:

    lines = f'00:00:00>SCEN {filename[:-4]}\n' + \
            f'00:00:00>PCALL {scen_dir}/{filename}\n'
    
    batch_lines.append(lines)

# create a general batch
lines = '\n'.join(batch_lines)
with open(f'citywidebaselinebatch.scn', 'w') as file:
    file.write(lines + '\n')  # Add a newline after each string

