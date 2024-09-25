from itertools import product

scen_dir = 'vienna'

experiment_cases = {
    'concepts' : {
        'baseline'    : ''

    },
    'densities' : {
        '300',
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
    'densitycutoff':[
        '0.25-0.5',
    ],
    'seeds': [
        '748180',
        '825078',
        '102890',
        '824289',
        '466213',  
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
    filename = f'{concept}_traf{density}_clust{cluster}_replanlimit{replanlimit}_replanratio{replanratio}_graphweights{graph_weights}_seed{seed}_CRON.scn'

    # write the file
    with open(f'{scen_dir}/{filename}', 'w') as file:
        file.write(lines + '\n') 

    
    filenames.append(filename)


# at the end make a batch file for all of this
batch_lines = []
for filename in filenames:

    lines = f'00:00:00>SCEN {filename[:-4]}\n' + \
            f'00:00:00>PCALL FLOWVIENNA/{scen_dir}/{filename}\n'
    
    batch_lines.append(lines)

# create a general batch
lines = '\n'.join(batch_lines)
with open(f'baselinebatch.scn', 'w') as file:
    file.write(lines + '\n')  # Add a newline after each string

