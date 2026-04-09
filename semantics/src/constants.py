DEFAULT_WEIGHT = 0.5

MEMORY_THRESHOLD_PERCENT = 0.95
ALLOCATED_MEMORY_GB = 128
MEMORY_THRESHOLD = MEMORY_THRESHOLD_PERCENT * ALLOCATED_MEMORY_GB


DAG_NODES_MAP = {'cancer': 5,
                 'earthquake': 5,
                 'survey': 6,
                 'asia': 8,
                 'sachs': 11,
                 'child': 20,
                 'insurance': 27}

DAG_EDGES_MAP = {'cancer': 4,
                 'earthquake': 4,
                 'survey': 6,
                 'asia': 8,
                 'sachs': 17,
                 'child': 25,
                 'insurance': 52}


ALPHA = 0.01
INDEP_TEST = 'fisherz'
SAMPLE_SIZE = 5000
SEED = 2024
