# create logger
import logging
import os
from datetime import datetime
import numpy as np


def logger_setup(output_file:str="", continue_logging=False):
    if not os.path.exists('.temp'):
        os.makedirs('.temp')
    if output_file == "":
        output_file = f'.temp/{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    elif ".log" not in output_file: ## when one passes only the name of the file
        output_file = f'.temp/{output_file}.log'

    file_mode = 'a' if continue_logging else 'w'
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        datefmt='%m-%d %H:%M',
                        filename= output_file,
                        filemode=file_mode, force=True)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-8s %(module)-12s - %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    # format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    if len(logging.getLogger('').handlers) < 2:
        logging.getLogger('').addHandler(console)

# @profile
def get_freer_gpu():
    import torch
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    if len(memory_available)>0:
        return np.argmax(memory_available)
    elif torch.cuda.is_available():
        return 0

def random_stability(seed_value=0, deterministic=True, verbose=False):
    '''
        seed_value : int A random seed
        deterministic : negatively effect performance making (parallel) operations deterministic
    '''
    if verbose:
        print('Random seed {} set for:'.format(seed_value))
    try:
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        if verbose:
            print(' - PYTHONHASHSEED (env)')
    except:
        pass
    try:
        import random
        random.seed(seed_value)
        if verbose:
            print(' - random')
    except:
        pass
    try:
        import numpy as np
        np.random.seed(seed_value)
        if verbose:
            print(' - NumPy')
    except:
        pass
    # try:
    #     import torch
    #     torch.manual_seed(seed_value)
    #     torch.cuda.manual_seed_all(seed_value)
    #     if verbose:
    #         print(' - PyTorch')
    #     if deterministic:
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False
    #         if verbose:
    #             print('   -> deterministic')
    # except:
        pass