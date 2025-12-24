# create logger
import logging
import os
from datetime import datetime
from pathlib import Path
import numpy as np


def logger_setup(output_file:str="", continue_logging=False):
    results_root = Path(__file__).resolve().parent.parent / "results"
    if output_file == "":
        temp_dir = Path(".temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        output_path = Path(output_file)
        if output_path.parent == Path("") or output_path.parent == Path("."):
            output_dir = results_root / output_path.stem
        else:
            output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_path.suffix:
            output_path = output_dir / output_path.name
        else:
            output_path = output_dir / f"{output_path.name}.log"

    file_mode = 'a' if continue_logging else 'w'
    # File logs should include timestamps so long-running jobs can be reconstructed.
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-8s %(module)-12s - %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S',
        filename=str(output_path),
        filemode=file_mode,
        force=True,
    )
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
