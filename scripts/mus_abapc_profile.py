#!/usr/bin/env python3
"""Profile ABAPC removal vs MUS solving across node sizes.

This script provides a friendly CLI wrapper around tests_mus.py functionality,
allowing easy configuration of node sizes, timeouts, and MUS/MCS limits without
the verbose pytest command syntax.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import test infrastructure
import tests_mus  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile ABAPC removal vs MUS solving time/memory across node sizes. "
        "This is a CLI wrapper around tests_mus.py functionality.",
        epilog="Example: python scripts/mus_abapc_profile.py --node-sizes 5,7 --max-muses '' --solve-timeout 120"
    )
    parser.add_argument(
        "--node-sizes",
        type=str,
        default="5,6,7,8",
        help="Comma-separated node sizes to profile (default: 5,6,7,8)"
    )
    parser.add_argument(
        "--solve-timeout",
        type=int,
        default=60,
        help="Wall-clock timeout (seconds) per instance (default: 60)"
    )
    parser.add_argument(
        "--edge-per-node",
        type=int,
        default=2,
        help="Edge-per-node multiplier for random DAG generation (default: 2)"
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        default="ER",
        help="Random graph type for PC simulation (default: ER)"
    )
    parser.add_argument(
        "--out-n",
        type=int,
        default=0,
        help="Clingo -n model bound for ABAPC removal (0=all; default: 0)"
    )
    parser.add_argument(
        "--opt-mode",
        type=str,
        default="optN",
        help="Clingo opt_mode for ABAPC removal (default: optN)"
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=2004,
        help="Base seed for random graph generation (default: 2004)"
    )
    parser.add_argument(
        "--max-muses",
        type=str,
        default="",
        help="Max MUS to enumerate: '' (empty)=omit -n flag (WASP default: 1 MUS), '0'=unlimited, '>0'=limit (default: '')"
    )
    parser.add_argument(
        "--mcs-threshold",
        type=int,
        default=0,
        help="CAMUS MCS threshold (0=unlimited, default: 0)"
    )
    parser.add_argument(
        "--mus-threshold",
        type=int,
        default=0,
        help="CAMUS MUS threshold (0=unlimited, default: 0)"
    )
    parser.add_argument(
        "--emit-lp",
        type=str,
        default="",
        help="Path to emit complete MUS program (use {n} for node count placeholder, e.g., /tmp/test_{n}node.lp)"
    )
    parser.add_argument(
        "--emit-lp-dir",
        type=str,
        default="",
        help="Auto-name and write adorned MUS programs to this directory (names: mus_<n_nodes>_<seed>.lp)"
    )
    parser.add_argument(
        "--rep-unsat",
        type=int,
        default=0,
        help="Retries with new seeds when an instance is SAT (no conflicts) (default: 0)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging noise (set logging level to WARNING for tidier copy/paste output)"
    )
    
    args = parser.parse_args(argv)

    # Configure test suite globals
    tests_mus.MUS_SOLVE_TIMEOUT = args.solve_timeout
    tests_mus.MUS_NODE_SIZES = tests_mus._parse_node_sizes(args.node_sizes)
    tests_mus.MUS_EDGE_PER_NODE = args.edge_per_node
    tests_mus.MUS_MAX_MUSES = args.max_muses
    tests_mus.MUS_MCS_THRESHOLD = args.mcs_threshold
    tests_mus.MUS_MUS_THRESHOLD = args.mus_threshold
    tests_mus.MUS_EMIT_LP = args.emit_lp
    tests_mus.MUS_GRAPH_TYPE = args.graph_type
    tests_mus.MUS_ABAPC_OUT_N = args.out_n
    tests_mus.MUS_ABAPC_OPT_MODE = args.opt_mode

    # Setup logging with custom filter for quiet mode
    class QuietModeFilter(logging.Filter):
        """Filter to suppress noisy messages in quiet mode."""
        def __init__(self, quiet: bool):
            super().__init__()
            self.quiet = quiet
            
        def filter(self, record: logging.LogRecord) -> bool:
            if not self.quiet:
                return True
            
            msg = record.getMessage()
            
            # Suppress probability warnings (can be INFO or WARNING level)
            if 'Probability values don' in msg or 'Differ by:' in msg or 'Adjusting values' in msg:
                return False
            
            # Suppress these verbose INFO messages
            suppress_patterns = [
                'Removal iteration',
                'solve budget',
                'Times: {',
                'Recompiling and regrounding',
                'Number of total independence',
                'Number of facts from PC:',
                'Number of wrong facts:',
                'Fully directed edges',
                'Undirected edges',
                'True DAG:',
                'Initial solve budget',
                'Running MUS solver',
                'Compiling the program',
                'Adding Specific Rules',
                'of all node pairs',
                'active paths added',
                'Measuring grounding time',
                '   Grounding...',
                '   Grounding time:',
                'Measured grounding time',
                'Sim config:',
                'Simulating data with',
                'Running PC algorithm',
                'Starting Orientation',
                'Starting propagation',
                'Step 1:',
                'Step 3:',
                'Running CausalABA',
                'Parsed',
                'Found',
                'Dumping specific',
                'Dumped',
                'MUS Build time:',
                '   Solving...',
                'Number of models:',
                'Number of facts removed:',
                'You can use',
                'You are using',
                'backend',
                'Total MUS found:',
                'Total MCS found:',
                'MUS Solve time:',
                'MUS facts resolved:',
                'MUS facts (resolved):',
                'Step 2:',
                'ABAPC removed',
                'Wrong among',
            ]
            
            for pattern in suppress_patterns:
                if pattern in msg:
                    return False
            
            return True
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stdout,
        force=True,
    )
    
    # Add filter to root logger and all existing loggers
    quiet_filter = QuietModeFilter(args.quiet)
    logging.getLogger().addFilter(quiet_filter)
    
    # Apply to all existing loggers as well
    if args.quiet:
        for logger_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            if isinstance(logger, logging.Logger):
                logger.addFilter(QuietModeFilter(True))
        
        # Override tests_mus.logger_setup to ensure new loggers also get the filter
        original_logger_setup = tests_mus.logger_setup
        def _quiet_logger_setup(*args, **kwargs):
            original_logger_setup(*args, **kwargs)
            # Apply filter to root and all loggers after setup
            for logger_name in list(logging.Logger.manager.loggerDict.keys()) + ['']:
                logger = logging.getLogger(logger_name if logger_name else None)
                if isinstance(logger, logging.Logger):
                    logger.addFilter(QuietModeFilter(True))
        tests_mus.logger_setup = _quiet_logger_setup
    
    # Also suppress Python warnings module warnings
    if args.quiet:
        warnings.filterwarnings('ignore', message='.*Probability values.*')
        warnings.filterwarnings('ignore', message='.*Differ by.*')
        warnings.filterwarnings('ignore', message='.*Adjusting values.*')
        
        # Redirect stderr to suppress clingo info messages (do this early and aggressively)
        import io
        sys.stderr = io.open(os.devnull, 'w')
        
        # Also try at file descriptor level
        try:
            import os as os_module
            stderr_fd = os_module.dup(2)  # Save original stderr
            devnull_fd = os_module.open(os.devnull, os_module.O_WRONLY)
            os_module.dup2(devnull_fd, 2)  # Redirect fd 2 (stderr) to /dev/null
        except:
            pass  # If this fails, we still have the Python-level redirect
        
        # Capture and suppress warnings that bypass the logging system
        original_showwarning = warnings.showwarning
        def _suppress_showwarning(message, category, filename, lineno, file=None, line=None):
            msg_str = str(message)
            if 'Probability' in msg_str or 'Differ by' in msg_str:
                return  # Suppress
            original_showwarning(message, category, filename, lineno, file, line)
        warnings.showwarning = _suppress_showwarning
        
        # Also monkey-patch logging.Logger.warning to suppress probability warnings
        _original_logger_warning = logging.Logger.warning
        def _patched_warning(self, msg, *args, **kwargs):
            msg_str = str(msg)
            if 'Probability' in msg_str or 'Differ by' in msg_str or 'Adjusting values' in msg_str:
                return  # Suppress
            _original_logger_warning(self, msg, *args, **kwargs)
        logging.Logger.warning = _patched_warning
        
        # Disable tqdm progress bars
        try:
            import tqdm
            # Patch tqdm to disable by default
            original_tqdm_init = tqdm.tqdm.__init__
            def _patched_tqdm_init(self, *args, **kwargs):
                kwargs['disable'] = True
                original_tqdm_init(self, *args, **kwargs)
            tqdm.tqdm.__init__ = _patched_tqdm_init
        except ImportError:
            pass  # tqdm not installed, no problem

    logging.info(
        f"Profile config: node_sizes={tests_mus.MUS_NODE_SIZES}, solve_timeout={tests_mus.MUS_SOLVE_TIMEOUT}s, "
        f"edge_per_node={tests_mus.MUS_EDGE_PER_NODE}, graph_type={tests_mus.MUS_GRAPH_TYPE}, "
        f"opt_mode={tests_mus.MUS_ABAPC_OPT_MODE}, out_n={tests_mus.MUS_ABAPC_OUT_N}, "
        f"max_muses={tests_mus.MUS_MAX_MUSES!r}, mcs_threshold={tests_mus.MUS_MCS_THRESHOLD}, mus_threshold={tests_mus.MUS_MUS_THRESHOLD}, "
        f"emit_lp={tests_mus.MUS_EMIT_LP or '(none)'}"
    )

    # Create test instance
    test_instance = tests_mus.TestMUSAnalysis()

    # Run profile for each node size
    for n_nodes in tests_mus.MUS_NODE_SIZES:
        curr_seed = args.seed_base
        attempts = 0
        
        while attempts < (args.rep_unsat + 1):
            if not args.quiet:
                logging.info(f"\n{'='*90}")
                logging.info(f"Profiling n_nodes={n_nodes} (seed={curr_seed})")
                logging.info('='*90)
            
            # Handle emit-lp-dir
            if args.emit_lp_dir:
                Path(args.emit_lp_dir).mkdir(parents=True, exist_ok=True)
                tests_mus.MUS_EMIT_LP = str(Path(args.emit_lp_dir) / f"mus_{n_nodes}_{curr_seed}.lp")
            
            # Run the test
            try:
                result = test_instance._run_mus_mcs_for_size(n_nodes, seed=curr_seed)
                # Check if we should retry (instance was SAT)
                if result.get('was_sat') and attempts < args.rep_unsat:
                    logging.info(f"Instance was SAT, retrying with new seed...")
                    curr_seed += 1
                    attempts += 1
                    continue
                else:
                    # Success or no more retries
                    break
            except (TimeoutError, AssertionError) as e:
                # Timing summary was already printed; timeout is indicated in results line
                logging.debug(f"Caught {type(e).__name__}: {e}")
                break
            except Exception as e:
                logging.error(f"Failed to profile n_nodes={n_nodes}: {e}", exc_info=True)
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
