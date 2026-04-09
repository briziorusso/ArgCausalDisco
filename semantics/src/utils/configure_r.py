import os


def configure_r(default_rpath='/usr/bin/Rscript',
                default_r_libs='~/R/library'):
    """
    Configure R settings for the cdt package.
    Is necessary for SID metric evaluation.
    """
    import cdt

    rpath = os.environ.get('RPATH', default_rpath)

    cdt.SETTINGS.rpath = rpath

    # Configure R to run non-interactively and disable pagination
    os.environ['R_INTERACTIVE'] = 'FALSE'  # Force non-interactive mode
    os.environ['R_PAPERSIZE'] = 'letter'   # Avoid unnecessary prompts
    os.environ['PAGER'] = 'cat'            # Disable R’s pager (no "q" prompt)

    # Set R options to suppress interactive checks (e.g., package updates)
    os.environ['R_OPTS'] = '--no-save --no-restore --quiet'  # Silent execution

    r_libs = os.environ.get('R_LIB_DIR', default_r_libs)
    os.environ['R_LIBS_USER'] = os.path.expanduser(r_libs)
