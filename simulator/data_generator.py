import argparse
import os
import yaml
import numpy as np
from .data_simulator import generateSample, simulateFlats, simulateStripes
from .realdata_loader import generate_real_data


def makeDirectories(dataDir, sampleNo, shifts, mode):
    """Make sub-directories for data generation.
    Sub-directories created depend on `mode`.
    Parameters:
        dataDir : str
            Root directory under which all sub-directories will be created.
        sampleNo : int
            Sample Number. If mode is one of ['simple', 'complex', 'raw',
            'paired'], a sub-directory named after this parameter will be
            created.
        shifts : int
            Number of vertical shifts in data when scanned. If mode is one of
            ['simple', 'complex', 'raw'], a sub-directory will be created for
            each and every shift.
        mode : str
            The mode that determines how sub-directories will be created.
            Must be one of:
                ['complex', 'raw']:
                    For synthetic data with noise ('complex'), or real data
                    loaded directly from an HDF file without any pre- or post-
                    processing ('raw').
                    Creates the following structure:
                        <dataDir>
                        ├── <sampleNo>
                        │   ├── clean
                        │   ├── shift00
                        │   ├── shift01
                        ... ...
                'simple':
                    For synthetic data with no noise.
                    Creates the following structure:
                        <dataDir>
                        ├── <sampleNo>
                        │   ├── clean
                        │   └── stripe
                        ...
                'paired':
                    For real life data that simulates artifacts in clean
                    sinograms, and creates clean sinograms for sinograms with
                    real artifacts.
                    Creates the following structure:
                        <dataDir>
                        ├── <sampleNo>
                        │   ├── fake_artifacts
                        │   │   ├── clean
                        │   │   └── stripe
                        │   └── real_artifacts
                        │       ├── clean
                        │       └── stripe
                        ...
                'dynamic':
                    For dynamic tomographic scans. All "frames" of a dynamic
                    scan are stored under one directory.
                    Creates the following structure:
                        <dataDir>
                            └── dynamic
    Returns:
        str
            The main root of the created directories. For modes ['simple',
            'complex', 'raw], this is '<dataDir>/<sampleNo>'.
            For modes ['paired', 'dynamic'], this is <dataDir>.
    """
    mainPath = dataDir
    if mode in ['complex', 'raw']:
        mainPath = os.path.join(dataDir, f'{sampleNo:04}')
        os.makedirs(mainPath, exist_ok=True)
        cleanPath = os.path.join(mainPath, 'clean')
        os.makedirs(cleanPath, exist_ok=True)
        for shift in range(shifts):
            shiftPath = os.path.join(mainPath, f'shift{shift:02}')
            os.makedirs(shiftPath, exist_ok=True)
    elif mode == 'simple':
        mainPath = os.path.join(dataDir, f'{sampleNo:04}')
        os.makedirs(mainPath, exist_ok=True)
        cleanPath = os.path.join(mainPath, 'clean')
        os.makedirs(cleanPath, exist_ok=True)
        stripePath = os.path.join(mainPath, 'stripe')
        os.makedirs(stripePath, exist_ok=True)
    elif mode in ['paired', 'patch']:
        mainPath = os.path.join(dataDir, f'{sampleNo:04}')
        cleanRealArtPath = os.path.join(mainPath, 'real_artifacts', 'clean')
        stripeRealArtPath = os.path.join(mainPath, 'real_artifacts', 'stripe')
        os.makedirs(cleanRealArtPath, exist_ok=True)
        os.makedirs(stripeRealArtPath, exist_ok=True)
        cleanFakeArtPath = os.path.join(mainPath, 'fake_artifacts', 'clean')
        stripeFakeArtPath = os.path.join(mainPath, 'fake_artifacts', 'stripe')
        os.makedirs(cleanFakeArtPath, exist_ok=True)
        os.makedirs(stripeFakeArtPath, exist_ok=True)
    elif mode == 'dynamic':
        dynamicPath = os.path.join(dataDir, 'dynamic')
        os.makedirs(dynamicPath, exist_ok=True)
    else:
        raise ValueError(
            "Mode should be one of [simple, complex raw, paired, dynamic]. "
            f"Instead got '{mode}'.")
    return mainPath


def get_args():
    parser = argparse.ArgumentParser(description="Create directories and "
                                                 "generate samples of data.")
    parser.add_argument('-m', '--mode', type=str, default='complex',
                        help="Type of data to generate. Must be one of: "
                             "['simple', 'complex', 'raw', 'paired', 'dynamic'"
                             ", 'patch']")
    parser.add_argument('-r', '--root', type=str, default=None,
                        help="Directory to save data in.")
    parser.add_argument('-S', "--samples", type=int, default=1,
                        help="Number of samples to generate.")
    parser.add_argument("--start", type=int, default=0,
                        help="Sample number to begin counting at (useful if "
                             "some data has already been generated).")
    parser.add_argument('-s', "--shifts", type=int, default=1,
                        help="Number of vertical shifts for each sample. "
                             "Only affects modes 'complex', 'raw', 'paired', "
                             "and 'patch'.")
    parser.add_argument('-p', "--shiftstep", type=int, default=5,
                        help="Shift step of a sample in pixels. "
                           "Only affects modes 'complex', 'raw' and 'paired'.")
    parser.add_argument('-N', "--size", type=int, default=256,
                        help="Size of sample generated (cubic). "
                             "Only affects modes 'simple' and 'complex'.")
    parser.add_argument('-o', "--objects", type=int, default=300,
                        help="Number of objects to generate for each sample. "
                             "Only affects modes 'simple' and 'complex'.")
    parser.add_argument('-f', "--flatsnum", type=int, default=20,
                        help="Number of the flat fields to generate. "
                             "Only affects 'complex' mode.")
    parser.add_argument('-I', "--I0", type=int, default=40000,
                        help="Full-beam photon flux intensity. "
                             "Only affects 'complex' mode.")
    parser.add_argument("--pipeline", type=str, default='tomo_pipeline.yml',
                        help="HTTomo YAML pipeline file for loading HDF data. "
                             "Only affects Real-life data modes.")
    parser.add_argument("--hdf-file", type=str, default=None,
                        help="Nexus file to load HDF data from. "
                             "Only affects Real-life data modes.")
    parser.add_argument('-C', "--chunk-size", type=int, default=243,
                        help="Size of chunks to load real-life data in. "
                             "Only affects Real-life data modes.")
    parser.add_argument("--flats", type=str, default=None,
                        help="Path to HDF file containing flat & dark fields. "
                             "Only affects Real-life data modes.")
    parser.add_argument("--mask", type=str, default=None,
                        help="Path to mask on stripe locations in data. "
                             "If left blank, a mask will be generated. Only "
                             "affects modes 'paired' and 'patch'.")
    parser.add_argument("--frame-angles", type=int, default=900,
                        help="Number of angles per 'frame' of a scan. "
                             "Only affects 'dynamic' mode.")
    parser.add_argument("--patch-size", type=int, default=[1801, 256], nargs=2,
                        help="Size of patches to split data into. Only "
                             "affects 'patch' mode.")
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="Print some extra information when running")
    return parser.parse_args()


if __name__ == '__main__':
    current_dir = os.path.basename(os.path.abspath(os.curdir))
    if current_dir != 'NoStripesNet':
        raise RuntimeError(f"Current Directory should be '.../NoStripesNet/'. "
                           f"Instead got '{current_dir}'.\n"
                           f"If Current Directory is not 'NoStripesNet', "
                           f"file and directory creation will be incorrect.")
    args = get_args()
    root = args.root
    if root is None:
        root = os.path.join(os.curdir, 'data')
    samples = args.samples
    shifts = args.shifts
    size = args.size
    objects = args.objects
    I0 = args.I0
    flatsnum = args.flatsnum
    shift_step = args.shiftstep
    angles_per_frame = args.frame_angles
    verbose = args.verbose
    start = args.start
    total_samples = start + samples
    for sampleNo in range(start, total_samples):
        if verbose:
            print(f"Generating sample [{sampleNo:04} / {total_samples-1:04}]")
        mainPath = makeDirectories(root, sampleNo, shifts, args.mode)
        if args.mode == 'simple':
            cleanPath = os.path.join(mainPath, 'clean')
            sample_clean = generateSample(size,
                                          objects,
                                          output_path=cleanPath,
                                          sampleNo=sampleNo,
                                          verbose=verbose)
            # TO-DO: Turn all the parameters below into CLI arguments
            sample_shifts = simulateStripes(sample_clean,
                                            percentage=1.2,
                                            max_thickness=3.0,
                                            intensity=0.25,
                                            kind='mix',
                                            variability=0,
                                            output_path=mainPath,
                                            sampleNo=sampleNo,
                                            verbose=verbose)
        elif args.mode == 'complex':
            # don't save 'clean' sample after it's generated
            # instead save 'clean' sample after flat noise has been added
            sample_clean = generateSample(size,
                                          objects,
                                          sampleNo=sampleNo,
                                          verbose=verbose)
            sample_shifts = simulateFlats(sample_clean,
                                          size,
                                          I0=I0,
                                          flatsnum=flatsnum,
                                          shifted_positions_no=shifts,
                                          shift_step=shift_step,
                                          output_path=mainPath,
                                          sampleNo=sampleNo,
                                          verbose=verbose)
        elif args.mode in ['raw', 'paired', 'dynamic', 'patch']:
            if args.hdf_file is None:
                raise ValueError(
                    "HDF File is None. Please include '--hdf-file' option.")
            mask = args.mask
            if mask is not None:
                mask = np.load(mask)
            patch_size = args.patch_size
            generate_real_data(mainPath,
                               args.hdf_file,
                               args.mode,
                               args.chunk_size,
                               sampleNo,
                               shifts,
                               args.flats,
                               mask=mask,
                               frame_angles=angles_per_frame,
                               patch_size=patch_size)
        else:
            raise ValueError(f"Option '--mode' should be one of 'simple', "
                             f"'complex', 'raw', 'paired', 'dynamic', 'patch'."
                             f" Instead got '{args.mode}'.")
