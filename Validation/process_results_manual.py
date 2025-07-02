import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import os
from pathlib import Path

from amuse.units import units, constants, nbody_system # type: ignore
from amuse.lab import Particles, Particle # type: ignore

os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "true"

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))
plot_path = arbeit_path / 'Plots'

from Validation.validation_funcs import process_result, merge_h5_files, load_result

job_id = '4248304'
results_path = arbeit_path / f'Validation/val_results/{job_id}/mp_results'
output_path = arbeit_path / f'Validation/val_results/{job_id}'

h5_files = list(output_path.glob("*.h5"))
if len(h5_files) != 1:
    raise ValueError(f"Expected exactly one .h5 file in {output_path}, but found {len(h5_files)}.")
output_file = h5_files[0]

merge_h5_files(results_path, output_file, delete = True)

# process_result(output_path, output_filename, maj_param, log_error=FalseH, filter_outliers=False, loglog=False)
process_result(output_path, output_file.name, log_error=True, filter_outliers=False, loglog=True)
