# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""

import os
import random
import sys
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
from absl import app, flags, logging

from alphafold.data import pipeline, pipeline_multimer
from alphafold.model import config, data, model
from alphafold.predict_structure import ModelsToRelax, predict_structure
from alphafold.relax import relax

# Internal import (7716).

logging.set_verbosity(logging.INFO)


flags.DEFINE_string("precomputed_msa", None, "MSA to use for this run")

flags.DEFINE_string("data_dir", None, "Path to directory of supporting data.")
flags.DEFINE_string(
    "output_dir", None, "Path to a directory that will " "store the results."
)
flags.DEFINE_enum(
    "model_preset",
    "monomer",
    ["monomer", "monomer_casp14", "monomer_ptm", "multimer"],
    "Choose preset model configuration - the monomer model, "
    "the monomer model with extra ensembling, monomer model with "
    "pTM head, or multimer model",
)
flags.DEFINE_boolean(
    "benchmark",
    False,
    "Run multiple JAX model evaluations "
    "to obtain a timing that excludes the compilation time, "
    "which should be more indicative of the time required for "
    "inferencing many proteins.",
)
flags.DEFINE_integer(
    "random_seed",
    None,
    "The random seed for the data "
    "pipeline. By default, this is randomly generated. Note "
    "that even if this is set, Alphafold may still not be "
    "deterministic, because processes like GPU inference are "
    "nondeterministic.",
)
flags.DEFINE_integer(
    "num_multimer_predictions_per_model",
    5,
    "How many "
    "predictions (each with a different random seed) will be "
    "generated per model. E.g. if this is 2 and there are 5 "
    "models then there will be 10 predictions per input. "
    "Note: this FLAG only applies if model_preset=multimer",
)
flags.DEFINE_enum_class(
    "models_to_relax",
    ModelsToRelax.BEST,
    ModelsToRelax,
    "The models to run the final relaxation step on. "
    "If `all`, all models are relaxed, which may be time "
    "consuming. If `best`, only the most confident model "
    "is relaxed. If `none`, relaxation is not run. Turning "
    "off relaxation might result in predictions with "
    "distracting stereochemical violations but might help "
    "in case you are having issues with the relaxation "
    "stage.",
)
flags.DEFINE_boolean(
    "use_gpu_relax",
    None,
    "Whether to relax on GPU. "
    "Relax on GPU can be much faster than CPU, so it is "
    "recommended to enable if possible. GPUs must be available"
    " if this setting is enabled.",
)

FLAGS = flags.FLAGS

RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if os.getenv("REFORMAT_PATH") is None:
        logging.warning(
            "REFORMAT_PATH variable is not set. FASTA files can't be used as MSA can be used if reformat.pl is not supplied"
        )

    run_multimer_system = "multimer" in FLAGS.model_preset

    if FLAGS.model_preset == "monomer_casp14":
        num_ensemble = 8
    else:
        num_ensemble = 1

    monomer_data_pipeline = pipeline.DataPipeline()

    if run_multimer_system:
        num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
        data_pipeline = pipeline_multimer.DataPipeline()
    else:
        num_predictions_per_model = 1
        data_pipeline = monomer_data_pipeline

    model_runners = {}
    model_names = config.MODEL_PRESETS[FLAGS.model_preset]
    for model_name in model_names:
        model_config = config.model_config(model_name)
        if run_multimer_system:
            model_config.model.num_ensemble_eval = num_ensemble
        else:
            model_config.data.eval.num_ensemble = num_ensemble
        model_params = data.get_model_haiku_params(
            model_name=model_name, data_dir=FLAGS.data_dir
        )
        model_runner = model.RunModel(model_config, model_params)
        for i in range(num_predictions_per_model):
            model_runners[f"{model_name}_pred_{i}"] = model_runner

    logging.info("Have %d models: %s", len(model_runners), list(model_runners.keys()))

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=FLAGS.use_gpu_relax,
    )

    random_seed = FLAGS.random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_runners))
    logging.info("Using random seed %d for the data pipeline", random_seed)

    basename = os.path.basename(
        FLAGS.precomputed_msa
    )  # Get the filename with extension
    fasta_name = os.path.splitext(basename)[0]  # Remove the extension

    # Predict structure for each of the sequences.
    predict_structure(
        precomputed_msa=FLAGS.precomputed_msa,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        models_to_relax=FLAGS.models_to_relax,
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(
        [
            "precomputed_msa",
            "output_dir",
            "data_dir",
            "use_gpu_relax",
        ]
    )

    app.run(main)
