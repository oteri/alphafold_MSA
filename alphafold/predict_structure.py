import enum
import json
import os
import pickle
import time
from typing import Any, Dict, Union

import jax.numpy as jnp
import numpy as np
from absl import logging

from alphafold.common import protein, residue_constants
from alphafold.data import pipeline, pipeline_multimer
from alphafold.model import model
from alphafold.relax import relax


@enum.unique
class ModelsToRelax(enum.Enum):
    ALL = 0
    BEST = 1
    NONE = 2


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


def predict_structure(
    precomputed_msa: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    models_to_relax: ModelsToRelax,
):
    """Predicts structure using AlphaFold for the given sequence."""
    logging.info("Predicting %s", fasta_name)
    timings = {}
    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get features.
    t_0 = time.time()
    feature_dict = data_pipeline.process(precomputed_msa=precomputed_msa)
    timings["features"] = time.time() - t_0

    # Write out features as a pickled dictionary.
    features_output_path = os.path.join(output_dir, "features.pkl")
    with open(features_output_path, "wb") as f:
        pickle.dump(feature_dict, f, protocol=4)

    unrelaxed_pdbs = {}
    unrelaxed_proteins = {}
    relaxed_pdbs = {}
    relax_metrics = {}
    ranking_confidences = {}

    # Run the models.
    num_models = len(model_runners)
    for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
        logging.info("Running model %s on %s", model_name, fasta_name)
        t_0 = time.time()
        model_random_seed = model_index + random_seed * num_models
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=model_random_seed
        )
        timings[f"process_features_{model_name}"] = time.time() - t_0

        t_0 = time.time()
        prediction_result, _ = model_runner.predict(
            processed_feature_dict, random_seed=model_random_seed
        )
        t_diff = time.time() - t_0
        timings[f"predict_and_compile_{model_name}"] = t_diff
        logging.info(
            "Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs",
            model_name,
            fasta_name,
            t_diff,
        )

        if benchmark:
            t_0 = time.time()
            prediction_result, _ = model_runner.predict(
                processed_feature_dict, random_seed=model_random_seed
            )
            t_diff = time.time() - t_0
            timings[f"predict_benchmark_{model_name}"] = t_diff
            logging.info(
                "Total JAX model %s on %s predict time (excludes compilation time): %.1fs",
                model_name,
                fasta_name,
                t_diff,
            )

        plddt = prediction_result["plddt"]
        ranking_confidences[model_name] = prediction_result[
            "ranking_confidence"
        ].tolist()  # It is converted to list so that it is serialzable

        # Remove jax dependency from results.
        np_prediction_result = _jnp_to_np(dict(prediction_result))

        # Save the model outputs.
        result_output_path = os.path.join(output_dir, f"result_{model_name}.pkl")
        with open(result_output_path, "wb") as f:
            pickle.dump(np_prediction_result, f, protocol=4)

        # Add the predicted LDDT in the b-factor column.
        # Note that higher predicted LDDT value means higher model confidence.
        plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1
        )
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not model_runner.multimer_mode,
        )

        unrelaxed_proteins[model_name] = unrelaxed_protein
        unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
        with open(unrelaxed_pdb_path, "w") as f:
            f.write(unrelaxed_pdbs[model_name])

    # Rank by model confidence.
    ranked_order = [
        model_name
        for model_name, confidence in sorted(
            ranking_confidences.items(), key=lambda x: x[1], reverse=True
        )
    ]

    # Relax predictions.
    if models_to_relax == ModelsToRelax.BEST:
        to_relax = [ranked_order[0]]
    elif models_to_relax == ModelsToRelax.ALL:
        to_relax = ranked_order
    elif models_to_relax == ModelsToRelax.NONE:
        to_relax = []

    for model_name in to_relax:
        t_0 = time.time()
        relaxed_pdb_str, _, violations = amber_relaxer.process(
            prot=unrelaxed_proteins[model_name]
        )
        relax_metrics[model_name] = {
            "remaining_violations": violations,
            "remaining_violations_count": sum(violations),
        }
        timings[f"relax_{model_name}"] = time.time() - t_0

        relaxed_pdbs[model_name] = relaxed_pdb_str

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(output_dir, f"relaxed_{model_name}.pdb")
        with open(relaxed_output_path, "w") as f:
            f.write(relaxed_pdb_str)

    # Write out relaxed PDBs in rank order.
    for idx, model_name in enumerate(ranked_order):
        ranked_output_path = os.path.join(output_dir, f"ranked_{idx}.pdb")
        with open(ranked_output_path, "w") as f:
            if model_name in relaxed_pdbs:
                f.write(relaxed_pdbs[model_name])
            else:
                f.write(unrelaxed_pdbs[model_name])

    ranking_output_path = os.path.join(output_dir, "ranking_debug.json")
    with open(ranking_output_path, "w") as f:
        label = "iptm+ptm" if "iptm" in prediction_result else "plddts"
        f.write(
            json.dumps({label: ranking_confidences, "order": ranked_order}, indent=4)
        )

    logging.info("Final timings for %s: %s", fasta_name, timings)

    timings_output_path = os.path.join(output_dir, "timings.json")
    with open(timings_output_path, "w") as f:
        f.write(json.dumps(timings, indent=4))
    if models_to_relax != ModelsToRelax.NONE:
        relax_metrics_path = os.path.join(output_dir, "relax_metrics.json")
        with open(relax_metrics_path, "w") as f:
            f.write(json.dumps(relax_metrics, indent=4))
