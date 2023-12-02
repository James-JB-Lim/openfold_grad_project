import logging
import math
import numpy as np
import os

from openfold.utils.script_utils import load_models_from_command_line, parse_fasta, run_model, prep_output, \
    update_timings, relax_protein

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

from openfold.data import templates, feature_pipeline, data_pipeline

import pickle
import random
import time
import torch

from openfold.config import model_config
from openfold.np import residue_constants, protein
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)

def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL

#with open("./T1124_pkl/features_of.pkl", "wb") as f:
#    pickle.dump(feature_dict, f)
with open("./T1124_pkl/features_of.pkl", "rb") as f:
    feature_dict = pickle.load(f)
tag = "T1124"
device = "cuda:0"
config_preset = "model_1"
trace_model = False
output_dir = os.getcwd()
multimer_ri_gap = 200
subtract_plddt = False
cif_output = False
output_name = "T1124_of"

config = model_config("model_1_ptm", long_sequence_inference="store_true")

model_generator = load_models_from_command_line(
    config,
    "cuda:0",
    "openfold/resources/openfold_params/finetuning_ptm_2.pt",
    None,
    "./")

feature_processor = feature_pipeline.FeaturePipeline(config.data)

for model, output_directory in model_generator:
    cur_tracing_interval = 0
    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode='predict',
    )

    processed_feature_dict = {
        k:torch.as_tensor(v, device=device)
        for k,v in processed_feature_dict.items()
    }

    if(trace_model):
        if(rounded_seqlen > cur_tracing_interval):
            logger.info(
                f"Tracing model at {rounded_seqlen} residues..."
            )
            t = time.perf_counter()
            trace_model_(model, processed_feature_dict)
            tracing_time = time.perf_counter() - t
            logger.info(
                f"Tracing time: {tracing_time}"
            )
            cur_tracing_interval = rounded_seqlen

    out = run_model(model, processed_feature_dict, tag, output_dir)

    # Toss out the recycling dimensions --- we don't need them anymore
    processed_feature_dict = tensor_tree_map(
        lambda x: np.array(x[..., -1].cpu()),
        processed_feature_dict
    )
    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    unrelaxed_protein = prep_output(
        out,
        processed_feature_dict,
        feature_dict,
        feature_processor,
        config_preset,
        multimer_ri_gap,
        subtract_plddt
    )

    unrelaxed_file_suffix = "_unrelaxed.pdb"
    if cif_output:
        unrelaxed_file_suffix = "_unrelaxed.cif"
    unrelaxed_output_path = os.path.join(
        output_directory, f'{output_name}{unrelaxed_file_suffix}'
    )

    with open(unrelaxed_output_path, 'w') as fp:
        if cif_output:
            fp.write(protein.to_modelcif(unrelaxed_protein))
        else:
            fp.write(protein.to_pdb(unrelaxed_protein))

    logger.info(f"Output written to {unrelaxed_output_path}...")
