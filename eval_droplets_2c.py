"""
Run instanseg inference for droplet segmentation

before running, run:
`aws s3 sync s3://sonata-resources/sonata_ml/training/instanseg/instanseg/torchscripts/ /home/ec2-user/instanseg_models/torchscripts/`

Steps to run:
1) copy the job id of a sonata job that you want re-run droplet inference on.
2) run `python eval_droplets_2c.py -j ####` and optionally see the flags for specifying the
    conductor env or non-default instanseg model path
3) The script will save a zarr array of droplets masks to the arranger assets prefix with the
    filename "99_instanseg_nuclei_predictions.zarr"
4) The sonata config file has the following section:
      properties:
        tn:
          droplet_masks_s3_path: null  <- replace null with the full path to the predictions
"""

import argparse
import gc
import os
import time

import numpy as np
import torch
from dask.array import core as dask_core
from dask.diagnostics.progress import ProgressBar
from tabs import CONDUCTOR_ENV
from tabs.img_utils import transformations
from tabs.io import file_io
from tabs.metadata import Assets, JobMetadata, TaxonomicMetadata
from tabs.pipeline import steps
from tabs.visualize import chip_contents

from instanseg import InstanSeg

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def log_mem(tag: str) -> None:
    rss_gb = None
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            rss_gb = proc.memory_info().rss / (1024**3)
        except Exception:
            rss_gb = None
    if rss_gb is None:
        try:
            import resource

            # On Linux, ru_maxrss is in KB.
            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        except Exception:
            rss_gb = None
    if torch.cuda.is_available():
        try:
            cuda_gb = torch.cuda.memory_allocated() / (1024**3)
        except Exception:
            cuda_gb = None
    else:
        cuda_gb = None
    rss_text = f'{rss_gb:.2f}GB' if rss_gb is not None else 'n/a'
    cuda_text = f'{cuda_gb:.2f}GB' if cuda_gb is not None else 'n/a'
    print(
        f'[{time.strftime("%H:%M:%S")}] {tag} RSS={rss_text} CUDA={cuda_text}',
        flush=True,
    )


def run(job_id: int, env: CONDUCTOR_ENV, model_path: str):
    """
    Run the instanseg model on the dyes/brightfield image for the given taxonomic metadata.
    """
    md = JobMetadata(
        job_id=job_id, env=env, taxonomic_metadata=TaxonomicMetadata.from_job_id(job_id, env)
    )
    mtg_sat_lims = md.load_asset(Assets.MTG_SAT_LIMS)
    chip = md.load_asset(Assets.REGISTERED_CHIP)

    fp = chip_contents.level_filepath(md.tmd, level=0)
    if file_io.check_if_file_exists(fp):
        full_mtg_darr = dask_core.from_zarr(fp)
        normalize = False
    else:
        full_mtg_darr = chip.render()
        normalize = True

    log_mem('start')
    with ProgressBar():
        full_mtg = np.asarray(full_mtg_darr)
    log_mem('after full_mtg')
    del full_mtg_darr

    summed_dyes = steps.sum_dyes(
        jmd=md,
        full_montage=full_mtg,
        montage_saturation_limits=mtg_sat_lims,
    )
    log_mem('after summed_dyes')
    bf_norm = (
        transformations.rescale_in_blocks(
            np.asarray(full_mtg[-1]), *mtg_sat_lims[chip.brightfield_channel]
        )
        if normalize
        else full_mtg[-1]
    )
    log_mem('after bf_norm')
    stacked = np.stack(
        [
            summed_dyes,
            bf_norm,
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    del full_mtg
    del summed_dyes
    del bf_norm
    log_mem('after stacked')
    print(f'{stacked.shape=}')
    torchscript_object = torch.jit.load(model_path)
    model = InstanSeg(torchscript_object, image_reader='skimage.io')
    del torchscript_object
    del chip
    del mtg_sat_lims
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_mem('after gc before eval_medium_image')
    log_mem('before eval_medium_image')
    instances = model.eval_medium_image(
        image=stacked,  # type: ignore
        tile_size=600,
        batch_size=16,
        target='nuclei',
        return_image_tensor=False,
        normalise=False,
    )
    log_mem('after eval_medium_image')
    instances = np.squeeze(np.asarray(instances))
    log_mem('after instances squeeze')
    save_path = md.tmd.build_arranger_asset_filepath(
        asset_suffix='99_instanseg_nuclei_predictions.zarr'
    )
    print(f'Saving droplet mask predictions to "{save_path}"')
    dask_core.to_zarr(
        dask_core.asarray(instances).rechunk((2048, 2048)),
        str(save_path),
        overwrite=True,
        storage_options={'mode': 'w'},
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', '-j', type=int, required=True)
    parser.add_argument('--env', '-e', type=str, required=False, default='prod')
    parser.add_argument(
        '--model_path',
        '-m',
        type=str,
        required=False,
        default=(
            '/home/ec2-user/instanseg_models/torchscripts/'
            '256px square premerge and post merge 2c 50 epochs.pt'
        ),
    )
    args = parser.parse_args()
    run(args.job_id, args.env, args.model_path)
