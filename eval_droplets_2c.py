"""
Run instanseg inference for droplet segmentation

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

import numpy as np
import torch
from dask.array import core as dask_core
from dask.diagnostics.progress import ProgressBar
from tabs import CONDUCTOR_ENV
from tabs.img_utils import transformations
from tabs.metadata import Assets, JobMetadata, TaxonomicMetadata
from tabs.pipeline import steps

from instanseg import InstanSeg


def run(
    job_id: int,
    env: CONDUCTOR_ENV,
    model_path: str,
):
    """
    Run the instanseg model on the dyes/brightfield image for the given taxonomic metadata.
    """
    md = JobMetadata(
        job_id=job_id,
        env=env,
        taxonomic_metadata=TaxonomicMetadata.from_job_id(job_id, env),
    )
    mtg_sat_lims = md.load_asset(Assets.MTG_SAT_LIMS)
    chip = md.load_asset(Assets.REGISTERED_CHIP)
    with ProgressBar():
        full_mtg = np.asarray(chip.render())

    summed_dyes = steps.sum_dyes(
        jmd=md,
        full_montage=full_mtg,
        montage_saturation_limits=mtg_sat_lims,
    )
    stacked = np.asarray(
        np.stack(
            [
                summed_dyes,
                transformations.rescale_in_blocks(
                    np.asarray(full_mtg[-1]),
                    *mtg_sat_lims[chip.brightfield_channel],
                ),
            ],
            axis=0,
        )
    )
    torchscript_object = torch.jit.load(model_path)
    model = InstanSeg(torchscript_object)
    instances = model.eval_medium_image(
        image=stacked,  # type: ignore
        tile_size=1024,
        batch_size=16,
        target='nuclei',
        return_image_tensor=False,
    )
    instances = np.squeeze(np.asarray(instances))
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
    parser.add_argument('--env', '-e', type=str, required=True, default='prod')
    parser.add_argument(
        '--model_path',
        '-m',
        type=str,
        required=False,
        default=(
            '/home/ec2-user/code/instanseg/instanseg/torchscripts/'
            '256px square premerge and post merge 2c 50 epochs.pt'
        ),
    )
    args = parser.parse_args()
    run(
        args.job_id,
        args.env,
        args.model_path,
    )
