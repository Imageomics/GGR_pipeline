import argparse
import sys

from VAREID.libraries.io.format_funcs import load_config
from VAREID.libraries.io.logging import log_subprocess, setup_logging
from VAREID.libraries.io.workflow_funcs import build_config, decode_config

def get_inputs(config, inter=False, intra=False):
    inputs = [config["mid_out_path"]]

    if intra:
        inputs.append(config["eg_out_path"])
    elif inter:
        inputs.append(config["representative_out_path"])
    elif config["data_video"]:
        inputs.append(config["fs_out_path"])
    else:
        inputs.append(config["ia_filtered_out_path"])

    return inputs


def get_outputs(config, inter=False, intra=False):
    # if config.get("lca_separate_by_fields"):
    #     # For multi-field separation, we don't know exact outputs without knowing field values
    #     # Return base path - actual outputs will be generated based on field combinations
    #     outputs = [config["lca_dir"]]  # Base directory contains all outputs
    # elif config.get("lca_separate_viewpoints"):
    #     # Legacy viewpoint separation
    #     outputs = [config["post_left_in_path"], config["post_right_in_path"]]
    # else:
    if intra:
        outputs = [config["encounter_lca_out_path"]]
    elif inter:
        outputs = [config["inter_lca_out_path"]]
    else:
        outputs = [config["lca_out_path"]]

    return outputs


def main(args):
    # SELECT THE CORRECT CONFIG
    if args.config:
        config = decode_config(args.config)
    else:
        config = build_config(load_config(args.config_path))

    intra = args.intra
    inter = args.inter

    if intra:
        input = config["eg_out_path"]
    elif inter:
        input = config["representative_out_path"]
    else:
        input = config["fs_out_path"] if config["data_video"] else config["ia_filtered_out_path"]
    video_flag = "--video" if config["data_video"] else ""
    
    # Handle field separation (new) vs viewpoint separation (legacy)
    if config.get("lca_separate_by_fields"):
        # New multi-field separation
        fields = config["lca_separate_by_fields"]
        if isinstance(fields, str):
            fields = fields.split()  # Convert string to list if needed
        separation_flag = f"--separate_by_fields {' '.join(fields)}"
    elif config.get("lca_separate_viewpoints"):
        # Legacy viewpoint separation
        separation_flag = "--separate_viewpoints"
    else:
        separation_flag = ""
    lca_dir = config["lca_dir"]
    lca_subunit_logs = config["lca_subunit_logs"]
    intra_flag = ""
    log_path = config["lca_logs"]
    ui_db_path = config.get("lca_ui_db_path")
    if intra:
        lca_dir = config["encounter_lca_dir"]
        intra_flag = " --intra"
        log_path = config["encounter_lca_logs"]
        lca_subunit_logs = config["encounter_lca_subunit_logs"]
        ui_db_path = config.get("encounter_lca_ui_db_path")
    elif inter:
        lca_dir = config["inter_lca_dir"]
        intra_flag = " --inter"
        log_path = config["inter_lca_logs"]
        lca_subunit_logs = config["inter_lca_subunit_logs"]
        ui_db_path = config.get("inter_lca_ui_db_path")

    # Human-review database for the ui_human reviewer (optional in config.yaml)
    ui_db_flag = f"--ui_db_path {ui_db_path}" if ui_db_path else ""
    
    # Output naming must match get_outputs(): {lca_out_prefix}_{lca_out_suffix}.json
    naming_flag = f'--output_prefix {config["lca_out_prefix"]} --output_suffix {config["lca_out_suffix"]}'

    command = f'python -u -m VAREID.algo.lca.lca {input} {config["mid_out_path"]} {lca_dir} {lca_subunit_logs} {log_path} {video_flag} {separation_flag} {intra_flag} {naming_flag} {ui_db_flag}'
    
    logger = setup_logging(log_path)
    returncode = log_subprocess(command, logger)
    # Propagate failure so Snakemake reports the error itself rather than a
    # missing output file.
    if returncode:
        sys.exit(returncode)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Driver script to run the LCA component of the pipeline. Clusters annotations."
    )
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--config",
        type=str,
        default=None,
        help="The built config file as a base64 encoded string. Config file MUST be structured like config.yaml!",
    )
    group.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="A path to the config file to load. Config file MUST be structured like config.yaml!",
    )
    parser.add_argument("--intra", action="store_true", help="True if LCA should run on the intra config file.")
    parser.add_argument("--inter", action="store_true", help="True if LCA should run on the inter config file.")
    
    args = parser.parse_args()

    main(args)
    