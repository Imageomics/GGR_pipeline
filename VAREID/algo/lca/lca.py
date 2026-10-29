import argparse
import os
import yaml

from beta_stability import config as beta_stability_config
from beta_stability.init_logger import init_logger
from beta_stability.run import run_from_config

from VAREID.libraries.io.format_funcs import load_config, load_json, save_json
from VAREID.libraries.utils import path_from_file


def write_viewpoint_files(combined_path, output_dir, prefix, suffix, viewpoints=None):
    """
    Split the combined clustering output into one file per viewpoint,
    named {prefix}_{viewpoint}_{suffix}.json. Postprocessing reads the left and
    right files separately; beta_stability writes a single combined file.

    Args:
        combined_path: the combined annotation file written by beta_stability
        output_dir: directory to write the per-viewpoint files into
        prefix, suffix: output naming (same as the combined file)
        viewpoints: viewpoints to write; defaults to those present in the data.
            A viewpoint with no annotations still gets a file (empty annotation list)
            so downstream steps that expect both files can load them.
    """
    data = load_json(combined_path)
    annotations = data.get("annotations", [])
    present = sorted({a.get("viewpoint") for a in annotations if a.get("viewpoint") is not None})
    targets = list(viewpoints) if viewpoints else present
    written = []
    for viewpoint in targets:
        subset = dict(data)
        subset["annotations"] = [a for a in annotations if a.get("viewpoint") == viewpoint]
        out_path = os.path.join(output_dir, f"{prefix}_{viewpoint}_{suffix}.json")
        save_json(subset, out_path)
        written.append(out_path)
        print(f"Wrote {len(subset['annotations'])} {viewpoint} annotations to {out_path}")
    return written


if __name__ == "__main__":
    print("Loading data...")
    parser = argparse.ArgumentParser(description="Run LCA clustering")
    parser.add_argument("annots", type=str, help="The path to the annotation file.")
    parser.add_argument("embeddings", type=str, help="The path to the embeddings file.")
    #parser.add_argument("verifiers_probs", type=str, help="The path to the verifier probabilities.")
    parser.add_argument("lca_dir", type=str, help="The directory to save files into.")
    parser.add_argument("log_subunit_file", type=str, help="The path to the log file for the LCA algorithm itself.")
    parser.add_argument("log_file", type=str, help="The path to the log file.")
    parser.add_argument("--video", action="store_true", help="True if LCA should run on the video (drone) config file.")
    parser.add_argument("--intra", action="store_true", help="True if LCA should run on the intra config file.")
    parser.add_argument("--inter", action="store_true", help="True if LCA should run on the inter config file.")
    parser.add_argument("--separate_viewpoints", action="store_true", help="True if LCA should be run independently for left and right. (Legacy - use --separate_by_fields instead)")
    parser.add_argument("--separate_by_fields", nargs="+", help="List of fields to separate runs by, e.g., viewpoint encounter")
    parser.add_argument("--ui_db_path", type=str, default=None, help="Override data.ui_db_path in the LCA input config.")
    parser.add_argument("--max_human_reviews", type=int, default=None, help="Override stability.max_human_reviews in the LCA input config.")
    parser.add_argument("--lca_config", type=str, default=None, help="Path to a custom LCA input config (YAML). Takes precedence over --intra/--inter/--video/--image defaults.")
    parser.add_argument("--output_prefix", type=str, default=None, help="Prefix of the saved annotation file ({prefix}_{suffix}.json). Passed by lca_driver from lca_out_prefix.")
    parser.add_argument("--output_suffix", type=str, default=None, help="Suffix of the saved annotation file ({prefix}_{suffix}.json). Passed by lca_driver from lca_out_suffix.")

    args = parser.parse_args()


    # No longer wipe args.lca_dir — the old workflow needed a clean slate
    # for its `git clone` into `<lca_dir>/lca_code`. With the direct import
    # workflow there is no clone, and blowing away a shared results dir would
    # both destroy sibling outputs and fail on group-restricted parents.
    os.makedirs(args.lca_dir, exist_ok=True)

    # Config for this driver (not the input config to Beta Stability itself)
    lca_config = load_config(path_from_file(__file__, "lca_config.yaml"))

    # When True, run HDBSCAN via the same beta_stability entry point (config
    # override) instead of the Beta Stability algorithm.
    lca_alternative_clustering = lca_config["lca_alternative_clustering"]

    # OPEN INPUT CONFIG
    if args.lca_config:
        # User-supplied config takes precedence over the mode flags.
        input_config_name = os.path.basename(args.lca_config)
        input_config = load_config(args.lca_config)
        print(f"Using custom LCA config: {args.lca_config}")
    elif args.video:
        input_config_name = "lca_drone.yaml"
        input_config = load_config(path_from_file(__file__, input_config_name))
    elif args.intra:
        input_config_name = "lca_intra.yaml"
        input_config = load_config(path_from_file(__file__, input_config_name))
    elif args.inter:
        input_config_name = "lca_inter.yaml"
        input_config = load_config(path_from_file(__file__, input_config_name))
    else:
        input_config_name = "lca_image.yaml"
        input_config = load_config(path_from_file(__file__, input_config_name))

    # # ADD CONFIG INFO
    input_config["data"]["output_path"] = args.lca_dir
    input_config["data"]["annotation_file"] = args.annots
    input_config["data"]["embedding_file"] = args.embeddings

    # CLI overrides — only applied when explicitly given. Otherwise the
    # value already in the input config (YAML) is authoritative.
    if args.separate_by_fields:
        input_config["data"]["separate_by_fields"] = args.separate_by_fields
        if "separate_viewpoints" in input_config["data"]:
            del input_config["data"]["separate_viewpoints"]
    elif args.separate_viewpoints:
        input_config["data"]["separate_viewpoints"] = args.separate_viewpoints

    if args.ui_db_path is not None:
        input_config["data"]["ui_db_path"] = args.ui_db_path

    # Output file naming. beta_stability names the saved annotation file
    # {prefix}_{suffix}.json and falls back to beta_stability_annots.json when
    # neither is set; the Snakemake rules expect {lca_out_prefix}_{lca_out_suffix}.json.
    if args.output_prefix is not None or args.output_suffix is not None:
        output_cfg = input_config.setdefault("output", {}) or {}
        if args.output_prefix is not None:
            output_cfg["prefix"] = args.output_prefix
        if args.output_suffix is not None:
            output_cfg["suffix"] = args.output_suffix
        input_config["output"] = output_cfg
    if args.max_human_reviews is not None:
        input_config.setdefault("stability", {})["max_human_reviews"] = args.max_human_reviews

    # A config that asks for the ui_human reviewer needs a database to talk to.
    # Without one beta_stability silently answers reviews from the name_keys
    # field (a simulated reviewer), which is never what a pipeline run wants.
    verifier_names = str(input_config.get("edge_weights", {}).get("verifier_names", "")).split()
    if "ui_human" in verifier_names and not input_config["data"].get("ui_db_path"):
        raise SystemExit(
            f"{input_config_name} requests the ui_human reviewer but no ui_db_path is set. "
            "Set lca_ui_db_file in the pipeline config, pass --ui_db_path, or change "
            "edge_weights.verifier_names to no_human."
        )

    input_config["logging"]["log_file"] = args.log_subunit_file # should append LCA outputs into same log file used by this script
    # The pipeline expects the log at exactly log_subunit_file. update_log_file
    # would append a timestamp to it, and beta_stability re-applies that for every
    # field combination, so the name grows until it is too long for the filesystem.
    input_config["logging"]["update_log_file"] = False

    # HDBSCAN alternative: override the algorithm_type; everything else in the
    # config still applies. The unified beta_stability runner picks the right
    # algorithm off `algorithm_type`.
    if lca_alternative_clustering:
        input_config["algorithm_type"] = "hdbscan"
        print("Beta Stability driver: HDBSCAN alternative selected")
    else:
        print(
            f"Beta Stability driver: algorithm_type="
            f"{input_config.get('algorithm_type', 'stability')}"
        )

    # Merge the input config over beta_stability's declared defaults, the same
    # way `python -m beta_stability --config` does.
    run_config = beta_stability_config.load_mapping(input_config)

    # WRITE CONFIG SNAPSHOT alongside the results (for reproducibility)
    config_loc = os.path.join(args.lca_dir, input_config_name)
    with open(config_loc, "w") as f:
        yaml.dump(run_config, f)

    # RUN — beta_stability runs every field combination and writes the combined
    # annotation file ({prefix}_{suffix}.json) into lca_dir itself.
    print("Begin Beta Stability subunit...")
    init_logger()
    combined_path = run_from_config(run_config, save_dir=args.lca_dir)

    # Postprocessing consumes one file per viewpoint ({prefix}_left_{suffix}.json,
    # {prefix}_right_{suffix}.json). Derive them from the combined file for the
    # plain (non-encounter) LCA stage.
    if not (args.intra or args.inter):
        out_prefix = args.output_prefix or "beta_stability"
        out_suffix = args.output_suffix or "annots"
        write_viewpoint_files(combined_path, args.lca_dir, out_prefix, out_suffix,
                              viewpoints=run_config["data"].get("viewpoint_list"))
