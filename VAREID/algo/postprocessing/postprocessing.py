import argparse
import os
import sqlite3
import sys
import time
import yaml
import pandas as pd
import json
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
from PIL import Image

from VAREID.libraries.io.format_funcs import load_config, save_json, split_dataframe, join_dataframe_dict
from VAREID.libraries.utils import path_from_file

# -----------------------------------------------------------------------------
# IPython and Database Connectors
# -----------------------------------------------------------------------------

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except ImportError:
    widgets = None
    print("ipywidgets not available; falling back to console input for interactive decisions.")

# Try to import database functions (NEW - for database mode)
try:
    from VAREID.libraries.ui.db_scripts import init_db, add_image_pair, get_decisions, check_pair_exists
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

except ImportError:
    widgets = None
    print("ipywidgets not available; falling back to console input for interactive decisions.")

# -----------------------------------------------------------------------------
# GENERAL HELPER FUNCTIONS (image display, decisions, time parsing …)
# -----------------------------------------------------------------------------

def get_user_decision(prompt="Merge clusters? (Yes/No): ", interactive_mode=True):
    """Gets a Yes/No decision from the user, via dropdown (widget) or stdin."""
    if interactive_mode and widgets is not None:
        dropdown = widgets.Dropdown(
            options=['Select', 'Yes', 'No'],
            value='Select',
            description=prompt,
            style={'description_width': 'initial'},
            layout={'width': 'max-content'}
        )
        display(dropdown)
        while dropdown.value == 'Select':
            plt.pause(0.1)
        decision = dropdown.value
        dropdown.close()
        clear_output(wait=True)
        return decision
    else:
        while True:
            decision_input = input(prompt).strip().lower()
            if decision_input.startswith('y'):
                return 'Yes'
            if decision_input.startswith('n'):
                return 'No'
            print('Invalid input. Please enter Yes or No.')

def get_image_path_from_uuid(images_list, image_uuid):
    """Looks up the relative image_path given an image_uuid in the images list."""
    for meta in images_list:
        if meta.get('uuid') == image_uuid:
            return meta.get('image_path')
    return None

# -----------------------------------------------------------------------------
# SAVE HELPERS (preserve full JSON structure)
# -----------------------------------------------------------------------------

def save_json_with_stage(data, original_filename, stage_suffix, final=False):
    """Saves JSON with an optional stage suffix (unless final=True)."""
    base, ext = os.path.splitext(original_filename)
    new_filename = f"{base}{ext}" if final else f"{base}_{stage_suffix}{ext}"
    df = pd.DataFrame(data["annotations"])
    final_data_to_save = split_dataframe(df)
    save_json(final_data_to_save, new_filename)
    print(f"Saved file: {new_filename}")
    return new_filename

# -----------------------------------------------------------------------------
# CLUSTER / ANNOTATION INTROSPECTION
# -----------------------------------------------------------------------------

def print_viewpoint_cluster_mapping(data, viewpoint, output_key='LCA_clustering_id'):
    """Human‑readable summary of Tracking‑IDs per cluster."""
    grouped = defaultdict(set)
    for ann in data.get('annotations', []):
        cid = ann.get(output_key)
        if cid is not None:
            grouped[cid].add(ann.get('tracking_id'))
    print(f"\n--- {viewpoint.capitalize()} Viewpoint Cluster ➜ Tracking‑ID Mapping ---")
    for cid in sorted(grouped.keys(), key=str):
        print(f"  Cluster {cid}: TIDs {sorted(list(grouped[cid]), key=str)}")

# -----------------------------------------------------------------------------
# TIME / ID HELPERS
# -----------------------------------------------------------------------------

def get_parent_id(tid):
    """Returns the base numeric part of tracking_id (before '_new')."""
    return str(tid).split('_new')[0]

def parse_timestamp(ts):
    """Parses many allowed timestamp formats to datetime."""
    if isinstance(ts, float) or isinstance(ts, int):
        return ts
    elif not isinstance(ts, str):
        raise ValueError('Timestamp must be a string')
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f'Unrecognised timestamp: {ts}')

def get_cluster_time_interval(anns):
    times = [parse_timestamp(a['timestamp']) for a in anns if a.get('timestamp')]
    return (min(times), max(times)) if times else (None, None)

def intervals_overlap(s1, e1, s2, e2):
    return all((s1, e1, s2, e2)) and (s1 <= e2 and s2 <= e1)

# -----------------------------------------------------------------------------
# GROUPING HELPERS
# -----------------------------------------------------------------------------

def group_annotations_by_lca(data, output_key='LCA_clustering_id'):
    grouped = defaultdict(list)
    for ann in data.get('annotations', []):
        cid = ann.get(output_key)
        if cid is not None:
            grouped[cid].append(ann)
    return grouped


def group_annotations_by_lca_with_viewpoint(data, viewpoint, output_key='LCA_clustering_id'):
    grouped = defaultdict(list)
    for ann in data.get('annotations', []):
        cid = ann.get(output_key)
        if cid is not None:
            grouped[f"{cid}_{viewpoint}"].append(ann)
    return grouped

# -----------------------------------------------------------------------------
# CLUSTER MERGE / SPLIT HELPERS
# -----------------------------------------------------------------------------

def get_cluster_best_ann_for_display(anns):
    return max(anns, key=lambda x: x.get('CA_score', 0.0)) if anns else None

def _update_cluster_merge_deterministic(grouped, source_id, target_id, output_key='LCA_clustering_id'):
    """Merge all anns from source → target (keeps target's cid)."""
    if source_id == target_id:
        return
    if source_id not in grouped or target_id not in grouped:
        return
    base_cid = grouped[target_id][0][output_key]
    for ann in grouped[source_id]:
        ann[output_key] = base_cid
    grouped[target_id].extend(grouped[source_id])
    del grouped[source_id]


def _update_split_no_merge_deterministic(grouped, anchor_id, other_id):
    """User said 'No' to merging two clusters with shared TIDs → rename duplicates."""
    if anchor_id not in grouped or other_id not in grouped:
        return False
    anchor_tids = {a['tracking_id'] for a in grouped[anchor_id]}
    renamed = 0
    for ann in grouped[other_id]:
        if ann['tracking_id'] in anchor_tids:
            ann['tracking_id'] = f"{ann['tracking_id']}_new"
            renamed += 1
    if renamed:
        print(f"      No merge: renamed {renamed} conflicting TIDs in {other_id}.")
    return bool(renamed)

# -----------------------------------------------------------------------------
# DATABASE OPERATIONS
# -----------------------------------------------------------------------------

def submit_pair_to_database(best_ann1, best_ann2, image_dir, db_path, output_key='LCA_clustering_id'):
    """Submit a pair to database, returns pair_id and existing decision if any"""
    # Order UUIDs to create consistent ID
    uuid1 = best_ann1['uuid']
    uuid2 = best_ann2['uuid']
    if uuid1 > uuid2:
        uuid1, uuid2 = uuid2, uuid1
        best_ann1, best_ann2 = best_ann2, best_ann1

    unique_id = f"{uuid1}_{uuid2}"

    # Check if this pair already exists
    exists, status, decision = check_pair_exists(uuid1, uuid2, db_path)

    if exists:
        if status in ['checked', 'sent'] and decision != 'none':
            return {'pair_id': unique_id, 'decision': decision}
        else:
            return {'pair_id': unique_id, 'decision': None}

    # Get image paths from annotation's image_path field
    image_path1 = best_ann1.get('image_path')
    image_path2 = best_ann2.get('image_path')

    # Extract cluster information
    cluster1 = str(best_ann1.get(output_key, 'UNKNOWN'))
    cluster2 = str(best_ann2.get(output_key, 'UNKNOWN'))

    # Extract bounding boxes
    bbox1 = None
    bbox2 = None
    if "bbox" in best_ann1 and best_ann1["bbox"]:
        x, y, w, h = best_ann1["bbox"]
        bbox1 = json.dumps([x, y, x + w, y + h])
    if "bbox" in best_ann2 and best_ann2["bbox"]:
        x, y, w, h = best_ann2["bbox"]
        bbox2 = json.dumps([x, y, x + w, y + h])

    # Add to database
    # NOTE: pass db_path as a keyword — add_image_pair's signature has an
    # optional `score` parameter before db_path, so a positional db_path here
    # lands in `score` and triggers float() conversion of the path.
    add_image_pair(
        unique_id,
        uuid1,
        image_path1 or "NO_IMAGE",
        bbox1,
        cluster1,
        uuid2,
        image_path2 or "NO_IMAGE",
        bbox2,
        cluster2,
        db_path=db_path
    )
    
    return {'pair_id': unique_id, 'decision': None}

def wait_for_single_decision(db_path, pair_id, check_interval=5):
    """Wait for a specific pair to be completed"""
    print(f"Waiting for verification decision for pair {pair_id}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT cluster1, cluster2 FROM image_verification
        WHERE id = ? AND status IN ('awaiting', 'in_progress')
    """, (pair_id,))
    cluster1, cluster2 = cursor.fetchone()
    conn.close()

    print(f"Cluster 1: {cluster1} Cluster 2: {cluster2}")
    print("Please complete verification task in the UI...")
    
    while True:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT status FROM image_verification
            WHERE id = ? AND status IN ('awaiting', 'in_progress')
        """, (pair_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            break
        
        print(f"Still waiting for cluster pair {cluster1} - {cluster2} - Checking again in {check_interval} seconds...")
        time.sleep(check_interval)
    
    print(f"Verification completed for cluster pair {cluster1} - {cluster2}!")
    
# -----------------------------------------------------------------------------
# INTERACTIVE DISPLAY & DECISION
# -----------------------------------------------------------------------------

def pairwise_verification_interactive(grouped, c1_id, c2_id, data_context, image_dir, interactive_mode, db_path, context_message="", output_key='LCA_clustering_id'):
    """Show side‑by‑side crops for two representative anns and ask the user."""
    if (c1_id not in grouped or c2_id not in grouped or not grouped[c1_id] or not grouped[c2_id]):
        return 'No'

    print(f"\n--- User Verification: '{c1_id}'  vs  '{c2_id}' ---")
    if context_message:
        print(f"  Context: {context_message}")

    ann1 = get_cluster_best_ann_for_display(grouped[c1_id])
    # Handle optional database mode
    if interactive_mode == 'database':
        if not DATABASE_AVAILABLE:
            raise ImportError('Database mode requires UI.db_scripts available')
        res = submit_pair_to_database(ann1, get_cluster_best_ann_for_display(grouped[c2_id]), image_dir, db_path, output_key)
        if res['decision'] is not None:
            return 'Yes' if res['decision']=='correct' else 'No'
        wait_for_single_decision(db_path, res['pair_id'])
        db_decision = get_single_decision(db_path, res['pair_id'])
        return 'Yes' if db_decision=='correct' else 'No'
    
    ann2 = get_cluster_best_ann_for_display(grouped[c2_id])
    if not ann1 or not ann2:
        return 'No'

    if image_dir and os.path.isdir(image_dir):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for i, ann in enumerate([ann1, ann2]):
            ax = axes[i]
            uuid = ann.get('image_uuid')
            fpath = get_image_path_from_uuid(data_context['images'], uuid)
            if fpath and not os.path.isabs(fpath):
                fpath = os.path.join(image_dir, fpath)
            if fpath and os.path.exists(fpath):
                try:
                    img = Image.open(fpath)
                    x, y, w, h = map(int, ann['bbox'])
                    ax.imshow(img.crop((x, y, x + w, y + h)))
                    ax.set_title(f"Cls:{ann[output_key]}\nTID:{ann['tracking_id']}\nUUID:{ann.get('uuid', 'NA')}")
                    ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center'); ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Image N/A', ha='center', va='center'); ax.axis('off')
        plt.tight_layout()
        if widgets and interactive_mode:
            display(fig)
            plt.close(fig)
        else:
            plt.show(block=False)
            plt.pause(0.1)

    decision = get_user_decision(f"    Merge '{c1_id}' & '{c2_id}'? (Yes/No): ", interactive_mode)
    print(f"    User chose: {decision}")
    return decision

def get_single_decision(db_path, pair_id):
    """Get decision for a single pair"""
    decisions = get_decisions([pair_id], db_path)
    return decisions.get(pair_id)

# -----------------------------------------------------------------------------
# STAGE 1 – TID‑Split Verification (per‑view)
# -----------------------------------------------------------------------------

def tid_split_verification(grouped, data_view, viewpoint, image_dir, interactive_mode, db_path, output_key='LCA_clustering_id'):
    print(f"\n--- Verifying TID‑splits for {viewpoint} Viewpoint ---")
    while True:
        tid_to_clusters = defaultdict(set)
        for cid, anns in grouped.items():
            for ann in anns:
                tid_to_clusters[ann['tracking_id']].add(cid)
        conflicts = {tid: cset for tid, cset in tid_to_clusters.items() if len(cset) > 1}
        if not conflicts:
            print(f"  No conflicts in {viewpoint}. Viewpoint is stable.")
            break

        tid, cset = sorted(conflicts.items(), key=lambda kv: str(kv[0]))[0]
        print(f"  Conflict: TID '{tid}' appears in clusters {cset}")
        c1, c2 = sorted(list(cset), key=str)[:2]
        decision = pairwise_verification_interactive(grouped, c1, c2, data_view, image_dir, interactive_mode, db_path, output_key=output_key)
        anchor, other = sorted([c1, c2], key=str)
        if decision == 'Yes':
            _update_cluster_merge_deterministic(grouped, other, anchor, output_key)
        else:
            _update_split_no_merge_deterministic(grouped, anchor, other)

        # Refresh underlying annotations list for this view and re‑evaluate.
        data_view['annotations'] = [ann for sub in grouped.values() for ann in sub]

# -----------------------------------------------------------------------------
# STAGE 2 – Cross‑view Reconciliation (with memory of declined pairs)
# -----------------------------------------------------------------------------

def check_numeric_equivalence(grouped_all):
    """Build adjacency list linking clusters via shared *numeric* TIDs across views."""
    adj = {k: set() for k in grouped_all}
    tid_to_clusters = defaultdict(set)
    for cluster_key, anns in grouped_all.items():
        for ann in anns:
            if str(ann['tracking_id']).isdigit():
                tid_to_clusters[ann['tracking_id']].add(cluster_key)
    for clusters in tid_to_clusters.values():
        for c1, c2 in itertools.combinations(clusters, 2):
            if c1.endswith('_left') != c2.endswith('_left'):
                adj[c1].add(c2)
                adj[c2].add(c1)
    return adj


def find_conflicts(adj):
    return {node: nbrs for node, nbrs in adj.items() if len(nbrs) > 1}


def resolve_cross_view_conflicts_interactive(conflicts, grouped_all, data_map, image_dir, interactive_mode, declined_pairs, db_path, output_key='LCA_clustering_id'):
    """Attempt interactive merges; remembers user‑declined pairs via `declined_pairs`."""
    print("  Interactive conflict resolution …")
    for parent_key, targets in conflicts.items():
        for t1, t2 in itertools.combinations(sorted(list(targets), key=str), 2):
            pair = tuple(sorted((t1, t2)))
            if pair in declined_pairs:
                continue  # user already said no
            if t1 not in grouped_all or t2 not in grouped_all:
                continue

            # Heuristic context message
            s1, e1 = get_cluster_time_interval(grouped_all[t1])
            s2, e2 = get_cluster_time_interval(grouped_all[t2])
            ctx = "SIMULTANEOUS" if intervals_overlap(s1, e1, s2, e2) else "SEQUENTIAL"
            view = t1.split('_')[-1]
            decision = pairwise_verification_interactive(
                grouped_all, t1, t2, data_map[view], image_dir, interactive_mode, db_path, context_message=f"Clusters appear {ctx}", output_key=output_key)

            if decision == 'Yes':
                anchor, other = sorted([t1, t2])
                _update_cluster_merge_deterministic(grouped_all, other, anchor, output_key)
                return True, anchor  # merge happened; tell caller which cluster mutated
            else:
                declined_pairs.add(pair)
    print("  No merges accepted in this pass.")
    return False, None

# -----------------------------------------------------------------------------
# SPLIT HELPERS (unchanged from v2)
# -----------------------------------------------------------------------------

def generate_new_lca_id(base, existing):
    i = 1
    while True:
        new_id = f"{base}_split{i}"
        if new_id not in existing:
            return new_id
        i += 1

def split_conflicting_cluster(parent_key, targets, grouped_all, all_lca_ids_view, output_key='LCA_clustering_id'):
    """Split parent cluster so each numeric‑TID family gets its own cluster."""
    print(f"    Splitting {parent_key} (links to {len(targets)} clusters)")
    parent_anns = list(grouped_all[parent_key])
    base_id, view = parent_key.split('_')[0], parent_key.split('_')[1]
    moved_uuids = set()
    new_parts = defaultdict(list)
    sibling_keys = []

    for tkey in sorted(targets, key=str):
        # A conflict snapshot can name a target that an earlier split in this same
        # pass already emptied and removed (mutual X_left <-> X_right conflicts).
        # Its annotations have moved into that target's split-siblings, so there is
        # nothing to split against here; skip it and let the next reconciliation
        # cycle re-detect the conflict on the updated clustering.
        num_tids = {a['tracking_id'] for a in grouped_all.get(tkey, []) if str(a['tracking_id']).isdigit()}
        anns_for_split = [a for a in parent_anns if a['tracking_id'] in num_tids and a['uuid'] not in moved_uuids]
        if not anns_for_split:
            continue
        new_id = generate_new_lca_id(base_id, all_lca_ids_view)
        all_lca_ids_view.add(new_id)
        new_key = f"{new_id}_{view}"
        sibling_keys.append(new_key)
        for a in anns_for_split:
            a[output_key] = new_id
            new_parts[new_key].append(a)
            moved_uuids.add(a['uuid'])
        print(f"      → created {new_key} to link with {tkey}")

    remaining = [a for a in parent_anns if a['uuid'] not in moved_uuids]
    if remaining:
        grouped_all[parent_key] = remaining
        sibling_keys.append(parent_key)
        print(f"      → {len(remaining)} annotations remain in {parent_key}")
    else:
        del grouped_all[parent_key]
        print(f"      → {parent_key} emptied and removed")

    for k, lst in new_parts.items():
        grouped_all[k] = lst
    return sibling_keys if new_parts else []

# Extra verification helper to optionally merge remnants back (from v2)

def verify_remnant_against_splits(sibling_keys, grouped_all, data_map, viewpoint, image_dir, interactive_mode, db_path, output_key='LCA_clustering_id'):
    remnant = None
    split_keys = []
    for k in sibling_keys:
        if '_split' in k:
            split_keys.append(k)
        else:
            remnant = k
    if not remnant or not split_keys:
        return False
    print(f"    Verifying remnant {remnant} against its split‑offs …")
    for sk in split_keys:
        if remnant not in grouped_all or sk not in grouped_all:
            continue
        decision = pairwise_verification_interactive(
            grouped_all, remnant, sk, data_map[viewpoint], image_dir, interactive_mode, db_path, context_message='Does remnant belong with this split‑off part?', output_key=output_key)
        if decision == 'Yes':
            _update_cluster_merge_deterministic(grouped_all, remnant, sk, output_key)
            return True
    return False

# -----------------------------------------------------------------------------
# FINAL ID ASSIGNMENT (unchanged)
# -----------------------------------------------------------------------------

def assign_final_ids(grouped_all, data_left, data_right):
    print("\n--- Assigning Final IDs ---")
    adj = check_numeric_equivalence(grouped_all)
    visited = set()
    components = []
    for node in sorted(grouped_all.keys(), key=str):
        if node in visited:
            continue
        comp = {node}
        queue = [node]
        visited.add(node)
        head = 0
        while head < len(queue):
            cur = queue[head]; head += 1
            for nbr in adj.get(cur, []):
                if nbr not in visited:
                    visited.add(nbr); comp.add(nbr); queue.append(nbr)
        components.append(comp)

    id_map = {}
    print(f"  Found {len(components)} equivalence sets.")
    for idx, comp in enumerate(components, start=1):
        fid = str(idx)
        print(f"  Individual {fid}: {comp}")
        for key in comp:
            id_map[key] = fid

    for ann in data_left['annotations']:
        ann['final_id'] = id_map.get(f"{ann['LCA_clustering_id']}_left", 'UNASSIGNED')
    for ann in data_right['annotations']:
        ann['final_id'] = id_map.get(f"{ann['LCA_clustering_id']}_right", 'UNASSIGNED')
    print("  Done.")

# -----------------------------------------------------------------------------
# MAIN WORKFLOW
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Post‑process LCA outputs with interactive verification.')
    parser.add_argument('images', type=str, help='Directory of source images')
    parser.add_argument('in_left', type=str, help='Left‑view annotations JSON')
    parser.add_argument('in_right', type=str, help='Right‑view annotations JSON')
    parser.add_argument('out_left', type=str, help='Output JSON for left view')
    parser.add_argument('out_right', type=str, help='Output JSON for right view')
    
    parser.add_argument('--db', type=str, help='Path to verification database (optional)')
    parser.add_argument('--interaction_mode', type=str, choices=['console','ipywidgets','database'],
                        help='Interaction mode (overrides config)')
    args = parser.parse_args()

    # Load config
    cfg = load_config(path_from_file(__file__, "postprocessing_config.yaml"))

    # Get output_key from config or use default
    output_key = cfg.get('output_key', 'LCA_clustering_id') if cfg else 'LCA_clustering_id'

    # Determine interaction mode
    if args.interaction_mode:
        interaction_mode = args.interaction_mode
    elif cfg and ('interaction_mode' in cfg or 'interactive' in cfg):
        if 'interaction_mode' in cfg:
            interaction_mode = cfg['interaction_mode']
        else:
            interaction_mode = cfg.get('interactive', True)
            if interaction_mode == 'database':
                interaction_mode = 'database'
            elif interaction_mode:
                interaction_mode = 'ipywidgets'
            else:
                interaction_mode = 'console'
    else:
        interaction_mode = 'console'

    # Normalize
    if interaction_mode == 'ipywidgets':
        interaction_mode = True
    elif interaction_mode == 'console':
        interaction_mode = False
    # 'database' remains a string

    # Database setup
    db_path = None
    if interaction_mode == 'database':
        if not DATABASE_AVAILABLE:
            raise ImportError("Database mode requires UI.db_scripts module.")
        db_path = args.db or (cfg.get('database',{}).get('path') if cfg else None)
        if db_path is None:
            raise ValueError("Database mode requires --db or database.path in config.")
        init_db(db_path)
        print(f"Using database mode with database: {db_path}")
    elif interaction_mode is True and widgets is None:
        print("Warning: ipywidgets not available, falling back to console mode")
        interaction_mode = False

    image_dir = args.images if (args.images and os.path.isdir(args.images)) else (cfg.get('image',{}).get('directory') if cfg else None)
    if not image_dir:
        print(f"Warning: Image directory '{args.images}' not found or not specified. Image display will be skipped.")

    data_left = join_dataframe_dict(json.load(open(args.in_left)))
    data_right = join_dataframe_dict(json.load(open(args.in_right)))
    data_map = {'left': data_left, 'right': data_right}

    # ---------------- Initial state ----------------
    print('=' * 60, '\nINITIAL DATA STATE', '\n' + '=' * 60)
    print_viewpoint_cluster_mapping(data_left, 'left')
    print_viewpoint_cluster_mapping(data_right, 'right')

    # ---------------- Stage 1 ----------------
    print('\n' + '='*60 + '\nSTAGE 1: TID‑Split Verification' + '\n' + '='*60)
    grouped_left = group_annotations_by_lca(data_left)
    grouped_right = group_annotations_by_lca(data_right)
    tid_split_verification(grouped_left, data_left, 'Left', image_dir, interaction_mode, db_path)
    tid_split_verification(grouped_right, data_right, 'Right', image_dir, interaction_mode, db_path)
    save_json_with_stage(data_left, args.out_left, 'split_verified')
    save_json_with_stage(data_right, args.out_right, 'split_verified')

    print('\n' + '='*60 + '\nCLUSTER STATE AFTER STAGE 1' + '\n' + '='*60)
    print_viewpoint_cluster_mapping(data_left, 'left')
    print_viewpoint_cluster_mapping(data_right, 'right')

    # ---------------- Stage 2 ----------------
    print('\n' + '='*60 + '\nSTAGE 2: Cross‑view Reconciliation' + '\n' + '='*60)
    declined_pairs = set()
    MAX_LOOPS = 10
    for loop in range(1, MAX_LOOPS+1):
        print(f"\n--- Reconciliation Cycle {loop}/{MAX_LOOPS} ---")
        grouped_all = {
            **group_annotations_by_lca_with_viewpoint(data_left, 'left'),
            **group_annotations_by_lca_with_viewpoint(data_right, 'right')
        }
        adj = check_numeric_equivalence(grouped_all)
        conflicts = find_conflicts(adj)
        if not conflicts:
            print('  No conflicts. System is stable.')
            break

        print(f"  {len(conflicts)} clusters exhibit one‑to‑many conflicts.")
        for key, tset in conflicts.items():
            print(f"    {key} → {tset}")

        merge_made, merged_anchor = resolve_cross_view_conflicts_interactive(
            conflicts, grouped_all, data_map, image_dir, interaction_mode, declined_pairs, db_path)
        if merge_made:
            declined_pairs = {p for p in declined_pairs if merged_anchor not in p}
            print('  Merge executed. Restarting cycle …')
            # Push grouped_all back to left/right datasets
            data_left['annotations'] = [a for k, lst in grouped_all.items() if k.endswith('_left') for a in lst]
            data_right['annotations'] = [a for k, lst in grouped_all.items() if k.endswith('_right') for a in lst]

            # NEW: detailed cluster state after merge
            print('\n--- Cluster State After Merge ---')
            print_viewpoint_cluster_mapping(data_left, 'left')
            print_viewpoint_cluster_mapping(data_right, 'right')
            continue

        # No merges → automatic splits
        print('  No merges accepted. Proceeding with splits …')
        made_change = False
        sibling_groups = []
        lca_ids_left = {k.split('_')[0] for k in grouped_all if k.endswith('_left')}
        lca_ids_right = {k.split('_')[0] for k in grouped_all if k.endswith('_right')}

        conflicts = find_conflicts(check_numeric_equivalence(grouped_all))
        for conf_key, targets in conflicts.items():
            siblings = split_conflicting_cluster(
                conf_key, targets, grouped_all,
                lca_ids_left if conf_key.endswith('_left') else lca_ids_right)
            if siblings:
                sibling_groups.append({'keys': siblings, 'view': conf_key.split('_')[1]})
                made_change = True

        # Optional remnant verification
        for grp in sibling_groups:
            if verify_remnant_against_splits(grp['keys'], grouped_all, data_map, grp['view'], image_dir, interaction_mode, db_path):
                made_change = True

        if made_change:
            data_left['annotations'] = [a for k, lst in grouped_all.items() if k.endswith('_left') for a in lst]
            data_right['annotations'] = [a for k, lst in grouped_all.items() if k.endswith('_right') for a in lst]
            print('  Changes (split/merge) occurred. Restarting cycle …')
            # NEW: detailed cluster state after automatic changes
            print('\n--- Cluster State After Changes ---')
            print_viewpoint_cluster_mapping(data_left, 'left')
            print_viewpoint_cluster_mapping(data_right, 'right')
            continue
        else:
            print('  System stable after splits. No further action required.')
            break
    else:
        print('\n--- Maximum reconciliation cycles reached. Proceeding with current state. ---')

    # ---------------- Final IDs ----------------
    grouped_all = {
        **group_annotations_by_lca_with_viewpoint(data_left, 'left'),
        **group_annotations_by_lca_with_viewpoint(data_right, 'right')
    }
    assign_final_ids(grouped_all, data_left, data_right)

    # ---------------- Save final ----------------
    print('\n' + '='*60 + '\nFINAL RESULTS' + '\n' + '='*60)
    save_json_with_stage(data_left, args.out_left, 'final', final=True)
    save_json_with_stage(data_right, args.out_right, 'final', final=True)

    print_viewpoint_cluster_mapping(data_left, 'left')
    print_viewpoint_cluster_mapping(data_right, 'right')

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f"\nAn error occurred: {exc}")
        if 'the following arguments are required' in str(exc):
            print("\nThis script must be run with CLI arguments:")
            print("   python lca_postprocess_verbose.py <images_dir> <in_left.json> <in_right.json> <out_left.json> <out_right.json>")
