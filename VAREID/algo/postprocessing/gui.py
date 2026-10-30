"""
Pair verification GUI.

Reviewers see two annotations side by side and answer Same / Different / Can't tell.
Pairs come from an SQLite database (see VAREID.libraries.ui.db_scripts) that the
postprocessing step, or the LCA ui_human reviewer, fills and reads back.

Presentation
  * Each crop is padded to a common aspect ratio and scaled to the same display
    height, so a distant animal is shown at the same size as a close one.
  * Scroll to zoom, drag to pan, double-click to reset. A toggle shows the full
    image with the box drawn, for context.

Decisions
  * A decision is held for UNDO_GRACE_SECONDS before it is written to the
    database. Undo (or Z) within that time takes it back without any record.
  * Back reopens the previous pair whatever its state: an answered pair has its
    decision cleared, and a pair the clustering step already consumed is
    reopened so the new answer is sent again on the next poll.
"""

import argparse
import atexit
import base64
import io
import json
import math
import os
import threading
import time

import gradio as gr
from PIL import Image, ImageDraw

from VAREID.libraries.ui.db_scripts import (
    update_status, get_next_pair_atomic, release_pair, reopen_pair,
    start_heartbeat_system, init_db, get_instance_stats,
    update_heartbeat, INSTANCE_IDENTIFIER
)

os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/gradio_cache")

# ---------------------------------------------------------------------------
# Presentation settings
# ---------------------------------------------------------------------------
DISPLAY_HEIGHT = 520          # px; both panels are rendered at this height
CROP_ASPECT = 4 / 3           # width / height that every crop is padded to
PAD_COLOR = (32, 32, 32)
FULL_MAX_WIDTH = 1600         # full images are downscaled to this width before embedding
BOX_COLOR = (255, 214, 0)
UNDO_GRACE_SECONDS = 5
JPEG_QUALITY = 88
RENDER_CACHE_LIMIT = 64

LABELS = {"correct": "Same", "incorrect": "Different", "cant_tell": "Can't tell", "none": "none", None: "none"}

# ---------------------------------------------------------------------------
# State (one reviewer per process)
# ---------------------------------------------------------------------------
db_path = None
review_limit = None     # shown as "Reviewed: n / limit" when set
current_pair = None     # pair dict being shown, reserved (in_progress) by this instance
pending = None          # {"pair": dict, "label": str, "deadline": float} awaiting commit
history = []            # pairs shown before the current one, most recent last
show_full = False
queued_view = None      # outputs tuple timer_tick has queued for the panels
view_version = 0        # bumped when queued_view changes; drives the redraw
heartbeat_timer = None
_render_cache = {}


def configure(path, limit=None):
    """Set the database this process works on (used by main and by tests), and
    the review limit shown next to the reviewed count (None shows no limit)."""
    global db_path, review_limit
    db_path = path
    review_limit = limit


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _pad_to_aspect(img, aspect):
    w, h = img.size
    target_w = max(w, int(round(h * aspect)))
    target_h = max(h, int(round(target_w / aspect)))
    canvas = Image.new("RGB", (target_w, target_h), PAD_COLOR)
    canvas.paste(img, ((target_w - w) // 2, (target_h - h) // 2))
    return canvas


def _scale_to_display(img):
    """Resize a CROP_ASPECT-padded image to the exact display size. Padding leaves
    the aspect within a pixel of CROP_ASPECT, so the residual distortion is below
    a third of a percent, and every panel gets identical pixel dimensions."""
    return img.resize((int(round(DISPLAY_HEIGHT * CROP_ASPECT)), DISPLAY_HEIGHT), Image.LANCZOS)


def _clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def prepare_view(image_path, bbox_json, full_view):
    """Return a PIL image of the crop (or the full frame with the box drawn),
    padded to CROP_ASPECT and scaled to DISPLAY_HEIGHT. None if unavailable."""
    if not image_path or image_path == "NO_IMAGE" or not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    bbox = None
    if bbox_json:
        try:
            bbox = _clamp_bbox(json.loads(bbox_json), img.width, img.height)
        except Exception as e:
            print(f"Error parsing bbox {bbox_json}: {e}")

    if full_view:
        if bbox:
            ImageDraw.Draw(img).rectangle(bbox, outline=BOX_COLOR, width=max(3, img.width // 400))
        if img.width > FULL_MAX_WIDTH:
            img = img.resize((FULL_MAX_WIDTH, max(1, int(round(img.height * FULL_MAX_WIDTH / img.width)))), Image.LANCZOS)
    elif bbox:
        img = img.crop(bbox)

    return _scale_to_display(_pad_to_aspect(img, CROP_ASPECT))


def _placeholder_html(text):
    return (f'<div class="vz vz-empty" style="height:{DISPLAY_HEIGHT}px">'
            f'<span>{text}</span></div>')


def render_side(image_path, bbox_json, full_view):
    """HTML for one panel: a zoomable container with the image embedded as a data URI."""
    key = (image_path, bbox_json, bool(full_view))
    if key in _render_cache:
        return _render_cache[key]

    img = prepare_view(image_path, bbox_json, full_view)
    if img is None:
        html = _placeholder_html("No image available")
    else:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        html = (f'<div class="vz" style="height:{DISPLAY_HEIGHT}px">'
                f'<img src="{uri}" draggable="false" alt="">'
                f'<span class="vz-hint">scroll: zoom · drag: pan · double-click: reset</span></div>')

    if len(_render_cache) >= RENDER_CACHE_LIMIT:
        _render_cache.clear()
    _render_cache[key] = html
    return html


HEAD = """
<style>
.vz{position:relative;overflow:hidden;width:100%;background:#202020;border-radius:8px;
    cursor:grab;user-select:none;touch-action:none}
.vz img{position:absolute;left:0;top:0;width:100%;height:100%;object-fit:contain;
    transform-origin:0 0;will-change:transform}
.vz .vz-hint{position:absolute;right:8px;bottom:6px;font:12px sans-serif;color:#ddd;
    opacity:.6;pointer-events:none}
.vz-empty{display:flex;align-items:center;justify-content:center;color:#999;
    font:16px sans-serif;cursor:default}
</style>
<script>
(function(){
  const state = new WeakMap();
  function get(el){ let s = state.get(el); if(!s){ s = {k:1, x:0, y:0, drag:null}; state.set(el, s); } return s; }
  function apply(el, s){ const img = el.querySelector('img'); if(img){ img.style.transform = `translate(${s.x}px,${s.y}px) scale(${s.k})`; } }
  document.addEventListener('wheel', function(e){
    const el = e.target.closest && e.target.closest('.vz'); if(!el || !el.querySelector('img')) return;
    e.preventDefault();
    const s = get(el), r = el.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const f = e.deltaY < 0 ? 1.15 : 1/1.15;
    const k = Math.min(12, Math.max(1, s.k * f)), ratio = k / s.k;
    s.x = px - (px - s.x) * ratio; s.y = py - (py - s.y) * ratio; s.k = k;
    if(k === 1){ s.x = 0; s.y = 0; }
    apply(el, s);
  }, {passive:false});
  document.addEventListener('mousedown', function(e){
    const el = e.target.closest && e.target.closest('.vz'); if(!el || e.button !== 0) return;
    const s = get(el); s.drag = {x: e.clientX - s.x, y: e.clientY - s.y};
    el.style.cursor = 'grabbing'; e.preventDefault();
  });
  document.addEventListener('mousemove', function(e){
    document.querySelectorAll('.vz').forEach(function(el){
      const s = state.get(el); if(s && s.drag){ s.x = e.clientX - s.drag.x; s.y = e.clientY - s.drag.y; apply(el, s); }
    });
  });
  document.addEventListener('mouseup', function(){
    document.querySelectorAll('.vz').forEach(function(el){ const s = state.get(el); if(s){ s.drag = null; el.style.cursor = 'grab'; } });
  });
  document.addEventListener('dblclick', function(e){
    const el = e.target.closest && e.target.closest('.vz'); if(!el) return;
    const s = get(el); s.k = 1; s.x = 0; s.y = 0; apply(el, s);
  });
  document.addEventListener('keydown', function(e){
    const t = e.target.tagName; if(t === 'INPUT' || t === 'TEXTAREA' || e.ctrlKey || e.metaKey || e.altKey) return;
    const map = {'1':'btn-same','2':'btn-different','3':'btn-cant-tell','z':'btn-undo','Z':'btn-undo','Backspace':'btn-back','f':'chk-full','F':'chk-full'};
    const id = map[e.key]; if(!id) return;
    const el = document.getElementById(id); if(!el) return;
    const target = el.tagName === 'BUTTON' ? el : el.querySelector('button,input');
    if(target){ target.click(); e.preventDefault(); }
  });
})();
</script>
"""


# ---------------------------------------------------------------------------
# Heartbeat for the pair on screen
# ---------------------------------------------------------------------------
def start_pair_heartbeat(pair_id):
    global heartbeat_timer

    def send_heartbeat():
        global heartbeat_timer
        if current_pair is not None and current_pair.get("id") == pair_id:
            update_heartbeat(pair_id, db_path)
            heartbeat_timer = threading.Timer(30.0, send_heartbeat)
            heartbeat_timer.daemon = True
            heartbeat_timer.start()

    stop_pair_heartbeat()
    send_heartbeat()


def stop_pair_heartbeat():
    global heartbeat_timer
    if heartbeat_timer:
        heartbeat_timer.cancel()
        heartbeat_timer = None


# ---------------------------------------------------------------------------
# View helpers
# ---------------------------------------------------------------------------
def _stats_msg(prefix):
    stats = get_instance_stats(db_path)
    # An answer still inside its undo grace period is not in the database yet.
    reviewed = stats['answered'] + (1 if pending is not None else 0)
    reviewed_str = f"{reviewed} / {review_limit}" if review_limit else str(reviewed)
    return (f"{prefix} | Reviewed: {reviewed_str} | Available: {stats['awaiting']} | "
            f"Active instances: {stats['active_instances']} | Instance: {INSTANCE_IDENTIFIER}")


def _undo_text():
    if pending is None:
        return ""
    remaining = max(0, math.ceil(pending["deadline"] - time.time()))
    return (f"Last answer: **{LABELS.get(pending['label'], pending['label'])}** "
            f"(pair {pending['pair']['id']}). Press **Undo** or **Z** within {remaining} s to take it back.")


def _view(status_msg):
    """Render the current state as the outputs tuple."""
    if current_pair is None:
        empty = _placeholder_html("No pair loaded")
        return empty, empty, status_msg, "", "", _undo_text()
    html1 = render_side(current_pair["image1"], current_pair["bbox1"], show_full)
    html2 = render_side(current_pair["image2"], current_pair["bbox2"], show_full)
    c1 = f"**Cluster ID: {current_pair.get('cluster1')}**"
    c2 = f"**Cluster ID: {current_pair.get('cluster2')}**"
    return html1, html2, status_msg, c1, c2, _undo_text()


def _skip():
    return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()


def _fetch_next():
    result = get_next_pair_atomic(db_path=db_path)
    if not result:
        return None
    pair_id, img1, img2, bbox1, bbox2, cluster1, cluster2, score = result
    return {"id": pair_id, "image1": img1, "image2": img2, "bbox1": bbox1, "bbox2": bbox2,
            "cluster1": cluster1, "cluster2": cluster2, "score": score}


# ---------------------------------------------------------------------------
# Decision bookkeeping
# ---------------------------------------------------------------------------
def commit_pending():
    """Write the held decision to the database (no-op if nothing is held)."""
    global pending
    if pending is None:
        return False
    ok = update_status(pending["pair"]["id"], pending["label"], db_path=db_path)
    if not ok:
        print(f"Warning: could not record decision for pair {pending['pair']['id']}")
    pending = None
    return ok


def commit_if_expired():
    if pending is not None and time.time() >= pending["deadline"]:
        commit_pending()
        return True
    return False


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def load_next_pair():
    global current_pair
    stop_pair_heartbeat()
    pair = _fetch_next()
    if pair is None:
        current_pair = None
        return _view(_stats_msg("No pairs available"))
    current_pair = pair
    start_pair_heartbeat(pair["id"])
    score = pair.get("score")
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
    return _view(_stats_msg(f"Loaded pair {pair['id']} | Score: {score_str}"))


def decide(label):
    """Hold the decision for the grace period and move on to the next pair."""
    global current_pair, pending
    if current_pair is None:
        return _view(_stats_msg("No pair loaded"))
    commit_pending()                       # the previous held answer becomes final
    stop_pair_heartbeat()
    pending = {"pair": current_pair, "label": label, "deadline": time.time() + UNDO_GRACE_SECONDS}
    history.append(current_pair)
    current_pair = None
    return load_next_pair()


def undo():
    """Take back the held answer; nothing was written. Otherwise behaves like Back."""
    global current_pair, pending
    if pending is None:
        return back()
    if current_pair is not None:
        release_pair(current_pair["id"], db_path=db_path)
    stop_pair_heartbeat()
    pair = pending["pair"]
    pending = None
    if history and history[-1]["id"] == pair["id"]:
        history.pop()
    current_pair = pair                    # still reserved by this instance
    start_pair_heartbeat(pair["id"])
    return _view(f"Answer withdrawn for pair {pair['id']}; nothing was recorded. Choose again.")


def back():
    """Return to the previous pair and clear its decision."""
    global current_pair, pending
    if pending is not None and history and history[-1]["id"] == pending["pair"]["id"]:
        return undo()
    commit_pending()
    if not history:
        return _view(_stats_msg("No previous pair to go back to"))

    previous = history.pop()
    ok, prev_status, prev_decision = reopen_pair(previous["id"], db_path=db_path)
    if not ok:
        if prev_status == "in_progress":
            note = "is being reviewed by another instance"
        elif prev_status is None:
            note = "no longer exists in the database"
        else:
            note = f"could not be reopened (status: {prev_status})"
        return _view(f"Previous pair {previous['id']} {note}.")

    if current_pair is not None:
        release_pair(current_pair["id"], db_path=db_path)
    stop_pair_heartbeat()
    current_pair = previous
    start_pair_heartbeat(previous["id"])

    answer = LABELS.get(prev_decision, prev_decision)
    if prev_status == "sent":
        msg = (f"Reopened pair {previous['id']}. Its answer '{answer}' was already consumed by the "
               f"clustering step; a new answer will be sent again on the next poll.")
    elif prev_status == "checked":
        msg = f"Reopened pair {previous['id']}. Its answer '{answer}' has been cleared; choose again."
    else:
        msg = f"Returned to pair {previous['id']} (it had no answer yet)."
    return _view(msg)


def toggle_full(value):
    global show_full
    show_full = bool(value)
    return _view(_stats_msg("Showing full images with box" if show_full else "Showing crops"))


def refresh():
    if current_pair is None:
        return load_next_pair()
    return _view(_stats_msg(f"Working on pair {current_pair['id']}"))


def timer_tick():
    """Once a second: commit an expired held answer, update the undo countdown,
    and pick up new pairs when nothing is on screen.

    Gradio dims every output of an event while it runs, so the tick only
    outputs the undo line and a hidden counter. When the panels need redrawing
    the tick queues the new view and bumps the counter, whose change event
    draws it; otherwise the panels are left alone and do not blink.
    """
    global queued_view, view_version
    committed = commit_if_expired()
    view = None
    if current_pair is None:
        stats = get_instance_stats(db_path)
        if stats["awaiting"] > 0:
            view = load_next_pair()
        elif committed:
            view = _view(_stats_msg("No pairs available"))
    if view is None:
        undo = _undo_text() if (pending is not None or committed) else gr.skip()
        return gr.skip(), undo
    queued_view = view
    view_version += 1
    return view_version, view[-1]


def show_queued_view():
    return queued_view


def cleanup_on_exit():
    commit_pending()
    stop_pair_heartbeat()
    if current_pair is not None and db_path:
        release_pair(current_pair["id"], db_path)


atexit.register(cleanup_on_exit)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
def build_interface():
    with gr.Blocks(head=HEAD, title="ID Verification") as demo:
        gr.Markdown(
            "## ID Verification Interface\n"
            "Scroll to zoom, drag to pan, double-click to reset. "
            "Keys: **1** Same, **2** Different, **3** Can't tell, **Z** Undo, **Backspace** Back, **F** full image."
        )
        status = gr.Textbox(label="Status", value="Loading...", interactive=False)
        undo_box = gr.Markdown("")

        with gr.Row():
            with gr.Column():
                cluster1_label = gr.Markdown("**Cluster ID: ?**", elem_id="cluster1")
                panel1 = gr.HTML(_placeholder_html("Loading..."), elem_id="panel1")
            with gr.Column():
                cluster2_label = gr.Markdown("**Cluster ID: ?**", elem_id="cluster2")
                panel2 = gr.HTML(_placeholder_html("Loading..."), elem_id="panel2")

        with gr.Row():
            full_toggle = gr.Checkbox(label="Show full image with box (F)", value=False, elem_id="chk-full")

        with gr.Row():
            btn_back = gr.Button("⬅ Back", elem_id="btn-back")
            btn_undo = gr.Button("↶ Undo (Z)", elem_id="btn-undo")
            btn_yes = gr.Button("Same (1)", elem_id="btn-same", variant="primary")
            btn_no = gr.Button("Different (2)", elem_id="btn-different", variant="primary")
            btn_cant_tell = gr.Button("Can't tell (3)", elem_id="btn-cant-tell")
            btn_refresh = gr.Button("🔄 Refresh")

        outputs = [panel1, panel2, status, cluster1_label, cluster2_label, undo_box]

        # The timer touches only the undo line and this hidden counter; see timer_tick.
        version = gr.Number(value=0, visible=False)
        timer = gr.Timer(value=1, active=True)
        timer.tick(fn=timer_tick, outputs=[version, undo_box], show_progress="hidden")
        version.change(show_queued_view, outputs=outputs, show_progress="hidden")

        btn_yes.click(lambda: decide("correct"), outputs=outputs)
        btn_no.click(lambda: decide("incorrect"), outputs=outputs)
        btn_cant_tell.click(lambda: decide("cant_tell"), outputs=outputs)
        btn_undo.click(undo, outputs=outputs)
        btn_back.click(back, outputs=outputs)
        btn_refresh.click(refresh, outputs=outputs)
        full_toggle.change(toggle_full, inputs=[full_toggle], outputs=outputs)
        demo.load(load_next_pair, outputs=outputs)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--allowed_dir', required=True, help='The directory that the GUI is allowed to access')
    parser.add_argument('--no_share', action='store_true', help='Do not create a public share link')
    parser.add_argument('--review_limit', type=int, default=None,
                        help='Review limit to show next to the reviewed count, e.g. the '
                             'max_human_reviews of the LCA config')
    args = parser.parse_args()
    print("GUI: Loading args...")

    configure(args.db, args.review_limit)
    init_db(db_path)
    start_heartbeat_system(db_path)
    print(f"Starting instance {INSTANCE_IDENTIFIER}")

    demo = build_interface()
    try:
        demo.launch(
            server_name="0.0.0.0",
            share=not args.no_share,
            allowed_paths=[args.allowed_dir]
        )
    finally:
        demo.close()
