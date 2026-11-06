
from pathlib import Path
import sys
import threading
import traceback
import uuid
import csv
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import rasterio
import torch

from mmcv import Config
from mmcv.parallel import collate
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
HELPER_DIR = ROOT / "hls-foundation-os"
CONFIG_PATH = HELPER_DIR / "configs" / "sen1floods11_config.py"
CHECKPOINT_NAME = "sen1floods11_Prithvi_100M.pth"

OUTPUT_DIR = ROOT / "outputs" / "app_predictions"
PROB_DIR = OUTPUT_DIR / "probabilities"
MAP_DIR = OUTPUT_DIR / "flood_maps"
TEMP_DIR = OUTPUT_DIR / "temporary_inputs"

DEMO_IMAGE = (
    ROOT / "data" / "Sen1Floods11-master"
    / "sample" / "S2" / "Spain_7370579_S2Hand.tif"
)

SELECTED_BANDS = [2, 3, 4, 9, 12, 13]
DEFAULT_THRESHOLD = 0.30

BG = "#080D13"
PANEL = "#111B25"
PANEL2 = "#192735"
GOLD = "#D6B36A"
TEXT = "#EDF2F5"
MUTED = "#A5B2BD"


# ============================================================
# IMAGE HELPERS
# ============================================================

def find_checkpoint():
    candidates = [
        ROOT / CHECKPOINT_NAME,
        ROOT / "checkpoints" / CHECKPOINT_NAME,
        HELPER_DIR / CHECKPOINT_NAME,
        ROOT / "models" / CHECKPOINT_NAME,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_NAME}"
    )


def make_rgb(path):
    with rasterio.open(path) as src:
        if src.count == 13:
            bands = src.read([4, 3, 2]).astype(np.float32)
        elif src.count == 6:
            bands = src.read([3, 2, 1]).astype(np.float32)
        else:
            raise ValueError("Expected a 6-band or 13-band image.")

    rgb = np.zeros(
        (bands.shape[1], bands.shape[2], 3),
        dtype=np.float32
    )

    for i in range(3):
        band = bands[i]
        finite = np.isfinite(band)

        if not finite.any():
            continue

        low, high = np.percentile(band[finite], [2, 98])

        if high <= low:
            high = low + 1

        rgb[:, :, i] = np.clip(
            (np.nan_to_num(band) - low) / (high - low),
            0,
            1
        )

    return rgb


def prepare_input(path):
    with rasterio.open(path) as src:
        if src.count == 13:
            indexes = SELECTED_BANDS
        elif src.count == 6:
            indexes = list(range(1, 7))
        else:
            raise ValueError(
                "Only 6-band or 13-band GeoTIFF is supported."
            )

        data = src.read(indexes)

        if src.width < 224 or src.height < 224:
            raise ValueError(
                "Image must be at least 224 x 224 pixels."
            )

        valid = np.all(
            src.read_masks(indexes) > 0,
            axis=0
        )

        if np.issubdtype(data.dtype, np.floating):
            valid &= np.all(np.isfinite(data), axis=0)

        if src.nodata is not None:
            valid &= np.all(data != src.nodata, axis=0)

        if not valid.any():
            raise ValueError("No valid image pixels found.")

        if src.count == 6:
            return str(path), valid, None

        profile = src.profile.copy()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = TEMP_DIR / (
        Path(path).stem + "_" + uuid.uuid4().hex + ".tif"
    )

    profile.update(count=6, dtype=data.dtype)

    with rasterio.open(temp_path, "w", **profile) as dst:
        dst.write(data)

    return str(temp_path), valid, temp_path


# ============================================================
# MAIN APPLICATION
# ============================================================

class FloodApp:

    def __init__(self, root):
        self.root = root
        self.root.title("AI Earth Guardian | Flood Intelligence")
        self.root.geometry("1450x950")
        self.root.minsize(1050, 760)
        self.root.configure(bg=BG)

        self.model = None
        self.pipeline = None

        self.input_path = None
        self.rgb = None
        self.probability = None
        self.valid_mask = None
        self.flood_map = None

        self.busy = False
        self.threshold = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.view_mode = tk.StringVar(value="Comparison")
        self.status = tk.StringVar(value="Ready")

        self.ui_queue = queue.Queue()

        self.drag_data = None
        self.axes = []
        self.map_axes = None

        self.pending_scroll = 0
        self.scroll_after_id = None

        self.build_ui()
        self.load_demo_image()

        self.root.after(80, self.process_ui_queue)

    # ========================================================
    # SCROLLABLE PAGE
    # ========================================================

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Gold.TButton",
            background=GOLD,
            foreground="#101820",
            font=("Segoe UI", 10, "bold"),
            padding=9
        )

        style.configure(
            "Dark.TButton",
            background=PANEL2,
            foreground=TEXT,
            padding=8
        )

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=PANEL2,
            background=GOLD
        )

        self.page = tk.Canvas(
            self.root,
            bg=BG,
            highlightthickness=0,
            bd=0
        )
        self.page.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(
            self.root,
            orient="vertical",
            command=self.page.yview
        )
        self.scrollbar.pack(side="right", fill="y")

        self.page.configure(
            yscrollcommand=self.scrollbar.set
        )

        self.outer = tk.Frame(
            self.page,
            bg=BG,
            padx=20,
            pady=16
        )

        self.window_id = self.page.create_window(
            (0, 0),
            window=self.outer,
            anchor="nw"
        )

        self.outer.bind(
            "<Configure>",
            self.update_scroll_region
        )

        self.page.bind(
            "<Configure>",
            self.resize_page_content
        )

        self.root.bind_all(
            "<MouseWheel>",
            self.route_mousewheel,
            add="+"
        )

        self.root.bind_all(
            "<Button-4>",
            self.route_mousewheel_linux,
            add="+"
        )

        self.root.bind_all(
            "<Button-5>",
            self.route_mousewheel_linux,
            add="+"
        )

        self.build_dashboard()

    def update_scroll_region(self, _event=None):
        self.page.configure(
            scrollregion=self.page.bbox("all")
        )

    def resize_page_content(self, event):
        self.page.itemconfigure(
            self.window_id,
            width=event.width
        )

    def is_widget_inside(self, widget, parent):
        while widget is not None:
            if widget == parent:
                return True

            widget = getattr(widget, "master", None)

        return False

    def route_mousewheel(self, event):
        widget = self.root.winfo_containing(
            event.x_root,
            event.y_root
        )

        if widget and self.is_widget_inside(
            widget,
            self.map_widget
        ):
            return "break"

        steps = int(-event.delta / 120)

        if steps == 0:
            steps = -1 if event.delta > 0 else 1

        self.queue_page_scroll(steps)
        return "break"

    def route_mousewheel_linux(self, event):
        widget = self.root.winfo_containing(
            event.x_root,
            event.y_root
        )

        if widget and self.is_widget_inside(
            widget,
            self.map_widget
        ):
            return "break"

        self.queue_page_scroll(
            -1 if event.num == 4 else 1
        )

        return "break"

    def queue_page_scroll(self, steps):
        self.pending_scroll += steps

        if self.scroll_after_id is None:
            self.scroll_after_id = self.root.after(
                12,
                self.apply_page_scroll
            )

    def apply_page_scroll(self):
        steps = self.pending_scroll
        self.pending_scroll = 0
        self.scroll_after_id = None

        if steps:
            self.page.yview_scroll(steps, "units")

    # ========================================================
    # DASHBOARD
    # ========================================================

    def build_dashboard(self):
        header = tk.Frame(self.outer, bg=BG)
        header.pack(fill="x", pady=(0, 16))

        tk.Label(
            header,
            text="◈  AI EARTH GUARDIAN",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 22, "bold")
        ).pack(side="left")

        tk.Label(
            header,
            text="DETECT  •  ANALYZE  •  PROTECT",
            bg=BG,
            fg=GOLD,
            font=("Segoe UI", 10, "bold")
        ).pack(side="right")

        body = tk.Frame(self.outer, bg=BG)
        body.pack(fill="both", expand=True)

        body.columnconfigure(0, weight=0, minsize=260)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # LEFT PANEL
        left = tk.Frame(
            body,
            bg=PANEL,
            padx=15,
            pady=16,
            width=260
        )

        left.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 14)
        )

        left.grid_propagate(False)

        tk.Label(
            left,
            text="MISSION CONTROL",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")

        tk.Label(
            left,
            text="Satellite flood monitoring",
            bg=PANEL,
            fg=MUTED
        ).pack(anchor="w", pady=(4, 16))

        tk.Label(
            left,
            text="SATELLITE INPUT",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w")

        self.file_label = tk.Label(
            left,
            text="No image selected",
            bg=PANEL2,
            fg=TEXT,
            justify="left",
            wraplength=220,
            padx=9,
            pady=10
        )

        self.file_label.pack(fill="x", pady=8)

        ttk.Button(
            left,
            text="Choose GeoTIFF",
            style="Dark.TButton",
            command=self.choose_file
        ).pack(fill="x", pady=3)

        ttk.Button(
            left,
            text="Load Demo Image",
            style="Dark.TButton",
            command=self.load_demo_image
        ).pack(fill="x", pady=3)

        tk.Frame(
            left,
            bg="#344454",
            height=1
        ).pack(fill="x", pady=14)

        tk.Label(
            left,
            text="DETECTION THRESHOLD",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w")

        self.threshold_label = tk.Label(
            left,
            text="30%",
            bg=PANEL,
            fg=GOLD,
            font=("Segoe UI", 20, "bold")
        )

        self.threshold_label.pack(anchor="w", pady=5)

        tk.Scale(
            left,
            from_=0.05,
            to=0.95,
            resolution=0.05,
            orient="horizontal",
            variable=self.threshold,
            showvalue=False,
            bg=PANEL,
            fg=TEXT,
            troughcolor="#536575",
            activebackground=GOLD,
            highlightthickness=0,
            command=self.update_threshold
        ).pack(fill="x", pady=(4, 12))

        self.run_button = ttk.Button(
            left,
            text="RUN FLOOD ANALYSIS",
            style="Gold.TButton",
            command=self.start_prediction
        )

        self.run_button.pack(fill="x", pady=5)

        self.progress = ttk.Progressbar(
            left,
            mode="indeterminate"
        )

        self.progress.pack(fill="x", pady=9)

        tk.Label(
            left,
            textvariable=self.status,
            bg=PANEL,
            fg="#B9D7C8",
            wraplength=220,
            justify="left",
            anchor="w"
        ).pack(fill="x", anchor="w", pady=5)

        tk.Frame(
            left,
            bg="#344454",
            height=1
        ).pack(fill="x", pady=13)

        tk.Label(
            left,
            text="MODEL INFORMATION",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w")

        tk.Label(
            left,
            text="Prithvi 100M\nSemantic segmentation\nDevice: CPU",
            bg=PANEL,
            fg=TEXT,
            justify="left"
        ).pack(anchor="w", pady=8)

        # RIGHT PANEL
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # METRIC CARDS
        cards = tk.Frame(right, bg=BG)
        cards.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 12)
        )

        for i in range(3):
            cards.columnconfigure(i, weight=1)

        self.metric_values = []

        titles = [
            "DETECTED FLOOD PIXELS",
            "CLASSIFIED PIXEL SHARE",
            "VALID IMAGE PIXELS"
        ]

        for i, title in enumerate(titles):
            card = tk.Frame(
                cards,
                bg=PANEL,
                padx=10,
                pady=11
            )

            card.grid(
                row=0,
                column=i,
                sticky="nsew",
                padx=(0, 7)
            )

            tk.Label(
                card,
                text=title,
                bg=PANEL,
                fg=MUTED,
                font=("Segoe UI", 8, "bold"),
                wraplength=140,
                justify="left"
            ).pack(anchor="w")

            value = tk.Label(
                card,
                text="Awaiting analysis",
                bg=PANEL,
                fg=TEXT,
                font=("Segoe UI", 11, "bold")
            )

            value.pack(anchor="w", pady=(7, 0))
            self.metric_values.append(value)

        # MAP PANEL
        map_panel = tk.Frame(
            right,
            bg=PANEL,
            padx=12,
            pady=12
        )

        map_panel.grid(
            row=1,
            column=0,
            sticky="nsew"
        )

        map_panel.columnconfigure(0, weight=1)
        map_panel.rowconfigure(1, weight=1)

        map_header = tk.Frame(map_panel, bg=PANEL)

        map_header.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )

        tk.Label(
            map_header,
            text="SATELLITE IMAGE ANALYSIS",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 12, "bold")
        ).pack(side="left")

        self.tab_buttons = {}

        tabs = tk.Frame(map_header, bg=PANEL)
        tabs.pack(side="right")

        for mode in ("Satellite", "Flood", "Comparison"):
            button = tk.Button(
                tabs,
                text=mode,
                command=lambda m=mode: self.set_view_mode(m),
                relief="flat",
                bd=0,
                padx=10,
                pady=6,
                font=("Segoe UI", 9, "bold"),
                cursor="hand2"
            )

            button.pack(side="left", padx=2)
            self.tab_buttons[mode] = button

        self.figure = Figure(
            figsize=(10, 6),
            dpi=90,
            facecolor=PANEL
        )

        self.figure_canvas = FigureCanvasTkAgg(
            self.figure,
            master=map_panel
        )

        self.map_widget = self.figure_canvas.get_tk_widget()

        self.map_widget.grid(
            row=1,
            column=0,
            sticky="nsew"
        )

        self.figure_canvas.mpl_connect(
            "scroll_event",
            self.on_map_scroll
        )

        self.figure_canvas.mpl_connect(
            "button_press_event",
            self.on_map_press
        )

        self.figure_canvas.mpl_connect(
            "button_release_event",
            self.on_map_release
        )

        self.figure_canvas.mpl_connect(
            "motion_notify_event",
            self.on_pan_move
        )

        footer = tk.Frame(map_panel, bg=PANEL)

        footer.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0)
        )

        tk.Label(
            footer,
            text="Wheel: Zoom  |  Drag: Pan  |  Double-click: Reset",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8)
        ).pack(side="left")

        ttk.Button(
            footer,
            text="FIT MAP",
            style="Dark.TButton",
            command=self.fit_map
        ).pack(side="right", padx=2)

        ttk.Button(
            footer,
            text="EXPORT PNG",
            style="Dark.TButton",
            command=self.export_png
        ).pack(side="right", padx=2)

        ttk.Button(
            footer,
            text="EXPORT CSV",
            style="Dark.TButton",
            command=self.export_csv
        ).pack(side="right", padx=2)

        # EXPLANATION PANEL
        explanation = tk.Frame(
            right,
            bg=PANEL,
            padx=15,
            pady=12
        )

        explanation.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(12, 0)
        )

        tk.Label(
            explanation,
            text="UNDERSTAND THIS IMAGE",
            bg=PANEL,
            fg=GOLD,
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")

        self.explanation_text = tk.Label(
            explanation,
            text=(
                "Satellite image: A natural-colour view of Earth's "
                "surface created from visible satellite bands.\n"
                "Red overlay: Pixels classified as flood by the model. "
                "This is a model prediction, not confirmed ground truth.\n"
                "Important: Normal rivers and lakes may also contain "
                "water. Water alone does not prove flooding."
            ),
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 9),
            justify="left",
            wraplength=950,
            anchor="w"
        )

        self.explanation_text.pack(
            fill="x",
            anchor="w",
            pady=(7, 0)
        )

        tk.Label(
            self.outer,
            text=(
                "AI Earth Guardian | Model-assisted flood mapping. "
                "Results require independent validation."
            ),
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 8)
        ).pack(anchor="w", pady=(12, 4))

        self.update_tab_styles()

    # ========================================================
    # INPUT FILES
    # ========================================================

    def load_demo_image(self):
        if not DEMO_IMAGE.exists():
            self.status.set(
                "Demo image not found. Choose a GeoTIFF."
            )
            return

        self.set_input(str(DEMO_IMAGE))

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Select Sentinel-2 GeoTIFF",
            filetypes=[
                ("GeoTIFF", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.set_input(path)

    def set_input(self, path):
        if self.busy:
            return

        try:
            with rasterio.open(path) as src:
                if src.count not in (6, 13):
                    raise ValueError(
                        "Only 6-band or 13-band GeoTIFF is supported."
                    )

                if src.width < 224 or src.height < 224:
                    raise ValueError(
                        "Image must be at least 224 x 224 pixels."
                    )

                details = (
                    f"{Path(path).name}\n"
                    f"{src.width} × {src.height} pixels\n"
                    f"{src.count} bands\n"
                    f"CRS: {src.crs or 'Not available'}"
                )

            rgb = make_rgb(path)

            self.input_path = path
            self.rgb = rgb
            self.probability = None
            self.valid_mask = None
            self.flood_map = None

            self.file_label.config(text=details)

            for metric in self.metric_values:
                metric.config(text="Awaiting analysis")

            self.view_mode.set("Comparison")
            self.update_tab_styles()
            self.draw_results(reset=True)

            self.status.set(
                "Image loaded. Ready for analysis."
            )

        except Exception as exc:
            messagebox.showerror("Input error", str(exc))

    # ========================================================
    # MODEL
    # ========================================================

    def load_model(self):
        if self.model is not None:
            return

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(str(CONFIG_PATH))

        checkpoint = find_checkpoint()

        if str(HELPER_DIR) not in sys.path:
            sys.path.insert(0, str(HELPER_DIR))

        import model_inference as helper

        cfg = Config.fromfile(str(CONFIG_PATH))
        cfg.model.backbone.pretrained = None

        model = init_segmentor(
            cfg,
            str(checkpoint),
            device="cpu"
        )

        pipeline_config = helper.process_test_pipeline(
            model.cfg.data.test.pipeline,
            bands=list(range(6))
        )

        self.pipeline = Compose(pipeline_config)
        self.model = model
        self.model.eval()

    def predict(self, path):
        model_path, valid_mask, temp_path = prepare_input(path)

        try:
            data = self.pipeline({
                "img_info": {"filename": model_path}
            })

            batch = collate(
                [data],
                samples_per_gpu=1
            )

            image = batch["img"][0]
            image_metas = batch["img_metas"].data[0][0]

            with torch.no_grad():
                output = self.model.inference(
                    image,
                    image_metas,
                    rescale=True
                )

            scores = (
                output[0, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if scores.shape != valid_mask.shape:
                raise ValueError(
                    "Prediction and image dimensions do not match."
                )

            scores[~valid_mask] = -1.0

            return scores, valid_mask

        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    # ========================================================
    # BACKGROUND THREAD
    # ========================================================

    def start_prediction(self):
        if self.busy:
            return

        if not self.input_path:
            messagebox.showwarning(
                "No image",
                "Choose an image first."
            )
            return

        self.busy = True
        self.run_button.config(state="disabled")
        self.progress.start(12)
        self.status.set("Starting model analysis...")

        threading.Thread(
            target=self.worker,
            args=(self.input_path,),
            daemon=True
        ).start()

    def worker(self, path):
        try:
            self.ui_queue.put((
                "status",
                "Loading Prithvi 100M model..."
            ))

            self.load_model()

            self.ui_queue.put((
                "status",
                "Running flood analysis on CPU..."
            ))

            scores, valid = self.predict(path)

            self.ui_queue.put((
                "success",
                (scores, valid, path)
            ))

        except Exception:
            self.ui_queue.put((
                "error",
                traceback.format_exc()
            ))

    def process_ui_queue(self):
        try:
            while True:
                task, payload = self.ui_queue.get_nowait()

                if task == "status":
                    self.status.set(payload)

                elif task == "success":
                    self.finish_success(*payload)

                elif task == "error":
                    self.finish_error(payload)

        except queue.Empty:
            pass

        self.root.after(
            80,
            self.process_ui_queue
        )

    def finish_success(self, scores, valid, path):
        self.probability = scores
        self.valid_mask = valid

        self.refresh_classification()

        try:
            self.save_outputs(
                path,
                scores,
                self.flood_map
            )

            self.status.set(
                "Analysis complete. GeoTIFF outputs saved."
            )

        except Exception as exc:
            self.status.set(
                f"Analysis complete; saving failed: {exc}"
            )

        self.busy = False
        self.progress.stop()
        self.run_button.config(state="normal")

    def finish_error(self, error):
        self.busy = False
        self.progress.stop()
        self.run_button.config(state="normal")

        self.status.set("Analysis failed.")

        messagebox.showerror(
            "Analysis error",
            error
        )

    # ========================================================
    # SAVE GEOTIFF OUTPUTS
    # ========================================================

    def save_outputs(self, path, scores, flood_map):
        PROB_DIR.mkdir(parents=True, exist_ok=True)
        MAP_DIR.mkdir(parents=True, exist_ok=True)

        stem = Path(path).stem

        with rasterio.open(path) as src:
            profile = src.profile.copy()

            profile.update(
                count=1,
                height=scores.shape[0],
                width=scores.shape[1],
                compress="lzw"
            )

            score_profile = profile.copy()
            score_profile.update(
                dtype="float32",
                nodata=-1.0
            )

            with rasterio.open(
                PROB_DIR / f"{stem}_model_scores.tif",
                "w",
                **score_profile
            ) as dst:
                dst.write(
                    scores.astype(np.float32),
                    1
                )

            map_profile = profile.copy()
            map_profile.update(
                dtype="uint8",
                nodata=255
            )

            with rasterio.open(
                MAP_DIR / f"{stem}_flood_map.tif",
                "w",
                **map_profile
            ) as dst:
                dst.write(
                    flood_map.astype(np.uint8),
                    1
                )

    # ========================================================
    # CLASSIFICATION AND METRICS
    # ========================================================

    def update_threshold(self, value):
        threshold = float(value)

        self.threshold_label.config(
            text=f"{threshold * 100:.0f}%"
        )

        if self.probability is not None:
            self.refresh_classification()

    def refresh_classification(self):
        threshold = float(self.threshold.get())

        self.flood_map = np.where(
            self.valid_mask,
            self.probability >= threshold,
            255
        ).astype(np.uint8)

        flood_pixels = int(
            np.count_nonzero(self.flood_map == 1)
        )

        valid_pixels = int(
            np.count_nonzero(self.valid_mask)
        )

        share = (
            flood_pixels / valid_pixels * 100
            if valid_pixels else 0
        )

        self.metric_values[0].config(
            text=f"{flood_pixels:,} pixels"
        )

        self.metric_values[1].config(
            text=f"{share:.2f}%"
        )

        self.metric_values[2].config(
            text=f"{valid_pixels:,} pixels"
        )

        self.draw_results()

    # ========================================================
    # TABS
    # ========================================================

    def set_view_mode(self, mode):
        if self.view_mode.get() == mode:
            return

        self.view_mode.set(mode)
        self.update_tab_styles()
        self.draw_results(reset=True)

    def update_tab_styles(self):
        selected = self.view_mode.get()

        for mode, button in self.tab_buttons.items():
            if mode == selected:
                button.configure(
                    bg=GOLD,
                    fg="#101820",
                    activebackground=GOLD,
                    activeforeground="#101820"
                )
            else:
                button.configure(
                    bg=PANEL2,
                    fg=TEXT,
                    activebackground="#26394A",
                    activeforeground=TEXT
                )

    # ========================================================
    # MAP DRAWING
    # ========================================================

    def draw_results(self, reset=False):
        if self.rgb is None:
            return

        height, width = self.rgb.shape[:2]
        mode = self.view_mode.get()

        self.figure.clear()
        self.axes = []

        if mode == "Comparison":
            # Equal panel dimensions and identical image extents.
            ax_left = self.figure.add_axes(
                [0.015, 0.025, 0.475, 0.91]
            )

            ax_right = self.figure.add_axes(
                [0.510, 0.025, 0.475, 0.91]
            )

            for ax in (ax_left, ax_right):
                ax.imshow(
                    self.rgb,
                    origin="upper",
                    extent=(0, width, height, 0),
                    interpolation="nearest"
                )

                ax.set_aspect("auto")
                ax.set_anchor("N")

            ax_left.set_title(
                "ORIGINAL SATELLITE IMAGE",
                color=TEXT,
                fontsize=10,
                pad=8
            )

            if self.flood_map is not None:
                overlay = np.zeros(
                    (height, width, 4),
                    dtype=np.float32
                )

                overlay[self.flood_map == 1] = [
                    1.0, 0.10, 0.02, 0.55
                ]

                ax_right.imshow(
                    overlay,
                    origin="upper",
                    extent=(0, width, height, 0),
                    interpolation="nearest"
                )

                ax_right.set_title(
                    "MODEL FLOOD DETECTION",
                    color=TEXT,
                    fontsize=10,
                    pad=8
                )

                ax_right.legend(
                    handles=[
                        Patch(
                            facecolor="#FF2E0A",
                            alpha=0.55,
                            label="Model-classified flood"
                        )
                    ],
                    loc="lower left",
                    facecolor=PANEL,
                    labelcolor="white",
                    fontsize=8
                )

            else:
                ax_right.set_title(
                    "RUN ANALYSIS TO VIEW DETECTION",
                    color=TEXT,
                    fontsize=10,
                    pad=8
                )

            self.axes = [ax_left, ax_right]
            self.map_axes = ax_left

        else:
            ax = self.figure.add_axes(
                [0.015, 0.025, 0.97, 0.91]
            )

            ax.imshow(
                self.rgb,
                origin="upper",
                extent=(0, width, height, 0),
                interpolation="nearest"
            )

            ax.set_aspect("auto")
            ax.set_anchor("N")

            if mode == "Satellite":
                ax.set_title(
                    "NATURAL COLOUR SATELLITE VIEW",
                    color=TEXT,
                    fontsize=10,
                    pad=8
                )

            elif mode == "Flood":
                ax.set_title(
                    "MODEL FLOOD DETECTION",
                    color=TEXT,
                    fontsize=10,
                    pad=8
                )

                if self.flood_map is not None:
                    overlay = np.zeros(
                        (height, width, 4),
                        dtype=np.float32
                    )

                    overlay[self.flood_map == 1] = [
                        1.0, 0.10, 0.02, 0.55
                    ]

                    ax.imshow(
                        overlay,
                        origin="upper",
                        extent=(0, width, height, 0),
                        interpolation="nearest"
                    )

                    ax.legend(
                        handles=[
                            Patch(
                                facecolor="#FF2E0A",
                                alpha=0.55,
                                label="Model-classified flood"
                            )
                        ],
                        loc="lower left",
                        facecolor=PANEL,
                        labelcolor="white"
                    )

                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Run analysis to view flood detection",
                        color=MUTED,
                        ha="center",
                        va="center",
                        transform=ax.transAxes
                    )

            self.axes = [ax]
            self.map_axes = ax

        for ax in self.axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#0B141D")

        self.figure.set_facecolor(PANEL)

        self.figure_canvas.draw()

        if reset:
            self.fit_map()
        else:
            self.sync_comparison_limits()

    # ========================================================
    # LINKED ZOOM AND PAN
    # ========================================================

    def sync_comparison_limits(self):
        if (
            self.view_mode.get() != "Comparison"
            or len(self.axes) != 2
        ):
            return

        xlim = self.map_axes.get_xlim()
        ylim = self.map_axes.get_ylim()

        for ax in self.axes:
            if ax is not self.map_axes:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        self.figure_canvas.draw_idle()

    def fit_map(self):
        if self.rgb is None:
            return

        height, width = self.rgb.shape[:2]

        for ax in self.axes:
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)

        self.figure_canvas.draw_idle()

    def zoom_at(self, x, y, factor):
        if self.map_axes is None or self.rgb is None:
            return

        ax = self.map_axes

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        new_width = min(
            max(abs(xlim[1] - xlim[0]) / factor, 40),
            self.rgb.shape[1]
        )

        new_height = min(
            max(abs(ylim[1] - ylim[0]) / factor, 40),
            self.rgb.shape[0]
        )

        fx = (x - xlim[0]) / (xlim[1] - xlim[0])
        fy = (y - ylim[0]) / (ylim[1] - ylim[0])

        left = x - fx * new_width
        top = y - fy * new_height

        ax.set_xlim(left, left + new_width)
        ax.set_ylim(top, top + new_height)

        self.sync_comparison_limits()
        self.figure_canvas.draw_idle()

    def on_map_scroll(self, event):
        if event.inaxes not in self.axes:
            return

        if event.xdata is None or event.ydata is None:
            return

        self.map_axes = event.inaxes

        factor = (
            1.15 if event.button == "up"
            else 1 / 1.15
        )

        self.zoom_at(
            event.xdata,
            event.ydata,
            factor
        )

    def on_map_press(self, event):
        if event.inaxes not in self.axes:
            return

        if event.button != 1:
            return

        if event.dblclick:
            self.fit_map()
            self.drag_data = None
            return

        self.map_axes = event.inaxes

        self.drag_data = {
            "x": event.x,
            "y": event.y,
            "xlim": event.inaxes.get_xlim(),
            "ylim": event.inaxes.get_ylim(),
            "moved": False
        }

    def on_pan_move(self, event):
        if self.drag_data is None:
            return

        if event.x is None or event.y is None:
            return

        data = self.drag_data

        dx_screen = event.x - data["x"]
        dy_screen = event.y - data["y"]

        if abs(dx_screen) + abs(dy_screen) > 4:
            data["moved"] = True

        if not data["moved"]:
            return

        ax = self.map_axes
        bbox = ax.get_window_extent()

        if bbox.width <= 0 or bbox.height <= 0:
            return

        x_range = data["xlim"][1] - data["xlim"][0]
        y_range = data["ylim"][1] - data["ylim"][0]

        dx = -dx_screen * x_range / bbox.width
        dy = dy_screen * y_range / bbox.height

        ax.set_xlim(
            data["xlim"][0] + dx,
            data["xlim"][1] + dx
        )

        ax.set_ylim(
            data["ylim"][0] + dy,
            data["ylim"][1] + dy
        )

        self.sync_comparison_limits()
        self.figure_canvas.draw_idle()

    def on_map_release(self, event):
        if self.drag_data is None:
            return

        was_drag = self.drag_data["moved"]
        self.drag_data = None

        if was_drag or self.probability is None:
            return

        if event.inaxes not in self.axes:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = int(event.xdata)
        y = int(event.ydata)

        height, width = self.probability.shape

        if not (0 <= x < width and 0 <= y < height):
            return

        if not self.valid_mask[y, x]:
            self.status.set(
                "Selected pixel has no valid data."
            )
            return

        score = float(self.probability[y, x])

        label = (
            "Flood-classified"
            if score >= float(self.threshold.get())
            else "Not classified as flood"
        )

        self.status.set(
            f"Pixel row {y}, column {x} | "
            f"Model score: {score * 100:.1f}% | {label}"
        )

    # ========================================================
    # EXPORT
    # ========================================================

    def export_png(self):
        if self.rgb is None:
            messagebox.showinfo(
                "No image",
                "Load an image first."
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )

        if not path:
            return

        try:
            self.figure.savefig(
                path,
                dpi=180,
                facecolor=PANEL,
                bbox_inches="tight"
            )

            self.status.set(
                "PNG exported successfully."
            )

        except Exception as exc:
            messagebox.showerror(
                "Export error",
                str(exc)
            )

    def export_csv(self):
        if self.probability is None:
            messagebox.showinfo(
                "No results",
                "Run flood analysis first."
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV report", "*.csv")]
        )

        if not path:
            return

        flood_pixels = int(
            np.count_nonzero(self.flood_map == 1)
        )

        valid_pixels = int(
            np.count_nonzero(self.valid_mask)
        )

        share = (
            flood_pixels / valid_pixels * 100
            if valid_pixels else 0
        )

        try:
            with open(
                path,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "AI Earth Guardian Flood Report"
                ])

                writer.writerow([
                    "Input image",
                    self.input_path
                ])

                writer.writerow([
                    "Model",
                    "Prithvi 100M"
                ])

                writer.writerow([
                    "Device",
                    "CPU"
                ])

                writer.writerow([
                    "Threshold",
                    self.threshold.get()
                ])

                writer.writerow([
                    "Detected flood pixels",
                    flood_pixels
                ])

                writer.writerow([
                    "Valid pixels",
                    valid_pixels
                ])

                writer.writerow([
                    "Classified pixel share (%)",
                    round(share, 3)
                ])

                writer.writerow([
                    "Interpretation",
                    "Model classification, not verified ground truth."
                ])

            self.status.set(
                "CSV report exported successfully."
            )

        except Exception as exc:
            messagebox.showerror(
                "CSV export error",
                str(exc)
            )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = FloodApp(root)
    root.mainloop()
