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
            rgb_indexes = [4, 3, 2]
        elif src.count == 6:
            rgb_indexes = [3, 2, 1]
        else:
            raise ValueError("Expected a 6-band or 13-band image.")
        bands = src.read(rgb_indexes).astype(np.float32)
        rgb_valid = np.all(src.read_masks(rgb_indexes) > 0, axis=0)
        if src.nodata is not None:
            rgb_valid &= np.all(bands != src.nodata, axis=0)
        rgb_valid &= np.all(np.isfinite(bands), axis=0)

    rgb = np.zeros(
        (bands.shape[1], bands.shape[2], 3),
        dtype=np.float32
    )

    for i in range(3):
        band = bands[i]
        finite = np.isfinite(band) & rgb_valid

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

    rgb[~rgb_valid] = 0.0
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
        self.root.title("TERRAFLUX | Satellite Flood Intelligence")
        self.root.geometry("1440x900")
        self.root.minsize(1280, 760)
        self.root.configure(bg=BG)

        self.model = None
        self.pipeline = None
        self.torch = None
        self.collate_fn = None

        self.input_path = None
        self.rgb = None
        self._raster_transform = None
        self._raster_crs = None
        self._coordinate_transformer = None
        self._row_area_cache = None
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
        self.view_limits = {}
        self.overlay_opacity = tk.DoubleVar(value=0.55)

        self.pending_scroll = 0
        self.scroll_after_id = None
        self.figures = {}
        self.canvases = {}
        self.axes_by_key = {}
        self.map_widgets = {}
        self.map_cards = {}
        self.map_axes = None

        self.build_ui()
        self.load_demo_image()
        self.root.after_idle(lambda: self.page.yview_moveto(0))

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
        # Keep the scrollbar visible: the dashboard is intentionally scrollable.
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

        # Scroll the page everywhere except over a Matplotlib map, where the
        # same wheel event is reserved for zooming that specific map axis.
        self.root.bind_all("<MouseWheel>", self.route_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self.route_mousewheel_linux, add="+")
        self.root.bind_all("<Button-5>", self.route_mousewheel_linux, add="+")
        self.root.bind_all("<Prior>", lambda _e: self.page.yview_scroll(-1, "pages"), add="+")
        self.root.bind_all("<Next>", lambda _e: self.page.yview_scroll(1, "pages"), add="+")

        self.build_dashboard()

    def update_scroll_region(self, _event=None):
        self.page.configure(scrollregion=self.page.bbox("all"))

    def resize_page_content(self, event):
        self.page.itemconfigure(self.window_id, width=event.width)
        self.page.after_idle(self.update_scroll_region)

    def is_widget_inside(self, widget, parent):
        while widget is not None:
            if widget == parent:
                return True

            widget = getattr(widget, "master", None)

        return False

    def pointer_over_map_axes(self, event):
        for key, widget in getattr(self, "map_widgets", {}).items():
            if not widget.winfo_exists(): continue
            lx = event.x_root - widget.winfo_rootx()
            ly = widget.winfo_height() - (event.y_root - widget.winfo_rooty())
            if lx < 0 or ly < 0 or lx > widget.winfo_width() or ly > widget.winfo_height():
                continue
            ax = self.axes_by_key.get(key)
            if ax is not None and ax.bbox.contains(lx, ly):
                return True
        return False


    def route_mousewheel(self, event):
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if self.pointer_over_map_axes(event):
            return None
        if hasattr(self, "scrollbar") and self.is_widget_inside(widget, self.scrollbar):
            return None
        steps = int(-event.delta / 120)
        if steps == 0:
            steps = -1 if event.delta > 0 else 1
        self.queue_page_scroll(steps)
        return "break"

    def route_mousewheel_linux(self, event):
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if self.pointer_over_map_axes(event):
            return None
        if hasattr(self, "scrollbar") and self.is_widget_inside(widget, self.scrollbar):
            return None
        self.queue_page_scroll(-1 if event.num == 4 else 1)
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

        # Do not clear Matplotlib references here. The maps must remain alive
        # while the dashboard scrolls; clearing these dictionaries broke map
        # zoom, pan, redraw and image interaction after the first scroll.
        if steps:
            self.page.yview_scroll(steps, "units")

    # ========================================================
    # DASHBOARD
    # ========================================================

    def build_dashboard(self):
        # Header remains part of the scrollable page; the dashboard reflows
        # while preserving the approved TERRAFLUX hierarchy.
        header = tk.Frame(self.outer, bg=BG, padx=12, pady=8)
        header.pack(fill="x", pady=(0, 10))
        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left", anchor="w")
        tk.Label(brand, text="◈", bg=BG, fg=GOLD,
                 font=("Segoe UI Symbol", 34, "bold")).pack(side="left", padx=(0, 14))
        namebox = tk.Frame(brand, bg=BG)
        namebox.pack(side="left", anchor="w")
        title = tk.Frame(namebox, bg=BG)
        title.pack(anchor="w")
        tk.Label(title, text="TERRA", bg=BG, fg=TEXT,
                 font=("Segoe UI", 27, "bold")).pack(side="left")
        tk.Label(title, text="FLUX", bg=BG, fg=GOLD,
                 font=("Segoe UI", 27, "bold")).pack(side="left")
        tk.Label(namebox, text="AI Earth Guardian · Satellite Flood Intelligence",
                 bg=BG, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(header, text="◎  Detect   •   Analyze   •   Protect", bg=BG,
                 fg=GOLD, font=("Segoe UI", 11, "bold")).pack(side="right", padx=12)
        tk.Frame(self.outer, bg="#26394A", height=1).pack(fill="x", pady=(0, 12))

        body = tk.Frame(self.outer, bg=BG)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=292)
        body.columnconfigure(1, weight=1)

        # Mission Control sidebar
        left = tk.Frame(body, bg=PANEL, padx=15, pady=14, width=292,
                        highlightbackground="#30465A", highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.grid_propagate(False)
        tk.Label(left, text="▰  MISSION CONTROL", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(left, text="Satellite flood monitoring", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(3, 14))
        tk.Label(left, text="▣  INPUT SOURCE", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.file_label = tk.Label(left, text="No image selected", bg=PANEL2,
            fg=TEXT, justify="left", wraplength=245, padx=9, pady=9,
            font=("Segoe UI", 8))
        self.file_label.pack(fill="x", pady=8)
        ttk.Button(left, text="📁  Choose GeoTIFF", style="Dark.TButton",
                   command=self.choose_file).pack(fill="x", pady=3)
        ttk.Button(left, text="▧  Load Demo Image", style="Dark.TButton",
                   command=self.load_demo_image).pack(fill="x", pady=3)
        tk.Frame(left, bg="#344454", height=1).pack(fill="x", pady=12)
        threshold_head = tk.Frame(left, bg=PANEL)
        threshold_head.pack(fill="x")
        tk.Label(threshold_head, text="⚙  DETECTION THRESHOLD", bg=PANEL,
                 fg=MUTED, font=("Segoe UI", 8, "bold")).pack(side="left")
        self.threshold_label = tk.Label(threshold_head, text="30%", bg=PANEL,
                 fg=GOLD, font=("Segoe UI", 18, "bold"))
        self.threshold_label.pack(side="right")
        tk.Scale(left, from_=0.05, to=0.95, resolution=0.05,
                 orient="horizontal", variable=self.threshold, showvalue=False,
                 bg=PANEL, fg=TEXT, troughcolor="#536575", activebackground=GOLD,
                 highlightthickness=0, bd=0, command=self.update_threshold
                 ).pack(fill="x", pady=(2, 0))
        tk.Frame(left, bg=PANEL).pack(fill="x")
        endpoints = tk.Frame(left, bg=PANEL)
        endpoints.pack(fill="x")
        tk.Label(endpoints, text="5%", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        tk.Label(endpoints, text="95%", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 8, "bold")).pack(side="right")
        tk.Label(left, text="Higher threshold = stricter detection", bg=PANEL2,
                 fg=TEXT, font=("Segoe UI", 8), padx=6, pady=4).pack(fill="x", pady=(3, 9))
        self.run_button = ttk.Button(left, text="▶  RUN ANALYSIS",
                 style="Gold.TButton", command=self.start_prediction)
        self.run_button.pack(fill="x", pady=3)
        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", pady=7)
        self.status_label = tk.Label(left, textvariable=self.status, bg=PANEL,
                 fg="#B9D7C8", justify="left", anchor="w", wraplength=245,
                 font=("Segoe UI", 8))
        self.status_label.pack(fill="x", pady=(2, 10))
        tk.Frame(left, bg="#344454", height=1).pack(fill="x", pady=10)
        tk.Label(left, text="▣  MODEL INFORMATION", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(left, text="Model: Prithvi 100M\nFramework: MMSegmentation\nDevice: CPU",
                 bg=PANEL, fg=TEXT, justify="left", font=("Segoe UI", 9)
                 ).pack(anchor="w", pady=8)

        # Main dashboard
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        cards = tk.Frame(right, bg=BG)
        cards.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        for i in range(4): cards.columnconfigure(i, weight=1, uniform="metric")
        self.metric_values = []
        metric_titles = ("Flooded Pixels", "Non-Flooded Pixels", "Valid Pixels", "Flooded Share")
        metric_icons = ("≋", "▲", "●", "◔")
        for i, (title_text, icon) in enumerate(zip(metric_titles, metric_icons)):
            card = tk.Frame(cards, bg=PANEL, padx=10, pady=10,
                            highlightbackground="#30465A", highlightthickness=1)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            tk.Label(card, text=icon, bg=PANEL, fg=("#168BFF", "#29C978", "#8B5CF6", GOLD)[i],
                     font=("Segoe UI Symbol", 22, "bold")).pack(side="left", padx=(0, 8))
            textbox = tk.Frame(card, bg=PANEL)
            textbox.pack(side="left", fill="x", expand=True)
            tk.Label(textbox, text=title_text, bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 8), wraplength=125, justify="left").pack(anchor="w")
            value = tk.Label(textbox, text="—", bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 12, "bold"))
            value.pack(anchor="w", pady=(3, 0))
            self.metric_values.append(value)

        tabs_row = tk.Frame(right, bg=BG)
        tabs_row.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        self.tab_buttons = {}
        for mode, label in (("Satellite", "✥  Satellite"), ("Flood", "≋  Flood"),
                            ("Comparison", "⚖  Comparison")):
            b = tk.Button(tabs_row, text=label, command=lambda m=mode: self.set_view_mode(m),
                relief="flat", bd=0, padx=14, pady=9, cursor="hand2",
                font=("Segoe UI", 9, "bold"), bg=PANEL2, fg=TEXT)
            b.pack(side="left", expand=True, fill="x", padx=(0, 5))
            self.tab_buttons[mode] = b

        # Two truly separate map cards, each with its own Figure, axes and controls.
        maps_row = tk.Frame(right, bg=BG, width=900, height=360)
        maps_row.grid(row=2, column=0, sticky="nsew")
        maps_row.grid_propagate(False)
        maps_row.columnconfigure(0, weight=1, uniform="maps")
        maps_row.columnconfigure(1, weight=1, uniform="maps")
        self.map_host = maps_row
        self.map_cards = {}
        self.figures = {}
        self.canvases = {}
        self.axes_by_key = {}
        self.map_widgets = {}
        for col, key, heading, icon in (
            (0, "satellite", "Original Satellite Image", "▧"),
            (1, "flood", "Model Flood Detection", "◆")
        ):
            card = tk.Frame(maps_row, bg=PANEL, padx=8, pady=8,
                            highlightbackground="#30465A", highlightthickness=1)
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 5, 5 if col == 0 else 0))
            self.map_cards[key] = card
            top = tk.Frame(card, bg=PANEL)
            top.pack(fill="x", pady=(0, 5))
            tk.Label(top, text=f"{icon}  {heading}", bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 9, "bold")).pack(side="left")
            controls = tk.Frame(top, bg=PANEL)
            controls.pack(side="right")
            for symbol, factor in (("+", 1.2), ("−", 1/1.2), ("⟳", 0)):
                ttk.Button(controls, text=symbol, style="Dark.TButton", width=3,
                    command=lambda k=key, f=factor: self.control_zoom(k, f)
                    ).pack(side="left", padx=2)
            fig = Figure(figsize=(5.0, 3.0), dpi=85, facecolor=PANEL)
            canvas = FigureCanvasTkAgg(fig, master=card)
            widget = canvas.get_tk_widget()
            # TkAgg otherwise requests its default 560x420 widget size,
            # forcing the two-column dashboard wider than the window.
            widget.configure(width=420, height=300, highlightthickness=0, bg="#0B141D")
            widget.pack(fill="both", expand=True)
            ax = fig.add_axes([0.015, 0.025, 0.97, 0.94])
            self.figures[key], self.canvases[key] = fig, canvas
            self.axes_by_key[key] = ax
            self.map_widgets[key] = widget
            canvas.mpl_connect("scroll_event", self.on_map_scroll)
            canvas.mpl_connect("button_press_event", self.on_map_press)
            canvas.mpl_connect("button_release_event", self.on_map_release)
            canvas.mpl_connect("motion_notify_event", self.on_pan_move)
            canvas.mpl_connect("motion_notify_event", self.on_map_motion)
            widget.bind(
                "<Configure>",
                lambda _event, c=canvas: c.draw_idle(),
                add="+"
            )

        opacity_row = tk.Frame(right, bg=BG)
        opacity_row.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        tk.Label(opacity_row, text="OVERLAY OPACITY", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="left", padx=(2, 6))
        self.opacity_label = tk.Label(opacity_row, text="55%", bg=BG, fg=GOLD,
                 font=("Segoe UI", 8, "bold"))
        self.opacity_label.pack(side="left")
        tk.Scale(opacity_row, from_=0.50, to=0.60, resolution=0.01,
                 orient="horizontal", variable=self.overlay_opacity, showvalue=False,
                 length=125, bg=BG, fg=TEXT, troughcolor="#536575",
                 activebackground=GOLD, highlightthickness=0, bd=0,
                 command=self.update_overlay_opacity).pack(side="left", padx=5)
        tk.Label(opacity_row, text="Scroll over a map to zoom · Drag to pan · Double-click to reset",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8)).pack(side="left", padx=12)

        bottom = tk.Frame(right, bg=BG)
        bottom.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        bottom.columnconfigure(0, weight=6)
        bottom.columnconfigure(1, weight=5)
        insight = tk.Frame(bottom, bg=PANEL, padx=13, pady=11,
                           highlightbackground="#30465A", highlightthickness=1)
        insight.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        tk.Label(insight, text="✦  KEY INSIGHTS", bg=PANEL, fg=GOLD,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.insights_label = tk.Label(insight,
            text="Insights will appear after a successful analysis.", bg=PANEL,
            fg=MUTED, font=("Segoe UI", 8), justify="left", anchor="w")
        self.insights_label.pack(fill="x", pady=(7, 5))
        insight_row = tk.Frame(insight, bg=PANEL)
        insight_row.pack(fill="x", expand=True)
        for i in range(3):
            insight_row.columnconfigure(i, weight=1, uniform="insight_columns")
        self.insight_area_value = tk.Label(insight_row, text="Flooded Area\n—",
            bg=PANEL, fg=TEXT, font=("Segoe UI", 8, "bold"), justify="left", anchor="w")
        self.insight_area_value.grid(row=0, column=0, sticky="ew", padx=(0, 7))
        self.insight_section_value = tk.Label(insight_row, text="Most Affected Section\n—",
            bg=PANEL, fg=TEXT, font=("Segoe UI", 8, "bold"), justify="left", anchor="w")
        self.insight_section_value.grid(row=0, column=1, sticky="ew", padx=7)
        self.insight_share_value = tk.Label(insight_row, text="Flooded Share\n—",
            bg=PANEL, fg=TEXT, font=("Segoe UI", 8, "bold"), justify="left", anchor="w")
        self.insight_share_value.grid(row=0, column=2, sticky="ew", padx=(7, 0))
        exports = tk.Frame(bottom, bg=PANEL, padx=11, pady=11,
                           highlightbackground="#30465A", highlightthickness=1)
        exports.grid(row=0, column=1, sticky="nsew")
        tk.Label(exports, text="⇩  EXPORT OPTIONS", bg=PANEL, fg=GOLD,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 7))
        buttons = tk.Frame(exports, bg=PANEL)
        buttons.pack(fill="x")
        for label, cmd in (("Export PNG", self.export_png), ("Export CSV", self.export_csv),
                           ("Export GeoTIFF", self.export_geotiff), ("Reset All", self.reset_all)):
            ttk.Button(buttons, text=label, style="Dark.TButton", command=cmd
                       ).pack(side="left", padx=(0, 4), fill="x", expand=True)

        tk.Label(self.outer, text="TERRAFLUX  ·  Model-assisted flood mapping. Results require independent validation.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w", pady=(10, 4))
        self.update_tab_styles()
        self.draw_results(reset=True)
        self.root.after_idle(lambda: self.page.yview_moveto(0))


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
                    f"CRS: {src.crs or 'Not available'}\n"
                    f"Resolution: {src.res}\n"
                    f"Bounds: {tuple(round(v, 5) for v in src.bounds)}"
                )
                raster_transform = src.transform
                raster_crs = src.crs

            rgb = make_rgb(path)

            self.input_path = path
            self.rgb = rgb
            self._raster_transform = raster_transform
            self._raster_crs = raster_crs
            self._row_area_cache = None
            self._coordinate_transformer = None
            if raster_crs is not None:
                try:
                    from pyproj import Transformer
                    self._coordinate_transformer = Transformer.from_crs(
                        raster_crs, "EPSG:4326", always_xy=True
                    )
                except Exception:
                    self._coordinate_transformer = None
            self.probability = None
            self.valid_mask = None
            self.flood_map = None
            self.view_limits.clear()
            self.clear_insights()

            self.file_label.config(text=details)

            for metric in self.metric_values:
                metric.config(text="—")

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

        try:
            import torch
            from mmcv import Config
            from mmcv.parallel import collate
            from mmseg.apis import init_segmentor
            from mmseg.datasets.pipelines import Compose
        except ImportError as exc:
            raise RuntimeError(
                "Prithvi dependencies are missing. Install the project's "
                "compatible PyTorch, MMCV and MMSegmentation versions."
            ) from exc

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
        self.torch = torch
        self.collate_fn = collate
        self.model.eval()

    def predict(self, path):
        model_path, valid_mask, temp_path = prepare_input(path)

        try:
            data = self.pipeline({
                "img_info": {"filename": model_path}
            })

            batch = self.collate_fn(
                [data],
                samples_per_gpu=1
            )

            image = batch["img"][0]
            image_metas = batch["img_metas"].data[0][0]

            with self.torch.no_grad():
                output = self.model.inference(
                    image,
                    image_metas,
                    rescale=True
                )

            if not self.torch.is_tensor(output) or output.ndim != 4:
                raise RuntimeError(
                    "Unexpected Prithvi inference output. Expected a 4-D "
                    "(batch, classes, height, width) logits tensor."
                )
            if output.shape[0] != 1 or output.shape[1] < 2:
                raise RuntimeError(
                    f"Expected one sample and at least two classes; got {tuple(output.shape)}."
                )
            # MMSegmentation inference returns class logits. Convert them to
            # class probabilities before applying the 0.05–0.95 threshold.
            probabilities = self.torch.softmax(output, dim=1)
            scores = (
                probabilities[0, 1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if scores.shape != valid_mask.shape:
                raise ValueError(
                    "Prediction and image dimensions do not match."
                )

            if not np.all(np.isfinite(scores[valid_mask])):
                raise RuntimeError("Model produced non-finite probabilities for valid pixels.")
            if np.any(scores[valid_mask] < 0.0) or np.any(scores[valid_mask] > 1.0):
                raise RuntimeError("Model probabilities are outside the expected [0, 1] range.")
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
        self.probability = None
        self.valid_mask = None
        self.flood_map = None
        self.clear_insights()
        for metric in self.metric_values:
            metric.config(text="—")
        self.view_limits.clear()
        self.draw_results(reset=True)
        self.run_button.config(state="disabled")
        self.progress.start(12)
        self.status.set("Analyzing... Loading Prithvi 100M on CPU.")

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
                try:
                    if task == "status":
                        self.status.set(payload)
                    elif task == "success":
                        self.finish_success(*payload)
                    elif task == "error":
                        self.finish_error(payload)
                except Exception as exc:
                    self.busy = False
                    self.progress.stop()
                    self.run_button.config(state="normal")
                    self.status.set(f"UI update failed · {str(exc)[:90]}")
                    print("TERRAFLUX UI callback error:", traceback.format_exc())
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(80, self.process_ui_queue)

    def finish_success(self, scores, valid, path):
        # End the processing state before redrawing; otherwise the map renderer
        # would leave its temporary "Analyzing" label over a completed result.
        self.busy = False
        self.progress.stop()
        self.run_button.config(state="normal")
        self.probability = scores
        self.valid_mask = valid

        self.refresh_classification()

        try:
            self.save_outputs(path, scores, self.flood_map)
            self.status.set("Ready · Analysis completed successfully.")
        except Exception as exc:
            self.status.set(f"Analysis complete; saving failed: {exc}")

    def finish_error(self, error):
        self.busy = False
        self.progress.stop()
        self.run_button.config(state="normal")

        short_error = error.strip().splitlines()[-1] if error.strip() else "Unknown error"
        self.status.set(f"Failed · {short_error[:100]}")
        self.draw_results(reset=True)
        print("TERRAFLUX analysis traceback:\n", error)
        messagebox.showerror("Analysis failed", short_error)

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

    def clear_insights(self):
        if not hasattr(self, "insights_label"):
            return
        self.insight_area_value.config(text="Flooded Area\n—")
        self.insight_section_value.config(text="Most Affected Section\n—")
        self.insight_share_value.config(text="Flooded Share\n—")
        if not self.insights_label.winfo_manager():
            self.insights_label.pack(fill="x", pady=(7, 5))

    # ========================================================
    # CLASSIFICATION AND METRICS
    # ========================================================

    def update_threshold(self, value):
        threshold = float(value)

        self.threshold_label.config(
            text=f"{threshold * 100:.0f}%"
        )

        if self.probability is not None and self.valid_mask is not None:
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

        non_flooded = valid_pixels - flood_pixels
        self.metric_values[0].config(text=f"{flood_pixels:,} px")
        self.metric_values[1].config(text=f"{non_flooded:,} px")
        self.metric_values[2].config(text=f"{valid_pixels:,} px")
        self.metric_values[3].config(text=f"{share:.2f}%")
        grid = []
        h, w = self.flood_map.shape
        for row in range(3):
            for col in range(3):
                y0, y1 = row * h // 3, (row + 1) * h // 3
                x0, x1 = col * w // 3, (col + 1) * w // 3
                region_valid = self.valid_mask[y0:y1, x0:x1]
                region_flood = self.flood_map[y0:y1, x0:x1] == 1
                grid.append((int(np.count_nonzero(region_flood & region_valid)), row + 1, col + 1))
        area_km2 = self.calculate_flooded_area_km2()
        area_text = f"≈ {area_km2:.2f} km²" if area_km2 is not None else "Unavailable"
        if grid and flood_pixels:
            peak = max(grid)
            position_names = {
                (1, 1): "Top Left", (1, 2): "Top Center", (1, 3): "Top Right",
                (2, 1): "Middle Left", (2, 2): "Center Region", (2, 3): "Middle Right",
                (3, 1): "Bottom Left", (3, 2): "Bottom Center", (3, 3): "Bottom Right"
            }
            section = position_names.get((peak[1], peak[2]), f"Row {peak[1]}, Column {peak[2]}")
            self.insight_section_value.config(text=f"Most Affected Section\n{section}\n(3×3 grid · {peak[0]:,} px)")
        else:
            self.insight_section_value.config(text="Most Affected Section\nNo flooded pixels")
        self.insights_label.pack_forget()
        self.insight_area_value.config(text=f"Flooded Area\n{area_text}")
        self.insight_share_value.config(text=f"Flooded Share\n{share:.2f}%")

        self.draw_results()

    # ========================================================
    # TABS
    # ========================================================

    def set_view_mode(self, mode):
        if mode not in ("Satellite", "Flood", "Comparison"):
            return
        self.view_mode.set(mode)
        self.update_tab_styles()
        self.draw_results(reset=False)


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
        if not self.figures or not self.axes_by_key:
            return

        h, w = self.rgb.shape[:2] if self.rgb is not None else (1, 1)
        mode = self.view_mode.get()
        visible = (
            ("satellite", "flood") if mode == "Comparison"
            else (("satellite",) if mode == "Satellite" else ("flood",))
        )

        for key in ("satellite", "flood"):
            fig = self.figures[key]
            canvas = self.canvases[key]
            ax = self.axes_by_key[key]

            # Save this map's limits separately before clearing its contents.
            old_xlim = ax.get_xlim()
            old_ylim = ax.get_ylim()
            had_custom_view = (
                not reset
                and self.rgb is not None
                and (
                    not np.allclose(old_xlim, (0, w))
                    or not np.allclose(old_ylim, (h, 0))
                )
            )

            # Clear the existing Axes instead of deleting/recreating it.
            # This keeps event routing and each map's identity stable.
            ax.clear()
            ax.set_facecolor("#050A12")
            ax.set_position([0.005, 0.005, 0.99, 0.99])
            ax.set_aspect("auto")
            ax.set_xticks([])
            ax.set_yticks([])

            if key == "satellite":
                if self.rgb is None:
                    ax.text(
                        0.5, 0.5, "Choose a GeoTIFF to begin",
                        color=MUTED, ha="center", va="center",
                        transform=ax.transAxes
                    )
                else:
                    ax.imshow(
                        self.rgb,
                        origin="upper",
                        extent=(0, w, h, 0),
                        interpolation="bilinear",
                        aspect="auto",
                        zorder=1
                    )
                    if self.busy:
                        ax.text(
                            0.5, 0.5, "Analyzing satellite data...",
                            color=TEXT, ha="center", va="center",
                            transform=ax.transAxes,
                            bbox={
                                "facecolor": PANEL,
                                "alpha": 0.88,
                                "edgecolor": GOLD,
                                "pad": 6
                            },
                            zorder=10
                        )
            else:
                if self.flood_map is None:
                    msg = (
                        "Choose a GeoTIFF to begin"
                        if self.rgb is None
                        else (
                            "Analyzing..."
                            if self.busy
                            else "Run analysis to view flood detection"
                        )
                    )
                    ax.text(
                        0.5, 0.5, msg, color=MUTED,
                        ha="center", va="center",
                        transform=ax.transAxes
                    )
                else:
                    rgba = np.zeros((h, w, 4), dtype=np.float32)
                    rgba[self.flood_map == 0] = [0.02, 0.04, 0.08, 1.0]
                    rgba[self.flood_map == 1] = [
                        0.02, 0.42, 1.0,
                        float(self.overlay_opacity.get())
                    ]
                    rgba[self.flood_map == 255] = [0, 0, 0, 0]
                    ax.imshow(
                        rgba,
                        origin="upper",
                        extent=(0, w, h, 0),
                        interpolation="nearest",
                        aspect="auto",
                        zorder=1
                    )
                    ax.legend(
                        handles=[
                            Patch(
                                facecolor="#0878FF",
                                alpha=0.9,
                                label="Flood Detected"
                            ),
                            Patch(
                                facecolor="#050A12",
                                label="Not Flooded"
                            )
                        ],
                        loc="lower right",
                        facecolor=PANEL,
                        labelcolor=TEXT,
                        fontsize=7
                    )

            # Set image-coordinate bounds explicitly after imshow. This makes
            # the raster occupy the complete map viewport rather than a small
            # aspect-constrained patch in the upper part of the card.
            if self.rgb is not None:
                # Add scale-bar artists first, then restore exact map limits;
                # otherwise Matplotlib's autoscaling can shrink the raster.
                self.add_scale_bar(ax)

                if had_custom_view:
                    ax.set_xlim(old_xlim)
                    ax.set_ylim(old_ylim)
                else:
                    ax.set_xlim(0, w)
                    ax.set_ylim(h, 0)

            fig.set_facecolor(PANEL)
            canvas.draw_idle()

            card = self.map_cards[key]
            if len(visible) == 2:
                card.grid(
                    row=0,
                    column=0 if key == "satellite" else 1,
                    sticky="nsew",
                    padx=(
                        (0, 5) if key == "satellite"
                        else (5, 0)
                    )
                )
            elif key in visible:
                card.grid(
                    row=0,
                    column=0,
                    columnspan=2,
                    sticky="nsew",
                    padx=0
                )
            else:
                card.grid_remove()

        self.axes = [self.axes_by_key[key] for key in visible]
        self.map_axes = self.axes[0] if self.axes else None

    def _raster_metadata(self):
        if not self.input_path:
            return None
        with rasterio.open(self.input_path) as src:
            return {
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
                "resolution": src.res,
            }

    def _pixel_area_m2_by_row(self):
        """Return per-row pixel area in m²; geographic CRS uses geodesic area."""
        if not self.input_path:
            return None
        if getattr(self, "_row_area_cache", None) is not None:
            return self._row_area_cache
        with rasterio.open(self.input_path) as src:
            if src.crs is None:
                return None
            transform = src.transform
            crs = src.crs
            if crs.is_projected:
                try:
                    factor = float(crs.linear_units_factor[1])
                except Exception:
                    factor = 1.0
                self._row_area_cache = np.full(src.height, abs(transform.a * transform.e - transform.b * transform.d) * factor * factor, dtype=np.float64)
                return self._row_area_cache
            if crs.is_geographic:
                try:
                    from pyproj import CRS as PyCRS
                    geod = PyCRS.from_user_input(crs).get_geod()
                except Exception:
                    return None
                areas = np.zeros(src.height, dtype=np.float64)
                col = src.width // 2
                for row in range(src.height):
                    corners = [
                        transform * (col, row),
                        transform * (col + 1, row),
                        transform * (col + 1, row + 1),
                        transform * (col, row + 1),
                    ]
                    lons = [pt[0] for pt in corners]
                    lats = [pt[1] for pt in corners]
                    area, _ = geod.polygon_area_perimeter(lons, lats)
                    areas[row] = abs(area)
                self._row_area_cache = areas
                return self._row_area_cache
        return None

    def calculate_flooded_area_km2(self):
        if self.flood_map is None or self.valid_mask is None:
            return None
        row_areas = self._pixel_area_m2_by_row()
        if row_areas is None:
            return None
        flooded_by_row = np.count_nonzero(self.flood_map == 1, axis=1)
        return float(np.sum(flooded_by_row * row_areas) / 1_000_000.0)

    def pixel_latlon(self, row, col):
        raster_transform = getattr(self, "_raster_transform", None)
        coordinate_transformer = getattr(self, "_coordinate_transformer", None)
        if raster_transform is None or coordinate_transformer is None:
            return None
        try:
            from rasterio.transform import xy
            x, y = xy(raster_transform, row, col, offset="center")
            lon, lat = coordinate_transformer.transform(x, y)
            return float(lat), float(lon)
        except Exception:
            return None

    def add_scale_bar(self, ax):
        """Draw a ground-distance scale bar using raster CRS units."""
        if not self.input_path or self.rgb is None:
            return
        try:
            with rasterio.open(self.input_path) as src:
                if src.crs is None:
                    return
                row = src.height // 2
                col = src.width // 2
                if src.crs.is_projected:
                    try:
                        unit_factor = float(src.crs.linear_units_factor[1])
                    except Exception:
                        unit_factor = 1.0
                    metres_per_pixel = float(np.hypot(src.transform.a, src.transform.d)) * unit_factor
                elif src.crs.is_geographic:
                    from pyproj import CRS as PyCRS
                    geod = PyCRS.from_user_input(src.crs).get_geod()
                    p1 = src.transform * (col, row)
                    p2 = src.transform * (col + 1, row)
                    metres_per_pixel = abs(geod.inv(p1[0], p1[1], p2[0], p2[1])[2])
                else:
                    return
                if not np.isfinite(metres_per_pixel) or metres_per_pixel <= 0:
                    return
                target_m = 2000.0
                pixels = target_m / metres_per_pixel
                if pixels > src.width * 0.30:
                    target_m = 1000.0
                    pixels = target_m / metres_per_pixel
                if pixels > src.width * 0.30:
                    target_m = 500.0
                    pixels = target_m / metres_per_pixel
                if pixels <= 0 or pixels > src.width * 0.45:
                    return
                x0 = src.width * 0.04
                x1 = x0 + pixels
                y = src.height * 0.94
                ax.plot([x0, x1], [y, y], color="white", linewidth=3, solid_capstyle="butt", zorder=20)
                ax.plot([x0, x0], [y - src.height * 0.008, y + src.height * 0.008], color="white", linewidth=1.2, zorder=20)
                ax.plot([x1, x1], [y - src.height * 0.008, y + src.height * 0.008], color="white", linewidth=1.2, zorder=20)
                label = f"{target_m / 1000:g} km" if target_m >= 1000 else f"{target_m:g} m"
                ax.text(x0, y - src.height * 0.025, label, color="white", fontsize=7, ha="left", va="bottom", zorder=21, bbox={"facecolor":"#080D13", "alpha":0.65, "edgecolor":"none", "pad":1})
        except Exception:
            return

    # ========================================================
    # LINKED ZOOM AND PAN
    # ========================================================

    def sync_comparison_limits(self):
        # Comparison maps intentionally keep independent view limits.
        return

    def update_overlay_opacity(self, value):
        self.opacity_label.config(text=f"{float(value) * 100:.0f}%")
        if self.rgb is not None:
            self.draw_results()

    def control_zoom(self, map_key, factor):
        ax = self.axes_by_key.get(map_key)
        if ax is None or self.rgb is None:
            return
        if factor == 0:
            self.fit_map(ax=ax)
        else:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            self.zoom_at(sum(xlim)/2, sum(ylim)/2, factor, ax=ax)


    def fit_map(self, ax=None):
        if self.rgb is None:
            return
        h, w = self.rgb.shape[:2]
        targets = [ax] if ax is not None else [self.axes_by_key[k] for k in ("satellite", "flood")]
        for target in targets:
            if target is not None:
                target.set_xlim(0, w); target.set_ylim(h, 0)
                target.figure.canvas.draw_idle()


    def zoom_at(self, x, y, factor, ax=None):
        if self.rgb is None:
            return
        ax = ax or self.map_axes
        if ax is None:
            return
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        span_x, span_y = abs(xlim[1]-xlim[0]), abs(ylim[1]-ylim[0])
        new_w = min(max(span_x / factor, 20), self.rgb.shape[1])
        new_h = min(max(span_y / factor, 20), self.rgb.shape[0])
        fx = (x-xlim[0])/(xlim[1]-xlim[0]) if xlim[1] != xlim[0] else .5
        fy = (y-ylim[0])/(ylim[1]-ylim[0]) if ylim[1] != ylim[0] else .5
        left, top = x-fx*new_w, y-fy*new_h
        ax.set_xlim(left, left+new_w); ax.set_ylim(top, top+new_h)
        ax.figure.canvas.draw_idle()


    def on_map_scroll(self, event):
        if event.inaxes not in self.axes or event.xdata is None or event.ydata is None:
            return
        factor = 1.15 if event.button == "up" else 1/1.15
        self.zoom_at(event.xdata, event.ydata, factor, ax=event.inaxes)


    def on_map_press(self, event):
        if event.inaxes not in self.axes or event.button != 1:
            return
        if event.dblclick:
            self.fit_map(ax=event.inaxes)
            self.drag_data = None
            return
        self.map_axes = event.inaxes
        self.drag_data = {"x": event.x, "y": event.y,
            "xlim": event.inaxes.get_xlim(), "ylim": event.inaxes.get_ylim(),
            "ax": event.inaxes, "moved": False}


    def on_map_motion(self, event):
        if self.drag_data is not None or event.inaxes not in self.axes:
            return
        if event.xdata is None or event.ydata is None or self.input_path is None:
            return
        col, row = int(event.xdata), int(event.ydata)
        if self.rgb is None or not (0 <= row < self.rgb.shape[0] and 0 <= col < self.rgb.shape[1]):
            return
        latlon = self.pixel_latlon(row, col)
        if latlon:
            self.status.set(f"Lat: {latlon[0]:.5f}°, Lon: {latlon[1]:.5f}° | Row {row}, Col {col}")


    def on_pan_move(self, event):
        if self.drag_data is None or event.x is None or event.y is None:
            return
        d = self.drag_data
        dxs, dys = event.x-d["x"], event.y-d["y"]
        if abs(dxs)+abs(dys) > 4: d["moved"] = True
        if not d["moved"]: return
        ax = d["ax"]; bbox = ax.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0: return
        dx = -dxs*(d["xlim"][1]-d["xlim"][0])/bbox.width
        dy = dys*(d["ylim"][1]-d["ylim"][0])/bbox.height
        ax.set_xlim(d["xlim"][0]+dx, d["xlim"][1]+dx)
        ax.set_ylim(d["ylim"][0]+dy, d["ylim"][1]+dy)
        ax.figure.canvas.draw_idle()


    def on_map_release(self, event):
        if self.drag_data is None:
            return
        was_drag = self.drag_data["moved"]
        self.drag_data = None
        if was_drag or self.probability is None or event.inaxes not in self.axes:
            return
        if event.xdata is None or event.ydata is None: return
        col, row = int(event.xdata), int(event.ydata)
        h, w = self.probability.shape
        if not (0 <= col < w and 0 <= row < h): return
        if not self.valid_mask[row, col]:
            self.status.set("Selected pixel has no valid data."); return
        score = float(self.probability[row, col])
        label = "Flood-classified" if self.flood_map[row, col] == 1 else "Not classified as flood"
        latlon = self.pixel_latlon(row, col)
        coord = f" | Lat {latlon[0]:.5f}, Lon {latlon[1]:.5f}" if latlon else ""
        self.status.set(f"Row {row}, Col {col} | Model score: {score*100:.1f}% | {label}{coord}")


    def export_png(self):
        if self.rgb is None:
            messagebox.showinfo("No image", "Load an image first."); return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if not path: return
        try:
            keys = ("satellite", "flood") if self.view_mode.get() == "Comparison" else (("satellite",) if self.view_mode.get() == "Satellite" else ("flood",))
            export_fig = Figure(figsize=(12 if len(keys)==2 else 7, 5), dpi=160, facecolor=BG)
            for i, key in enumerate(keys):
                ax = export_fig.add_subplot(1, len(keys), i+1)
                source_ax = self.axes_by_key[key]
                ax.set_xlim(source_ax.get_xlim()); ax.set_ylim(source_ax.get_ylim())
                if key == "satellite":
                    ax.imshow(self.rgb, origin="upper", extent=(0,self.rgb.shape[1],self.rgb.shape[0],0))
                    ax.set_title("Original Satellite Image", color=TEXT)
                elif self.flood_map is None:
                    ax.text(.5,.5,"No flood prediction available",ha="center",va="center",color=TEXT,transform=ax.transAxes)
                    ax.set_title("Model Flood Detection", color=TEXT)
                else:
                    rgba=np.zeros((*self.flood_map.shape,4),dtype=np.float32)
                    rgba[self.flood_map==0]=[.02,.04,.08,1]
                    rgba[self.flood_map==1]=[.02,.42,1,float(self.overlay_opacity.get())]
                    ax.imshow(rgba,origin="upper",extent=(0,self.rgb.shape[1],self.rgb.shape[0],0))
                    ax.set_title("Model Flood Detection", color=TEXT)
                ax.set_xticks([]); ax.set_yticks([])
            export_fig.savefig(path, dpi=160, facecolor=BG, bbox_inches="tight")
            self.status.set("PNG exported successfully.")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))


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
                    "TERRAFLUX Flood Intelligence Report"
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

                writer.writerow(["Non-flooded pixels", valid_pixels - flood_pixels])
                writer.writerow(["Valid pixels", valid_pixels])
                writer.writerow(["Flooded share (%)", round(share, 3)])
                area_km2 = self.calculate_flooded_area_km2()
                writer.writerow(["Flooded area (km²)", round(area_km2, 6) if area_km2 is not None else "Unavailable"])
                with rasterio.open(self.input_path) as src:
                    writer.writerow(["CRS", str(src.crs)])
                    writer.writerow(["Bounds", str(tuple(src.bounds))])
                    writer.writerow(["Pixel resolution", str(src.res)])
                    writer.writerow(["Raster width", src.width])
                    writer.writerow(["Raster height", src.height])
                    writer.writerow(["Band count", src.count])

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



    def reset_all(self):
        if self.busy: return
        self.threshold.set(DEFAULT_THRESHOLD); self.overlay_opacity.set(0.55)
        self.threshold_label.config(text="30%"); self.opacity_label.config(text="55%")
        self.view_mode.set("Comparison"); self.drag_data = None
        self.update_tab_styles()
        self.fit_map()
        if self.probability is not None and self.valid_mask is not None:
            self.refresh_classification()
        else:
            for metric in self.metric_values: metric.config(text="—")
            self.draw_results(reset=True)
        self.status.set("Ready · View and controls reset.")


    def export_geotiff(self):
        if self.flood_map is None or self.input_path is None:
            messagebox.showinfo("No results", "Run flood analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("GeoTIFF", "*.tif")],
            initialfile=f"{Path(self.input_path).stem}_flood_map.tif"
        )
        if not path:
            return
        try:
            with rasterio.open(self.input_path) as src:
                profile = src.profile.copy()
                profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
                with rasterio.open(path, "w", **profile) as dst:
                    dst.write(self.flood_map.astype(np.uint8), 1)
                    dst.set_band_description(1, "Flood classification: 1=flood, 0=not flooded, 255=NoData")
                    dst.update_tags(
                        model="Prithvi 100M",
                        device="CPU",
                        threshold=str(float(self.threshold.get())),
                        class_1="Flood detected",
                        class_0="Not flooded",
                        nodata_value="255",
                    )
            self.status.set("GeoTIFF exported successfully.")
        except Exception as exc:
            messagebox.showerror("GeoTIFF export error", str(exc))

# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = FloodApp(root)
    root.mainloop()
