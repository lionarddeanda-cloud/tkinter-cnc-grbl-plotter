from __future__ import annotations

import math
import os
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


class ImageProcessor:
    """Utilities to load and preprocess raster inputs for CNC plotting."""

    @staticmethod
    def load_input(path: str) -> Image.Image:
        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            if fitz is None:
                raise RuntimeError("PyMuPDF no está instalado. Instálalo con: pip install pymupdf")
            doc = fitz.open(path)
            if doc.page_count == 0:
                raise ValueError("El PDF no tiene páginas válidas.")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img

        img = Image.open(path)
        return img.convert("RGB")

    @staticmethod
    def preprocess(
        image: Image.Image,
        rotation_deg: float,
        mirror_x: bool,
        mirror_y: bool,
        threshold: int,
        work_width_mm: float = 200.0,
        sharpness: float = 1.5,
        contrast: float = 1.2,
    ) -> tuple[np.ndarray, Image.Image]:
        """Preprocess image with better edge detection and quality enhancement."""
        img = image.copy()
        
        # Apply contrast and sharpness enhancement before transformations
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)
        
        if rotation_deg:
            img = img.rotate(-rotation_deg, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
        if mirror_x:
            img = ImageOps.mirror(img)
        if mirror_y:
            img = ImageOps.flip(img)

        img = img.convert("L")
        
        # Apply Gaussian blur to reduce noise
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Binary threshold
        binary = img.point(lambda p: 255 if p > threshold else 0)

        # Convert to numpy for OpenCV processing
        arr = np.array(binary, dtype=np.uint8)
        
        # Use OpenCV for better edge detection
        edges = cv2.Canny(arr, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine binary and edges for better line detection
        combined = np.maximum(arr, edges)
        
        # Invert for proper masking (white = draw, black = no draw)
        mask = combined < 128
        processed = Image.fromarray((255 - (mask.astype(np.uint8) * 255)))

        # Resize to fit work width
        width_px, height_px = processed.size
        scale = work_width_mm / max(width_px, 1)
        new_w = max(1, int(width_px * scale))
        new_h = max(1, int(height_px * scale))
        processed = processed.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        final_mask = np.array(processed) < 128

        return final_mask.astype(bool), processed

    @staticmethod
    def raster_to_segments(mask: np.ndarray, min_segment_length: int = 2) -> list[tuple[float, float, float, float]]:
        """Convert a binary mask into horizontal raster segments (line-by-line scanning).
        
        Simulates drawing left-to-right, line-by-line from top to bottom, like a plotter.
        """
        rows, cols = mask.shape
        segments: list[tuple[float, float, float, float]] = []

        for y in range(rows):
            x_start = None
            x_end = None
            
            for x in range(cols):
                if mask[y, x]:  # Active pixel
                    if x_start is None:
                        x_start = x
                    x_end = x
                else:
                    # End of segment
                    if x_start is not None and (x_end - x_start + 1) >= min_segment_length:
                        segments.append((float(x_start), float(y), float(x_end), float(y)))
                    x_start = None
                    x_end = None
            
            # Handle segment that extends to end of line
            if x_start is not None and (x_end - x_start + 1) >= min_segment_length:
                segments.append((float(x_start), float(y), float(x_end), float(y)))

        return segments

    @staticmethod
    def estimate_time(segments: list[tuple[float, float, float, float]], travel_feed: float, draw_feed: float) -> float:
        """Estimate total machining time in minutes."""
        total_distance = 0.0
        
        # Calculate drawing distance
        for x0, y0, x1, y1 in segments:
            total_distance += abs(x1 - x0) + abs(y1 - y0)

        # Approximate travel distance between segments
        if segments:
            px, py = segments[0][0], segments[0][1]
            for x0, y0, x1, y1 in segments[1:]:
                travel_dist = math.hypot(x0 - px, y0 - py)
                total_distance += travel_dist * 1.1  # Add 10% for travel overhead
                px, py = x1, y1

        cut_time = total_distance / max(draw_feed, 1.0)
        return round(cut_time / 60, 2)  # Convert seconds to minutes


class GCodeGenerator:
    @staticmethod
    def generate(
        segments: list[tuple[float, float, float, float]],
        draw_feed: float,
        travel_feed: float,
        z_safe: float,
        z_work: float,
        mm_per_unit: float = 1.0,
        tool_diameter: float = 1.0,
    ) -> str:
        """Generate GRBL-compatible G-Code with optimized tool paths."""
        lines: list[str] = []
        lines.append("; Generated by Tkinter CNC GRBL Plotter (Python 3.12)")
        lines.append(f"; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"; Total segments: {len(segments)}")
        lines.append(f"; Draw Feed: {draw_feed:.1f} mm/min")
        lines.append(f"; Travel Feed: {travel_feed:.1f} mm/min")
        lines.append(f"; Z Safe: {z_safe:.3f} mm")
        lines.append(f"; Z Work: {z_work:.3f} mm")
        lines.append(f"; Tool Diameter: {tool_diameter:.2f} mm")
        lines.append("; ")
        lines.append("G21")  # Millimeters
        lines.append("G90")  # Absolute positioning
        lines.append("G94")  # Feed units per minute
        lines.append("G40")  # Tool radius compensation off
        lines.append("G49")  # Tool length compensation off
        lines.append(f"G0 Z{z_safe:.3f}")  # Move Z to safe height
        lines.append("G0 X0 Y0")  # Move to origin
        lines.append("")

        prev_x, prev_y = 0.0, 0.0
        
        for idx, seg in enumerate(segments):
            x0, y0, x1, y1 = seg
            x0_mm = x0 * mm_per_unit
            y0_mm = y0 * mm_per_unit
            x1_mm = x1 * mm_per_unit
            y1_mm = y1 * mm_per_unit

            # Rapid travel to segment start if we moved far
            travel_dist = math.hypot(x0_mm - prev_x, y0_mm - prev_y)
            if travel_dist > 0.5:
                lines.append(f"G0 X{x0_mm:.3f} Y{y0_mm:.3f} F{travel_feed:.1f}")
                lines.append(f"G0 Z{z_safe:.3f}")
            
            # Lower tool to work depth
            lines.append(f"G1 Z{z_work:.3f} F{draw_feed:.1f}")
            
            # Draw segment
            lines.append(f"G1 X{x1_mm:.3f} Y{y1_mm:.3f} F{draw_feed:.1f}")
            
            # Raise tool
            lines.append(f"G0 Z{z_safe:.3f}")
            
            prev_x, prev_y = x1_mm, y1_mm
            
            # Add newline every 10 segments for readability
            if (idx + 1) % 10 == 0:
                lines.append("")

        lines.append("")
        lines.append("; End of program")
        lines.append("G0 Z{:.3f}".format(z_safe))
        lines.append("G0 X0 Y0")
        lines.append("M5")  # Stop spindle
        lines.append("M30")  # Program end
        lines.append("%")  # End of program marker
        
        return "\n".join(lines) + "\n"


class PlotterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CNC GRBL Plotter - Versión Mejorada")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 750)

        self.input_path: str | None = None
        self.image: Image.Image | None = None
        self.mask: np.ndarray | None = None
        self.processed_preview: Image.Image | None = None
        self.gcode_text = ""
        self.segments: list[tuple[float, float, float, float]] = []

        self.setup_ui()

    def setup_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        main = tk.Frame(self.root, padx=16, pady=16)
        main.pack(fill="both", expand=True)
        main.grid_columnconfigure(1, weight=2)
        main.grid_columnconfigure(2, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # Left panel - Controls
        left = tk.Frame(main, padx=10, pady=10, bg="#f0f0f0")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.grid_columnconfigure(0, weight=1)

        # Center panel - Preview
        preview = tk.LabelFrame(main, text="Vista previa - Raster Scan", padx=10, pady=10, font=("Arial", 10, "bold"))
        preview.grid(row=0, column=1, sticky="nsew")
        preview.grid_columnconfigure(0, weight=1)
        preview.grid_rowconfigure(0, weight=1)

        # Right panel - G-Code Output
        right = tk.LabelFrame(main, text="G-Code Generado", padx=10, pady=10, font=("Arial", 10, "bold"))
        right.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL CONTENT ---
        scroll = tk.Frame(left)
        scroll.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(scroll, bg="#f0f0f0", highlightthickness=0)
        scrollbar = tk.Scrollbar(scroll, orient="vertical", command=canvas.scroll_region)
        scrollable = tk.Frame(canvas, bg="#f0f0f0")
        
        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # File input
        tk.Label(scrollable, text="📁 Archivo de entrada", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w", padx=5, pady=(5, 0))
        self.file_var = tk.StringVar(value="Sin archivo")
        self.file_label = tk.Label(scrollable, textvariable=self.file_var, relief="sunken", anchor="w", bg="white", height=2)
        self.file_label.pack(fill="x", padx=5, pady=(0, 5))

        self.load_btn = tk.Button(scrollable, text="Cargar imagen/PDF", command=self.load_input, bg="#4CAF50", fg="white", font=("Arial", 9, "bold"))
        self.load_btn.pack(fill="x", padx=5, pady=(0, 10))

        # Quality controls
        tk.Label(scrollable, text="🎨 Calidad de imagen", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w", padx=5, pady=(10, 5))
        
        tk.Label(scrollable, text="Contraste", bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w", padx=5)
        self.contrast_var = tk.DoubleVar(value=1.2)
        tk.Scale(scrollable, from_=0.5, to=3.0, orient="horizontal", variable=self.contrast_var, resolution=0.1, bg="#f0f0f0", length=220).pack(fill="x", padx=5)

        tk.Label(scrollable, text="Nitidez", bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w", padx=5, pady=(5, 0))
        self.sharpness_var = tk.DoubleVar(value=1.5)
        tk.Scale(scrollable, from_=0.5, to=3.0, orient="horizontal", variable=self.sharpness_var, resolution=0.1, bg="#f0f0f0", length=220).pack(fill="x", padx=5)

        # Transform controls
        tk.Label(scrollable, text="🔄 Transformaciones", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w", padx=5, pady=(10, 5))
        
        tk.Label(scrollable, text="Rotación (°)", bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w", padx=5)
        self.rotation_var = tk.DoubleVar(value=0.0)
        tk.Scale(scrollable, from_=-180, to=180, orient="horizontal", variable=self.rotation_var, length=220, bg="#f0f0f0").pack(fill="x", padx=5)

        self.mirror_x = tk.BooleanVar(value=False)
        self.mirror_y = tk.BooleanVar(value=False)
        tk.Checkbutton(scrollable, text="Espejo horizontal", variable=self.mirror_x, bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w", padx=5)
        tk.Checkbutton(scrollable, text="Espejo vertical", variable=self.mirror_y, bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w", padx=5, pady=(0, 10))

        # Threshold
        tk.Label(scrollable, text="🎚️ Umbral de binarización", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=(5, 0))
        self.threshold_var = tk.IntVar(value=180)
        self.threshold_label = tk.Label(scrollable, text="180", bg="#f0f0f0", font=("Arial", 9, "bold"))
        self.threshold_label.pack(anchor="w", padx=5)
        
        def update_threshold_label(val):
            self.threshold_label.config(text=str(int(float(val))))
        
        tk.Scale(scrollable, from_=0, to=255, orient="horizontal", variable=self.threshold_var, command=update_threshold_label, length=220, bg="#f0f0f0").pack(fill="x", padx=5, pady=(0, 10))

        # CNC Parameters
        tk.Label(scrollable, text="⚙️ Parámetros CNC", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w", padx=5, pady=(10, 5))

        tk.Label(scrollable, text="Ancho de herramienta (mm)", bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w", padx=5)
        self.tool_width_var = tk.DoubleVar(value=1.0)
        tk.Entry(scrollable, textvariable=self.tool_width_var, width=20).pack(fill="x", padx=5)

        tk.Label(scrollable, text="Feed de dibujo (mm/min)", bg="#f0f0f0", font=("Arial", 9), fg="#d32f2f").pack(anchor="w", padx=5, pady=(10, 0))
        self.draw_feed_var = tk.DoubleVar(value=1200)
        feed_draw_frame = tk.Frame(scrollable, bg="#f0f0f0")
        feed_draw_frame.pack(fill="x", padx=5)
        tk.Entry(feed_draw_frame, textvariable=self.draw_feed_var, width=15).pack(side="left", fill="x", expand=True)
        tk.Button(feed_draw_frame, text="−", width=2, command=lambda: self.adjust_feed("draw", -100)).pack(side="left", padx=(5, 2))
        tk.Button(feed_draw_frame, text="+", width=2, command=lambda: self.adjust_feed("draw", 100)).pack(side="left")

        tk.Label(scrollable, text="Feed de movimiento (mm/min)", bg="#f0f0f0", font=("Arial", 9), fg="#1976d2").pack(anchor="w", padx=5, pady=(10, 0))
        self.travel_feed_var = tk.DoubleVar(value=3000)
        feed_travel_frame = tk.Frame(scrollable, bg="#f0f0f0")
        feed_travel_frame.pack(fill="x", padx=5)
        tk.Entry(feed_travel_frame, textvariable=self.travel_feed_var, width=15).pack(side="left", fill="x", expand=True)
        tk.Button(feed_travel_frame, text="−", width=2, command=lambda: self.adjust_feed("travel", -500)).pack(side="left", padx=(5, 2))
        tk.Button(feed_travel_frame, text="+", width=2, command=lambda: self.adjust_feed("travel", 500)).pack(side="left")

        tk.Label(scrollable, text="Z trabajo (mm)", bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w", padx=5, pady=(10, 0))
        self.z_work_var = tk.DoubleVar(value=-1.5)
        tk.Entry(scrollable, textvariable=self.z_work_var, width=20).pack(fill="x", padx=5)

        tk.Label(scrollable, text="Z seguro (mm)", bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w", padx=5, pady=(5, 0))
        self.z_safe_var = tk.DoubleVar(value=2.0)
        tk.Entry(scrollable, textvariable=self.z_safe_var, width=20).pack(fill="x", padx=5)

        tk.Label(scrollable, text="Ancho útil de trabajo (mm)", bg="#f0f0f0", font=("Arial", 9)).pack(anchor="w", padx=5, pady=(5, 0))
        self.work_width_var = tk.DoubleVar(value=180.0)
        tk.Entry(scrollable, textvariable=self.work_width_var, width=20).pack(fill="x", padx=5, pady=(0, 10))

        # Action buttons
        btn_row = tk.Frame(scrollable, bg="#f0f0f0")
        btn_row.pack(fill="x", padx=5, pady=(10, 0))
        btn_row.grid_columnconfigure((0, 1), weight=1)

        self.preview_btn = tk.Button(btn_row, text="Actualizar vista", command=self.preview_current, bg="#2196F3", fg="white", font=("Arial", 9, "bold"))
        self.preview_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.generate_btn = tk.Button(btn_row, text="Generar G-Code", command=self.generate_gcode, bg="#FF9800", fg="white", font=("Arial", 9, "bold"))
        self.generate_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.export_btn = tk.Button(scrollable, text="📥 Exportar .nc", command=self.export_nc, bg="#4CAF50", fg="white", font=("Arial", 9, "bold"))
        self.export_btn.pack(fill="x", padx=5, pady=(10, 0))

        self.time_var = tk.StringVar(value="⏱ Tiempo estimado: 0.00 min")
        tk.Label(scrollable, textvariable=self.time_var, anchor="w", justify="left", font=("Arial", 9, "bold"), bg="#fff3e0").pack(fill="x", padx=5, pady=(5, 0))

        # --- CENTER PANEL - PREVIEW ---
        self.fig = Figure(figsize=(6.8, 6.8), dpi=100, facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=preview)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- RIGHT PANEL - G-CODE OUTPUT ---
        self.gcode_text_widget = scrolledtext.ScrolledText(right, wrap=tk.WORD, font=("Courier", 8), bg="#1e1e1e", fg="#00ff00", relief="flat")
        self.gcode_text_widget.pack(fill="both", expand=True)
        
        # Configure scrolled text styling
        self.gcode_text_widget.config(insertbackground="#00ff00")

    def adjust_feed(self, feed_type: str, delta: float):
        """Adjust feed rates in real-time."""
        if feed_type == "draw":
            current = self.draw_feed_var.get()
            new_value = max(100, current + delta)
            self.draw_feed_var.set(new_value)
        elif feed_type == "travel":
            current = self.travel_feed_var.get()
            new_value = max(500, current + delta)
            self.travel_feed_var.set(new_value)

    def load_input(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen o PDF",
            filetypes=[
                ("Imagen o PDF", "*.png *.jpg *.jpeg *.bmp *.pdf"),
                ("PNG", "*.png"),
                ("JPG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("PDF", "*.pdf"),
            ],
        )
        if not path:
            return

        self.input_path = path
        self.file_var.set(f"✓ {os.path.basename(path)}")

        try:
            self.image = ImageProcessor.load_input(path)
            self.preview_current()
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Error al cargar", str(exc))

    def preview_current(self):
        if self.image is None:
            messagebox.showwarning("Sin imagen", "Primero carga un archivo.")
            return

        try:
            self.mask, self.processed_preview = ImageProcessor.preprocess(
                self.image,
                rotation_deg=self.rotation_var.get(),
                mirror_x=self.mirror_x.get(),
                mirror_y=self.mirror_y.get(),
                threshold=self.threshold_var.get(),
                work_width_mm=self.work_width_var.get(),
                sharpness=self.sharpness_var.get(),
                contrast=self.contrast_var.get(),
            )
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Error en preprocesado", str(exc))
            return

        # Update preview with grid overlay
        self.ax.clear()
        self.ax.imshow(self.processed_preview, cmap="gray", origin="upper")
        
        # Add grid for scale reference
        height, width = self.processed_preview.size[::-1]
        step = max(height, width) // 10
        if step > 0:
            for i in range(0, width, step):
                self.ax.axvline(i, color="blue", alpha=0.1, linewidth=0.5)
            for i in range(0, height, step):
                self.ax.axhline(i, color="blue", alpha=0.1, linewidth=0.5)
        
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()
        self.fig.tight_layout(pad=1)
        self.canvas.draw()

    def generate_gcode(self):
        if self.mask is None:
            self.preview_current()
        if self.mask is None:
            messagebox.showwarning("No hay vista", "No hay una imagen procesada para generar G-Code.")
            return

        self.segments = ImageProcessor.raster_to_segments(self.mask)
        if not self.segments:
            messagebox.showwarning("Sin trazado", "No se detectaron rutas en la imagen.")
            return

        tool_width = self.tool_width_var.get()
        draw_feed = self.draw_feed_var.get()
        travel_feed = self.travel_feed_var.get()
        z_safe = self.z_safe_var.get()
        z_work = self.z_work_var.get()
        work_width = self.work_width_var.get()

        mm_per_unit = work_width / max(self.mask.shape[1], 1)

        gcode = GCodeGenerator.generate(
            segments=self.segments,
            draw_feed=draw_feed,
            travel_feed=travel_feed,
            z_safe=z_safe,
            z_work=z_work,
            mm_per_unit=mm_per_unit,
            tool_diameter=tool_width,
        )
        self.gcode_text = gcode
        estimated = ImageProcessor.estimate_time(self.segments, travel_feed, draw_feed)
        self.time_var.set(f"⏱ Tiempo estimado: {estimated:.2f} min")

        # Display G-Code in text widget
        self.gcode_text_widget.config(state=tk.NORMAL)
        self.gcode_text_widget.delete(1.0, tk.END)
        self.gcode_text_widget.insert(tk.END, gcode)
        self.gcode_text_widget.config(state=tk.DISABLED)

        messagebox.showinfo("G-Code generado", f"✓ Se generó el código para {len(self.segments)} segmentos.\n\nTiempo estimado: {estimated:.2f} min")

    def export_nc(self):
        if not self.gcode_text:
            self.generate_gcode()
        if not self.gcode_text:
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".nc",
            filetypes=[(("Archivos G-Code", "*.nc"), ("Todos los archivos", "*.*"))],
            initialfile="cnc_plot.nc",
        )
        if not output_path:
            return

        try:
            Path(output_path).write_text(self.gcode_text, encoding="utf-8")
            messagebox.showinfo("Exportación completada", f"✓ Se guardó el archivo:\n{output_path}")
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Error al exportar", str(exc))


def main():
    root = tk.Tk()
    app = PlotterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
