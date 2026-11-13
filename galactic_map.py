import math
from pathlib import Path
import numpy as np
import moderngl
import glm
from PIL import Image, ImageDraw, ImageFont


class GalacticMap:
    """
    Toggleable face-on Scientific Galactic Map Overlay.
    
    Reproduces the scientific Milky Way map appearance (NASA/JPL-Caltech/R. Hurt style):
    - Concentric radial distance rings (15,000, 30,000, 45,000, 60,000, 75,000 ly)
    - Faint dashed minor rings (7,500, 22,500, 37,500, 52,500, 67,500 ly)
    - Radial galactic longitude rays at 30° intervals (0° to 330°)
    - Distinct Sun reticle marker at R_sun ~ 26,000 ly (~15.6 model units)
    - Astrometrically positioned feature labels:
      Galactic Bar, Long Bar, Near 3 kpc Arm, Far 3 kpc Arm,
      Scutum–Centaurus Arm, Sagittarius Arm, Norma Arm, Perseus Arm,
      Outer Arm, Orion Spur, Sun
    - Distance ring and longitude angle labels
    - Aligned with galaxy coordinate system and rotates in lockstep with the Milky Way.
    """

    def __init__(self, ctx, base_dir):
        self.ctx = ctx
        self.base_dir = Path(base_dir)
        self.enabled = False
        self.alpha = 0.95

        # Galaxy scale: 45.0 model units = 75,000 ly => 1 unit = 1,666.67 ly
        self.LY_PER_UNIT = 1666.6667
        self.SUN_DIST = 26000.0 / self.LY_PER_UNIT  # ~15.6 units

        # Load shaders
        with open(self.base_dir / "shaders" / "map_grid_vert.glsl", encoding="utf-8") as f:
            g_vert = f.read()
        with open(self.base_dir / "shaders" / "map_grid_frag.glsl", encoding="utf-8") as f:
            g_frag = f.read()
        self.grid_program = self.ctx.program(vertex_shader=g_vert, fragment_shader=g_frag)

        with open(self.base_dir / "shaders" / "map_label_vert.glsl", encoding="utf-8") as f:
            l_vert = f.read()
        with open(self.base_dir / "shaders" / "map_label_frag.glsl", encoding="utf-8") as f:
            l_frag = f.read()
        self.label_program = self.ctx.program(vertex_shader=l_vert, fragment_shader=l_frag)

        # Build geometry
        self._build_grid()
        self._build_labels()

    def _build_grid(self):
        """Constructs lines for rings, rays, ticks, and Sun marker."""
        lines = []

        # Color palette: subtle semi-transparent grey-blue
        c_major = (0.38, 0.62, 0.88, 0.38)
        c_minor = (0.32, 0.52, 0.75, 0.18)
        c_ray   = (0.35, 0.58, 0.82, 0.28)
        c_sun   = (1.00, 0.88, 0.25, 0.90)

        # 1. Concentric distance rings
        major_radii_ly = [15000, 30000, 45000, 60000, 75000]
        minor_radii_ly = [7500, 22500, 37500, 52500, 67500]

        SEGMENTS = 180
        # Major solid circles
        for r_ly in major_radii_ly:
            r = r_ly / self.LY_PER_UNIT
            for i in range(SEGMENTS):
                a1 = (i / SEGMENTS) * 2.0 * math.pi
                a2 = ((i + 1) / SEGMENTS) * 2.0 * math.pi
                p1 = (r * math.sin(a1), 0.04, r * math.cos(a1))
                p2 = (r * math.sin(a2), 0.04, r * math.cos(a2))
                lines.extend([*p1, *c_major, *p2, *c_major])

        # Minor dashed circles
        DASH_COUNT = 90
        for r_ly in minor_radii_ly:
            r = r_ly / self.LY_PER_UNIT
            for i in range(DASH_COUNT):
                # Only draw every alternate segment for dashed effect
                if i % 2 == 0:
                    a1 = (i / DASH_COUNT) * 2.0 * math.pi
                    a2 = ((i + 0.55) / DASH_COUNT) * 2.0 * math.pi
                    p1 = (r * math.sin(a1), 0.04, r * math.cos(a1))
                    p2 = (r * math.sin(a2), 0.04, r * math.cos(a2))
                    lines.extend([*p1, *c_minor, *p2, *c_minor])

        # 2. Radial longitude rays (every 30° from 0° to 330°)
        # In Galactocentric coordinates, ray 0° points along +Z through the Sun
        R_INNER = 3.0
        R_OUTER = 45.5
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            # Polar ray: x = r * sin(rad), z = r * cos(rad)
            p1 = (R_INNER * math.sin(rad), 0.04, R_INNER * math.cos(rad))
            p2 = (R_OUTER * math.sin(rad), 0.04, R_OUTER * math.cos(rad))
            lines.extend([*p1, *c_ray, *p2, *c_ray])

            # Small tick marks at major circles
            for r_ly in major_radii_ly:
                r = r_ly / self.LY_PER_UNIT
                tick_len = 0.4
                perp_rad = rad + math.pi / 2.0
                t1 = (r * math.sin(rad) - tick_len * math.sin(perp_rad), 0.04,
                      r * math.cos(rad) - tick_len * math.cos(perp_rad))
                t2 = (r * math.sin(rad) + tick_len * math.sin(perp_rad), 0.04,
                      r * math.cos(rad) + tick_len * math.cos(perp_rad))
                lines.extend([*t1, *c_major, *t2, *c_major])

        # 3. Sun marker reticle (at x=0.0, z=SUN_DIST)
        sun_x = 0.0
        sun_z = self.SUN_DIST
        sun_r = 0.55
        SUN_SEGS = 36
        for i in range(SUN_SEGS):
            a1 = (i / SUN_SEGS) * 2.0 * math.pi
            a2 = ((i + 1) / SUN_SEGS) * 2.0 * math.pi
            p1 = (sun_x + sun_r * math.cos(a1), 0.06, sun_z + sun_r * math.sin(a1))
            p2 = (sun_x + sun_r * math.cos(a2), 0.06, sun_z + sun_r * math.sin(a2))
            lines.extend([*p1, *c_sun, *p2, *c_sun])

        # Sun crosshairs
        cross_len = 0.95
        lines.extend([
            sun_x - cross_len, 0.06, sun_z, *c_sun,
            sun_x + cross_len, 0.06, sun_z, *c_sun,
            sun_x, 0.06, sun_z - cross_len, *c_sun,
            sun_x, 0.06, sun_z + cross_len, *c_sun,
        ])

        grid_data = np.array(lines, dtype="f4")
        self.grid_vbo = self.ctx.buffer(grid_data.tobytes())
        self.grid_vao = self.ctx.vertex_array(
            self.grid_program,
            [(self.grid_vbo, "3f 4f", "in_position", "in_color")]
        )
        self.num_grid_vertices = len(lines) // 7

    def _build_labels(self):
        """Builds texture atlas and billboard quad geometry for labels."""
        # Try loading high-quality TrueType font; fallback gracefully if missing
        font_paths = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
        font = None
        small_font = None
        for fp in font_paths:
            if Path(fp).exists():
                try:
                    font = ImageFont.truetype(fp, 36)
                    small_font = ImageFont.truetype(fp, 26)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
            small_font = font

        # Label definitions: text, anchor (x, y, z), color, scale, is_small
        labels = [
            # Primary Features
            {"text": "GALACTIC BAR", "pos": (-0.6, 0.06, 1.8), "color": (1.0, 0.92, 0.55, 0.92), "scale": 1.20, "small": False},
            {"text": "LONG BAR", "pos": (2.2, 0.06, 4.8), "color": (1.0, 0.88, 0.50, 0.88), "scale": 1.05, "small": False},
            {"text": "Near 3 kpc Arm", "pos": (4.6, 0.06, 3.2), "color": (0.85, 0.90, 0.96, 0.85), "scale": 0.95, "small": True},
            {"text": "Far 3 kpc Arm", "pos": (-4.6, 0.06, -3.2), "color": (0.85, 0.90, 0.96, 0.85), "scale": 0.95, "small": True},
            {"text": "Scutum–Centaurus Arm", "pos": (8.5, 0.06, -14.5), "color": (0.75, 0.88, 1.0, 0.90), "scale": 1.15, "small": False},
            {"text": "Sagittarius Arm", "pos": (13.5, 0.06, 5.5), "color": (0.75, 0.88, 1.0, 0.90), "scale": 1.15, "small": False},
            {"text": "Norma Arm", "pos": (-7.0, 0.06, -14.0), "color": (0.75, 0.88, 1.0, 0.90), "scale": 1.10, "small": False},
            {"text": "Perseus Arm", "pos": (-18.0, 0.06, 8.5), "color": (0.75, 0.88, 1.0, 0.90), "scale": 1.20, "small": False},
            {"text": "Outer Arm", "pos": (-26.0, 0.06, 18.0), "color": (0.70, 0.85, 1.0, 0.85), "scale": 1.05, "small": False},
            {"text": "Orion Spur", "pos": (2.4, 0.06, self.SUN_DIST), "color": (0.95, 0.90, 0.60, 0.90), "scale": 0.95, "small": True},
            {"text": "SUN", "pos": (0.0, 0.06, self.SUN_DIST + 1.2), "color": (1.00, 0.88, 0.25, 0.98), "scale": 1.15, "small": False},
        ]

        # Distance ring labels (along 35° ray)
        ring_labels = [
            (15000, "15,000 ly"),
            (30000, "30,000 ly"),
            (45000, "45,000 ly"),
            (60000, "60,000 ly"),
            (75000, "75,000 ly"),
        ]
        ring_angle = math.radians(35.0)
        for r_ly, txt in ring_labels:
            r = r_ly / self.LY_PER_UNIT
            pos = (r * math.sin(ring_angle) + 0.8, 0.06, r * math.cos(ring_angle))
            labels.append({
                "text": txt,
                "pos": pos,
                "color": (0.65, 0.80, 0.95, 0.75),
                "scale": 0.85,
                "small": True
            })

        # Galactic longitude labels (around outer circle at r = 47.0)
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            r = 47.2
            pos = (r * math.sin(rad), 0.06, r * math.cos(rad))
            labels.append({
                "text": f"{deg}°",
                "pos": pos,
                "color": (0.70, 0.85, 1.00, 0.80),
                "scale": 0.85,
                "small": True
            })

        # Render labels to a high-resolution texture atlas
        ATLAS_W, ATLAS_H = 2048, 1024
        atlas_img = Image.new("RGBA", (ATLAS_W, ATLAS_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(atlas_img)

        quad_vertices = []
        cur_x, cur_y = 10, 10
        row_h = 0

        for item in labels:
            f = small_font if item["small"] else font
            bbox = draw.textbbox((0, 0), item["text"], font=f)
            tw = bbox[2] - bbox[0] + 16
            th = bbox[3] - bbox[1] + 14

            if cur_x + tw >= ATLAS_W - 10:
                cur_x = 10
                cur_y += row_h + 10
                row_h = 0

            # Subtle dark halo / outline for readability over stars
            for ox in [-2, -1, 0, 1, 2]:
                for oy in [-2, -1, 0, 1, 2]:
                    if ox != 0 or oy != 0:
                        draw.text((cur_x + 8 + ox, cur_y + 7 + oy), item["text"], font=f, fill=(5, 8, 15, 230))
            # Text fill (pure white/alpha, colored by vertex attribute)
            draw.text((cur_x + 8, cur_y + 7), item["text"], font=f, fill=(255, 255, 255, 255))

            # UV coordinates in texture atlas
            u0 = cur_x / ATLAS_W
            v0 = cur_y / ATLAS_H
            u1 = (cur_x + tw) / ATLAS_W
            v1 = (cur_y + th) / ATLAS_H

            # Quad size in world coordinates (view-space billboard units)
            base_scale = 0.016 * item["scale"]
            qw = tw * base_scale
            qh = th * base_scale

            ax, ay, az = item["pos"]
            cr, cg, cb, ca = item["color"]

            # Two triangles per label (6 vertices)
            # Offset x, offset y, uv_u, uv_v, anchor_x, anchor_y, anchor_z, r, g, b, a
            v_tl = (-qw * 0.5,  qh * 0.5, u0, v0, ax, ay, az, cr, cg, cb, ca)
            v_tr = ( qw * 0.5,  qh * 0.5, u1, v0, ax, ay, az, cr, cg, cb, ca)
            v_bl = (-qw * 0.5, -qh * 0.5, u0, v1, ax, ay, az, cr, cg, cb, ca)
            v_br = ( qw * 0.5, -qh * 0.5, u1, v1, ax, ay, az, cr, cg, cb, ca)

            quad_vertices.extend([
                *v_tl, *v_bl, *v_tr,
                *v_tr, *v_bl, *v_br
            ])

            cur_x += tw + 12
            row_h = max(row_h, th)

        # Upload texture atlas to ModernGL
        self.atlas_tex = self.ctx.texture((ATLAS_W, ATLAS_H), 4, atlas_img.tobytes())
        self.atlas_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.atlas_tex.build_mipmaps()

        # Upload quad VBO
        label_data = np.array(quad_vertices, dtype="f4")
        self.label_vbo = self.ctx.buffer(label_data.tobytes())
        self.label_vao = self.ctx.vertex_array(
            self.label_program,
            [(self.label_vbo, "2f 2f 3f 4f", "in_offset", "in_uv", "in_anchor", "in_color")]
        )
        self.num_label_vertices = len(quad_vertices) // 11

    def render(self, camera, aspect_ratio, rotation_angle):
        """Renders coordinate grid and billboarded feature labels."""
        if not self.enabled:
            return

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.DEPTH_TEST)

        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect_ratio)
        model_mat = glm.rotate(glm.mat4(1.0), rotation_angle, glm.vec3(0.0, 1.0, 0.0))

        # 1. Render Coordinate Grid Lines
        if "u_model" in self.grid_program:
            self.grid_program["u_model"].write(model_mat)
        if "u_view" in self.grid_program:
            self.grid_program["u_view"].write(view_mat)
        if "u_projection" in self.grid_program:
            self.grid_program["u_projection"].write(proj_mat)
        if "u_alpha" in self.grid_program:
            self.grid_program["u_alpha"].value = self.alpha

        self.grid_vao.render(moderngl.LINES)

        # 2. Render Feature & Coordinate Labels
        self.atlas_tex.use(location=0)
        if "u_atlas" in self.label_program:
            self.label_program["u_atlas"].value = 0
        if "u_model" in self.label_program:
            self.label_program["u_model"].write(model_mat)
        if "u_view" in self.label_program:
            self.label_program["u_view"].write(view_mat)
        if "u_projection" in self.label_program:
            self.label_program["u_projection"].write(proj_mat)
        if "u_alpha" in self.label_program:
            self.label_program["u_alpha"].value = self.alpha

        # Adjust label world scale dynamically with distance for perfect readability
        cam_dist = camera.absolute_distance
        scale_factor = max(0.65, min(2.5, cam_dist / 38.0))
        if "u_scale" in self.label_program:
            self.label_program["u_scale"].value = scale_factor

        self.label_vao.render(moderngl.TRIANGLES)
