import math
import glm


class Camera:
    """
    Camera controlling the viewing perspective of the Milky Way simulation.
    Supports smooth mouse-driven orbit (inclination + azimuth) with inertia,
    scroll-wheel zoom, and keyboard fallback controls.

    All accessors (get_position, get_target, get_forward, get_right, get_up,
    get_view_matrix) describe the SAME camera.  Basis vectors are cached once
    per frame in update() to avoid redundant trig.
    """

    # -----------------------------------------------------------------------
    # Tuning constants
    # -----------------------------------------------------------------------

    # Mouse sensitivity (radians per pixel)
    MOUSE_SENSITIVITY_X = 0.004   # azimuth (left-right)
    MOUSE_SENSITIVITY_Y = 0.004   # inclination (up-down)

    # How quickly the camera snaps to its target (easing speed)
    SMOOTH_SPEED = 8.0

    # Inertia: fraction of velocity kept each second when mouse is released
    # 1.0 = no damping, 0.0 = instant stop
    INERTIA_DECAY = 0.85          # ~85 % velocity kept per second
    INERTIA_STOP_THRESHOLD = 1e-5 # velocity magnitude below which we stop

    # Zoom
    ZOOM_STEP   = 0.08
    ZOOM_MIN    = 0.25
    ZOOM_MAX    = 10000.0
    ZOOM_SMOOTH = 8.0

    # Keyboard orbit step (radians per call)
    KB_INCLINATION_STEP = math.radians(3)
    KB_AZIMUTH_SPEED    = math.radians(60)  # radians / second

    # Inclination limits
    INCL_MIN = 0.0
    INCL_MAX = math.radians(89.0)

    # Pan sensitivity (world units per pixel)
    PAN_SENSITIVITY = 0.003

    def __init__(self):

        # ------------------------------------------------------------------
        # Orientation state
        # ------------------------------------------------------------------
        # 0° → face-on view, 90° → edge-on view
        self.inclination        = math.radians(75.0)
        self.target_inclination = self.inclination

        self.azimuth        = 0.0
        self.target_azimuth = 0.0

        # ------------------------------------------------------------------
        # Zoom state
        # ------------------------------------------------------------------
        self.distance        = 1.0
        self.target_distance = 1.0

        # ------------------------------------------------------------------
        # Pan state — offset applied to the lookAt target
        # ------------------------------------------------------------------
        self._pan_target = glm.vec3(0.0, 0.0, 0.0)
        self._focus_target = None  # optional glm.vec3 smooth pan easing goal

        # ------------------------------------------------------------------
        # Mouse drag state
        # ------------------------------------------------------------------
        self._dragging       = False   # left button orbit
        self._panning        = False   # middle button pan
        self._last_mouse_x   = 0
        self._last_mouse_y   = 0

        # Angular velocity kept alive by inertia after drag ends
        self._vel_azimuth     = 0.0   # radians / second
        self._vel_inclination = 0.0   # radians / second

        # ------------------------------------------------------------------
        # Derived shader value
        # ------------------------------------------------------------------
        self.disk_squish = math.cos(self.inclination)

        # ------------------------------------------------------------------
        # Cached per-frame basis vectors (set in update())
        # ------------------------------------------------------------------
        self._eye     = glm.vec3(0.0, 1.0, 0.0)
        self._target_pt = glm.vec3(0.0, 0.0, 0.0)
        self._forward = glm.vec3(0.0, 0.0, -1.0)
        self._right   = glm.vec3(1.0, 0.0, 0.0)
        self._up      = glm.vec3(0.0, 1.0, 0.0)
        self._view_matrix = glm.mat4(1.0)

        # Initialise the cache
        self._recompute_basis()

    # ==========================================================================
    # Mouse events  (call these from main.py)
    # ==========================================================================

    def on_mouse_press(self, x: int, y: int, button: int):
        """Begin a drag orbit (left) or pan (middle) when the button is pressed."""
        if button == 1:  # left button — orbit
            self._dragging      = True
            self._last_mouse_x  = x
            self._last_mouse_y  = y
            # Kill inertia when user grabs again
            self._vel_azimuth    = 0.0
            self._vel_inclination = 0.0
        elif button == 4:  # middle button — pan
            self._panning       = True
            self._last_mouse_x  = x
            self._last_mouse_y  = y

    def on_mouse_release(self, x: int, y: int, button: int):
        """End drag; inertia will coast the galaxy naturally."""
        if button == 1:
            self._dragging = False
        elif button == 4:
            self._panning = False

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int):
        """
        Called every frame while any mouse button is held.
        dx / dy are pixel deltas since the last call.
        """
        if self._dragging:
            # Compute angular deltas
            d_azimuth    = -dx * self.MOUSE_SENSITIVITY_X
            d_inclination =  dy * self.MOUSE_SENSITIVITY_Y

            # Apply directly to the *targets* for instant but smooth response
            self.target_azimuth    += d_azimuth
            self.target_inclination = self._clamp_incl(
                self.target_inclination + d_inclination
            )

            # Store per-pixel velocity (converted to per-second in update)
            # We'll accumulate it here; _update_inertia divides by dt once
            self._vel_azimuth    = d_azimuth
            self._vel_inclination = d_inclination

            self._last_mouse_x = x
            self._last_mouse_y = y

        if self._panning:
            # Manual middle-drag pan takes immediate priority over auto-focus easing
            self._focus_target = None
            # Pan in camera-space: move the lookAt target along right/up
            pan_scale = self.PAN_SENSITIVITY * self.distance
            self._pan_target += self._right * (-dx * pan_scale)
            self._pan_target += self._up    * ( dy * pan_scale)

            self._last_mouse_x = x
            self._last_mouse_y = y

    def on_mouse_scroll(self, x: int, y: int, x_offset: float, y_offset: float):
        """Scroll wheel zooms smoothly in/out using logarithmic scale."""
        self.target_distance *= math.exp(-y_offset * 0.2)
        self.target_distance  = max(self.ZOOM_MIN,
                                    min(self.ZOOM_MAX, self.target_distance))

    # ==========================================================================
    # Keyboard controls (unchanged API, kept for completeness)
    # ==========================================================================

    def look_up(self):
        self.target_inclination = self._clamp_incl(
            self.target_inclination - self.KB_INCLINATION_STEP
        )

    def look_down(self):
        self.target_inclination = self._clamp_incl(
            self.target_inclination + self.KB_INCLINATION_STEP
        )

    def rotate_left(self, dt: float):
        self.target_azimuth -= self.KB_AZIMUTH_SPEED * dt

    def rotate_right(self, dt: float):
        self.target_azimuth += self.KB_AZIMUTH_SPEED * dt

    def zoom_in(self, dt: float = 1/60.0):
        self.target_distance = max(self.ZOOM_MIN,
                                   self.target_distance * math.exp(-2.0 * dt))

    def zoom_out(self, dt: float = 1/60.0):
        self.target_distance = min(self.ZOOM_MAX,
                                   self.target_distance * math.exp(2.0 * dt))

    # ==========================================================================
    # Per-frame update
    # ==========================================================================

    def update(self, dt: float):
        """Advance smooth interpolation and inertia. Call once per frame."""

        # ------------------------------------------------------------------
        # Inertia: coast the galaxy when not dragging
        # ------------------------------------------------------------------
        if not self._dragging:
            # Convert last-frame pixel delta → per-second velocity
            if dt > 0:
                vel_scale = 1.0 / dt  # delta was for one frame, scale to /s
            else:
                vel_scale = 0.0

            # Decay velocity exponentially
            decay = self.INERTIA_DECAY ** (dt * 60)  # frame-rate independent
            self._vel_azimuth    *= decay
            self._vel_inclination *= decay

            # Apply inertial movement to targets
            if abs(self._vel_azimuth) > self.INERTIA_STOP_THRESHOLD:
                self.target_azimuth += self._vel_azimuth
            else:
                self._vel_azimuth = 0.0

            if abs(self._vel_inclination) > self.INERTIA_STOP_THRESHOLD:
                self.target_inclination = self._clamp_incl(
                    self.target_inclination + self._vel_inclination
                )
            else:
                self._vel_inclination = 0.0

        # ------------------------------------------------------------------
        # Smooth exponential easing toward targets
        # ------------------------------------------------------------------
        t = min(self.SMOOTH_SPEED * dt, 1.0)

        self.inclination += (self.target_inclination - self.inclination) * t
        self.azimuth     += (self.target_azimuth     - self.azimuth)     * t
        self.distance    += (self.target_distance    - self.distance)     * min(self.ZOOM_SMOOTH * dt, 1.0)

        if self._focus_target is not None:
            diff = self._focus_target - self._pan_target
            self._pan_target += diff * t
            if glm.length(diff) < 1e-4:
                self._pan_target = glm.vec3(self._focus_target)
                if self._focus_target == glm.vec3(0.0, 0.0, 0.0):
                    self._focus_target = None

        # ------------------------------------------------------------------
        # Derived value used by the black-hole shader
        # ------------------------------------------------------------------
        self.disk_squish = max(0.02, math.cos(self.inclination))

        # ------------------------------------------------------------------
        # Recompute cached basis vectors for this frame
        # ------------------------------------------------------------------
        self._recompute_basis()

    # ==========================================================================
    # Internal helpers
    # ==========================================================================

    def _clamp_incl(self, v: float) -> float:
        # Don't clamp completely to 0.0 to avoid gimbal lock with lookAt
        return max(0.001, min(self.INCL_MAX, v))

    def _recompute_basis(self):
        """Compute and cache eye position, target, and basis vectors."""
        # Spherical → Cartesian
        y = self.distance * math.cos(self.inclination)
        r = self.distance * math.sin(self.inclination)
        x = r * math.sin(self.azimuth)
        z = r * math.cos(self.azimuth)

        self._target_pt = glm.vec3(self._pan_target)
        self._eye = glm.vec3(x, y, z) + self._target_pt

        # Up vector
        world_up = glm.vec3(0.0, 1.0, 0.0)
        if abs(math.cos(self.inclination)) > 0.999:
            world_up = glm.vec3(0.0, 0.0, -1.0)

        # Forward, right, up — consistent with lookAt
        self._forward = glm.normalize(self._target_pt - self._eye)
        self._right   = glm.normalize(glm.cross(self._forward, world_up))
        self._up      = glm.cross(self._right, self._forward)

        # View matrix (uses the same eye/target/up)
        self._view_matrix = glm.lookAt(self._eye, self._target_pt, world_up)

    # ==========================================================================
    # Matrix Generation
    # ==========================================================================

    def get_view_matrix(self) -> glm.mat4:
        return self._view_matrix

    def get_projection_matrix(self, aspect_ratio: float) -> glm.mat4:
        fov = math.radians(45.0)
        near = 0.01
        far = 10000.0
        return glm.perspective(fov, aspect_ratio, near, far)

    @property
    def absolute_distance(self) -> float:
        return self.distance

    def get_position(self):
        return (self._eye.x, self._eye.y, self._eye.z)

    def get_target(self):
        return (self._target_pt.x, self._target_pt.y, self._target_pt.z)

    def get_forward(self):
        return (self._forward.x, self._forward.y, self._forward.z)

    def get_right(self):
        return (self._right.x, self._right.y, self._right.z)

    def get_up(self):
        return (self._up.x, self._up.y, self._up.z)

    def set_pan_target_smooth(self, target_pos):
        """Smoothly ease the camera look-at target toward target_pos=(x, y, z)."""
        self._focus_target = glm.vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))

    def clear_pan_target_smooth(self):
        """Stop automatic smooth pan easing without moving the current camera target."""
        self._focus_target = None

    def return_to_center_smooth(self):
        """Smoothly return the camera look-at target back to Sagittarius A* (0, 0, 0)."""
        self._focus_target = glm.vec3(0.0, 0.0, 0.0)

    def reset(self):
        self.target_inclination = math.radians(75.0)
        self.target_azimuth = 0.0
        self.target_distance = 1.0
        self._vel_azimuth = 0.0
        self._vel_inclination = 0.0
        self._pan_target = glm.vec3(0.0, 0.0, 0.0)
        self._focus_target = None