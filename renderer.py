from lod import LODManager


class Renderer:

    def __init__(
        self,
        ctx,
        near_program,
        near_vao,
        medium_program,
        medium_vao,
        billboard_program,
        billboard_vao
    ):

        self.ctx = ctx

        # Black hole renderer
        self.near_program = near_program
        self.near_vao = near_vao
        
        self.medium_program = medium_program
        self.medium_vao = medium_vao
        
        self.billboard_program = billboard_program
        self.billboard_vao = billboard_vao



    # ======================================================
    # Main Render
    # ======================================================

    def render(self, lod_state):
        near_w = lod_state["near_weight"]
        med_w = lod_state["medium_weight"]
        far_w = lod_state["far_weight"]

        # --------------------------------------------------
        # NEAR
        # --------------------------------------------------
        if near_w > 0.0:
            if "u_lodWeight" in self.near_program:
                self.near_program["u_lodWeight"].value = near_w
            self.near_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # --------------------------------------------------
        # MEDIUM
        # --------------------------------------------------
        if med_w > 0.0:
            if "u_lodWeight" in self.medium_program:
                self.medium_program["u_lodWeight"].value = med_w
            self.medium_vao.render(mode=self.ctx.TRIANGLE_STRIP)

        # --------------------------------------------------
        # FAR
        # --------------------------------------------------
        if far_w > 0.0:
            if "u_lodWeight" in self.billboard_program:
                self.billboard_program["u_lodWeight"].value = far_w
            self.billboard_vao.render(mode=self.ctx.TRIANGLE_STRIP)