import math
from pathlib import Path
import numpy as np
from PIL import Image

def test():
    base_dir = Path("c:/Users/pavan kumar/OneDrive/Desktop/MilkyWay/Milky-Way-Simulation")
    import sys
    sys.path.append(str(base_dir))
    
    from galaxy import GalacticBulge, GalacticBar
    from galactic_disk import GalacticDisk
    from nebula import NebulaManager
    import moderngl
    
    ctx = moderngl.create_context(standalone=True)
    nebula_mgr = NebulaManager(diagnostic_mode=False)
    disk = GalacticDisk(ctx, base_dir, nebula_stars=nebula_mgr.get_stars_array(), diagnostic_mode=False)
    bar = GalacticBar(ctx, base_dir)
    bulge = GalacticBulge(ctx, base_dir)
    
    print("Disk VBO size:", disk.vbo.size)
    print("Bar VBO size:", bar.vbo.size)
    print("Bulge VBO size:", bulge.vbo.size)
    
test()
