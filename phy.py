from manim import *

from manim_physics import *

class ElectricFieldExampleScene(Scene):
    def construct(self):
        # Create multiple charges in different positions
        charge1 = Charge(-2, LEFT * 2 + DOWN * 2)
        charge2 = Charge(3, RIGHT * 2 + DOWN * 2)
        charge3 = Charge(-1, UP * 2)
        charge4 = Charge(1, LEFT * 2 + UP)
        charge5 = Charge(-1.5, RIGHT * 2 + UP)
        charge6 = Charge(2, DOWN * 2)
        charge7 = Charge(-0.5, LEFT * 3)
        charge8 = Charge(1.5, RIGHT * 3)
        
        # Create electric field with higher resolution for more arrows
        field = ElectricField(
            charge1, charge2, charge3, charge4, charge5, charge6, charge7, charge8,
            x_range=[-4, 4, 0.3],  # More dense grid
            y_range=[-4, 4, 0.3],  # More dense grid
        )
        
        # Add all charges and field to scene
        self.add(charge1, charge2, charge3, charge4, charge5, charge6, charge7, charge8)
        self.add(field)



class MagneticFieldExample(ThreeDScene):
    def construct(self):
        wire = Wire(Circle(2).rotate(PI / 2, UP))
        mag_field = MagneticField(
            wire,
            x_range=[-4, 4],
            y_range=[-4, 4],
        )
        self.set_camera_orientation(PI / 3, PI / 4)
        self.add(wire, mag_field)

        



class RadialWaveExampleScene(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(60 * DEGREES, -45 * DEGREES)
        wave = RadialWave(
            LEFT * 2 + DOWN * 5,  # Two source of waves
            RIGHT * 2 + DOWN * 5,
            checkerboard_colors=[BLUE_D],
            stroke_width=0,
        )
        self.add(wave)
