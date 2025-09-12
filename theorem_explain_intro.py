from manim import *
import numpy as np
from moviepy import VideoFileClip


class TheoryExplainShowcase(Scene):
    """Quick intro, then a 2x2 quadrant showcase: Math, Physics, CS, Chemistry.

    Render (quick, low quality):
        manim -pql theorem_explain_intro.py TheoryExplainShowcase
    """

    def make_panel(self, title: str, center: np.ndarray, size=(6.2, 3.6), color=GRAY_E) -> VGroup:
        w, h = size
        rect = RoundedRectangle(corner_radius=0.2, width=w, height=h, stroke_color=color)
        rect.set_fill(color, opacity=0.06)
        rect.move_to(center)
        label = Text(title, weight=BOLD, font_size=28).next_to(rect.get_top(), DOWN, buff=0.2)
        return VGroup(rect, label)

    def construct(self):
        # 1) Quick intro
        self.next_section("Intro")
        title = Text("STEMFUN", weight=BOLD, font_size=52)
        subtitle = Text("Generative Manim-powered for Animated Learning", font_size=30, color=BLUE_D)
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(title, shift=UP), FadeIn(subtitle, shift=DOWN))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # 2) Layout 2x2 panels + separators
        self.next_section("Quadrants")
        # Subtle separators to emphasize 2x2 split
        sep_h = Line(LEFT * 7.0, RIGHT * 7.0, stroke_color=GRAY_D, stroke_opacity=0.35)
        sep_v = Line(UP * 4.0, DOWN * 4.0, stroke_color=GRAY_D, stroke_opacity=0.35)
        separators = VGroup(sep_h, sep_v)
        self.play(Create(separators), run_time=0.6)

        # Frame roughly width ~14, height ~8. Use centers that look balanced.
        UL = np.array([-3.5, 1.9, 0])   # Math
        UR = np.array([3.5, 1.9, 0])    # Physics
        LL = np.array([-3.5, -1.9, 0])  # CS
        LR = np.array([3.5, -1.9, 0])   # Chemistry

        p_math = self.make_panel("Math", UL)
        p_phys = self.make_panel("Physics", UR)
        p_cs = self.make_panel("Computer Science", LL)
        p_chem = self.make_panel("Chemistry", LR)
        grid = VGroup(p_math, p_phys, p_cs, p_chem)
        self.play(LaggedStart(*[Create(p[0]) for p in [p_math, p_phys, p_cs, p_chem]], lag_ratio=0.15))
        self.play(LaggedStart(*[Write(p[1]) for p in [p_math, p_phys, p_cs, p_chem]], lag_ratio=0.15))

        # Helper: attempt to place a small video preview inside a panel.
        # Falls back to symbolic visuals if the file is missing or cannot be read.
        def add_video_preview(panel_rect: RoundedRectangle, path: str, max_scale=0.95):
            try:
                clip = VideoFileClip(path)
            except Exception:
                return None

            try:
                # pick a middle frame as thumbnail
                t_sample = max(0.0, float(clip.duration) * 0.5)
                frame = clip.get_frame(t_sample)
            except Exception:
                try:
                    frame = clip.get_frame(0)
                except Exception:
                    clip.close()
                    return None

            img = ImageMobject(frame)  # RGB ndarray ok
            avail_w = panel_rect.width * max_scale
            avail_h = panel_rect.height * max_scale
            img.set_width(avail_w)
            if img.height > avail_h:
                img.set_height(avail_h)
            img.move_to(panel_rect.get_center())
            # small play icon overlay
            tri = Triangle(color=WHITE, fill_color=WHITE, fill_opacity=0.8).scale(0.15)
            tri.rotate(-PI/2)
            tri.move_to(img.get_center())
            group = VGroup(img, tri)
            self.add(group)
            self.play(FadeIn(group, scale=0.95), run_time=0.4)
            try:
                clip.close()
            except Exception:
                pass
            return group

        # 3) Content per panel (iconic visuals or video previews)
        self.next_section("Math")
        # Math: Euler's identity and a quick transform
        math_vid = add_video_preview(p_math[0], "media/previews/math.mp4")
        euler = None
        gauss = None
        if math_vid is None:
            euler = MathTex(r"e^{i\pi} + 1 = 0").scale(0.9)
            euler.move_to(p_math[0].get_center()).shift(DOWN*0.2)
            gauss = MathTex(r"\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}").scale(0.7)
            gauss.next_to(euler, DOWN, buff=0.3)

        self.next_section("Physics")
        # Physics: E=mc^2 and a small sine-wave polyline
        phys_vid = add_video_preview(p_phys[0], "media/previews/physics.mp4")
        emc2 = None
        wave = None
        orbit_path = None
        nucleus = None
        electron = None
        t_tracker = None
        if phys_vid is None:
            # Keep a bold formula but leave room for an orbit below it
            emc2 = MathTex(r"E = mc^2").scale(0.9)
            emc2.move_to(p_phys[0].get_center()).shift(UP*0.9)

            # Attractive orbital animation
            center_phys = p_phys[0].get_center()
            orbit_path = Ellipse(width=2.8, height=1.8, color=BLUE_D, stroke_opacity=0.5)
            orbit_path.move_to(center_phys + DOWN*0.15)
            nucleus = Dot(center_phys + DOWN*0.15, radius=0.09, color=RED_E)

            t_tracker = ValueTracker(0.0)
            electron = Dot(radius=0.06, color=YELLOW)
            electron.move_to(orbit_path.point_from_proportion(0.0))
            electron.add_updater(lambda m: m.move_to(orbit_path.point_from_proportion(t_tracker.get_value() % 1)))

        self.next_section("Computer Science")
        # Computer Science: small array and highlight (Binary Search)
        cs_vid = add_video_preview(p_cs[0], "media/previews/cs.mp4")
        cs_caption = None
        boxes = None
        highlight = None
        if cs_vid is None:
            # Reposition caption to avoid overlapping the panel label
            cs_caption = Text("Binary Search", font_size=24, color=GREEN_D)
            cs_caption.next_to(p_cs[1], DOWN, buff=0.12)
            boxes = VGroup()
            for i in range(8):
                sq = Square(side_length=0.6).set_stroke(GRAY_C)
                txt = Text(str(i*2), font_size=20)
                cell = VGroup(sq, txt)
                boxes.add(cell)
            boxes.arrange(RIGHT, buff=0.15)
            boxes.move_to(p_cs[0].get_center()).shift(DOWN*0.2)
            mid_idx = 3
            # Highlight the whole cell VGroup to avoid type issues
            if 0 <= mid_idx < len(boxes):
                highlight = SurroundingRectangle(boxes[mid_idx], color=YELLOW, buff=0.05)

        self.next_section("Chemistry")
        # Chemistry: reaction equation and a tiny H2O molecule sketch
        chem_vid = add_video_preview(p_chem[0], "media/previews/chemistry.mp4")
        reaction = None
        o = h1 = h2 = bond1 = bond2 = None
        if chem_vid is None:
            reaction = MathTex(r"2H_2 + O_2 \rightarrow 2H_2O").scale(0.8)
            reaction.move_to(p_chem[0].get_center()).shift(UP*0.3)
            o = Circle(radius=0.16, color=RED_B, fill_color=RED_B, fill_opacity=0.5)
            h1 = Circle(radius=0.1, color=WHITE, fill_color=WHITE, fill_opacity=0.8)
            h2 = Circle(radius=0.1, color=WHITE, fill_color=WHITE, fill_opacity=0.8)
            o_group_center = p_chem[0].get_center() + DOWN*0.4
            o.move_to(o_group_center)
            h1.move_to(o_group_center + LEFT*0.35 + DOWN*0.12)
            h2.move_to(o_group_center + RIGHT*0.35 + DOWN*0.12)
            bond1 = Line(o.get_center(), h1.get_center(), stroke_color=GRAY_C)
            bond2 = Line(o.get_center(), h2.get_center(), stroke_color=GRAY_C)

        # Animate content with gentle staggering
        if euler is not None:
            self.play(Write(euler))
        if gauss is not None:
            self.play(FadeIn(gauss, shift=DOWN*0.2))

        if emc2 is not None:
            self.play(Write(emc2))
        if wave is not None:
            self.play(Create(wave))
        if orbit_path is not None and electron is not None:
            self.play(Create(orbit_path), FadeIn(nucleus), FadeIn(electron))
            self.play(t_tracker.animate.set_value(1.0), run_time=3, rate_func=linear)

        if cs_caption is not None:
            self.play(FadeIn(cs_caption, shift=UP*0.1))
        if boxes is not None:
            self.play(LaggedStart(*[Create(cell[0]) for cell in boxes], lag_ratio=0.05))
            self.play(LaggedStart(*[FadeIn(cell[1], shift=UP*0.05) for cell in boxes], lag_ratio=0.05))
        if highlight is not None:
            self.play(Create(highlight))

        if reaction is not None:
            self.play(Write(reaction))
        if o is not None and h1 is not None and h2 is not None:
            self.play(FadeIn(o), FadeIn(h1), FadeIn(h2))
        if bond1 is not None and bond2 is not None:
            self.play(Create(bond1), Create(bond2))

        # Brief beat to view all
        self.wait(2)

        # Outro: subtle emphasis highlight then fade
        glow = SurroundingRectangle(grid, color=BLUE, buff=0.2)
        glow.set_stroke(width=2)
        self.play(Create(glow))
        self.wait(0.5)
        # Clean fade out of everything
        fade_targets = [grid, separators]
        # Add preview thumbnails if present
        for preview in [math_vid, phys_vid, cs_vid, chem_vid]:
            if preview is not None:
                fade_targets.append(preview)
        # Conditional targets only if they were created
        for m in [euler, gauss, emc2, wave, cs_caption, boxes, highlight, reaction, o, h1, h2, bond1, bond2, orbit_path, nucleus, electron]:
            if m is not None:
                fade_targets.append(m)
        self.play(FadeOut(glow), *[FadeOut(m) for m in fade_targets])
        self.wait(0.3)