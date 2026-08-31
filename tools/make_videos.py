#!/usr/bin/env python3
"""Animated module-overview videos: Pillow frames + Kokoro narration -> MP4.

One video per week, ~4 scenes each, drawn in the site's palette. Scenes are
parametric primitives (curves, scatters, grids, trees, attention arcs) animated
across the exact duration of their narration line.

Run with the _/tts venv python (kokoro, soundfile, numpy, pillow):

    /Users/john/Dropbox/_/tts/venv/bin/python make_videos.py           # all 13
    /Users/john/Dropbox/_/tts/venv/bin/python make_videos.py week_02   # one
"""
import math, os, random, subprocess, sys
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = "/tmp/ds207-anim"
W, H, FPS = 1280, 720, 15
os.makedirs(WORK, exist_ok=True)
os.makedirs(f"{ROOT}/video", exist_ok=True)

PAPER = (250, 250, 250); CARD = (255, 255, 255); INK = (44, 44, 44)
MUT = (102, 102, 102); NAVY = (26, 54, 93); BLUE = (43, 108, 176)
BORD = (226, 226, 226); GREEN = (39, 103, 73); RED = (155, 44, 44)
SKYBG = (219, 234, 247)

HN = "/System/Library/Fonts/HelveticaNeue.ttc"
def F(size, bold=False, med=False):
    return ImageFont.truetype(HN, size, index=1 if bold else (10 if med else 0))

def ease(t): return t * t * (3 - 2 * t)
def clamp01(x): return max(0.0, min(1.0, x))
def sub(t, a, b): return clamp01((t - a) / (b - a)) if b > a else 1.0
def mix(c1, c2, a): return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))

def frame_base(num, total, wk):
    im = Image.new("RGB", (W, H), PAPER)
    dr = ImageDraw.Draw(im)
    dr.text((80, 26), "DATASCI 207", font=F(22, bold=True), fill=NAVY)
    dr.text((80 + 160, 30), "Applied Machine Learning", font=F(16), fill=MUT)
    dr.text((W - 80, 28), f"Module {wk} · {num}/{total}", font=F(16), fill=MUT, anchor="ra")
    dr.rectangle([0, H - 8, W, H], fill=NAVY)
    return im, dr

def kick(dr, text, y=76):
    dr.text((80, y), text.upper(), font=F(17, bold=True), fill=BLUE)

def type_text(dr, xy, text, t, font, fill, cursor=True):
    n = int(round(len(text) * clamp01(t)))
    shown = text[:n]
    dr.text(xy, shown, font=font, fill=fill)
    if cursor and 0 < t < 1:
        cw = dr.textlength(shown, font=font)
        dr.rectangle([xy[0] + cw + 3, xy[1] + 4, xy[0] + cw + 6,
                      xy[1] + font.size - 2], fill=BLUE)

def axes(dr, x0, y0, w, h, xlabel="", ylabel=""):
    dr.rectangle([x0, y0, x0 + w, y0 + h], outline=BORD, width=2, fill=CARD)
    dr.line([x0, y0 + h, x0 + w, y0 + h], fill=MUT, width=3)
    dr.line([x0, y0, x0, y0 + h], fill=MUT, width=3)
    if xlabel: dr.text((x0 + w / 2, y0 + h + 10), xlabel, font=F(16), fill=MUT, anchor="ma")
    if ylabel: dr.text((x0 - 12, y0 - 26), ylabel, font=F(16), fill=MUT)

def plot(dr, x0, y0, w, h, fn, t, color, width=5):
    """Draw fn:[0,1]->[0,1] progressively to fraction t of x-range."""
    n = max(2, int(120 * clamp01(t)))
    pts = []
    for i in range(n):
        x = i / 119
        y = clamp01(fn(x))
        pts.append((x0 + x * w, y0 + h - y * h))
    if len(pts) > 1:
        dr.line(pts, fill=color, width=width, joint="curve")

# ------------------------------------------------------------ scene factories
def s_title(kicker, big, sub_line):
    def draw(dr, t, rng):
        kick(dr, kicker)
        a = ease(sub(t, 0.03, 0.25))
        dr.text((80, 140), big, font=F(60, bold=True), fill=mix(PAPER, NAVY, a))
        type_text(dr, (80, 270), sub_line, sub(t, 0.32, 0.8), F(34), INK)
    return draw

def s_bullets(kicker, title, lines, closing=False):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 120), title, font=F(44, bold=True), fill=NAVY)
        yy = 230
        for i, ln in enumerate(lines):
            at = 0.15 + i * (0.7 / max(1, len(lines)))
            if t > at:
                a = ease(sub(t, at, at + 0.12))
                dr.ellipse([84, yy + 14, 98, yy + 28], fill=mix(PAPER, BLUE, a))
                dr.text((116, yy), ln, font=F(30, med=closing), fill=mix(PAPER, INK, a))
            yy += 74
    return draw

def s_descent(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        x0, y0, w, h = 160, 200, 620, 380
        axes(dr, x0, y0, w, h, "parameter value", "loss")
        bowl = lambda x: 0.9 * (2.1 * (x - 0.55)) ** 2 + 0.06
        plot(dr, x0, y0, w, h, bowl, sub(t, 0.05, 0.3), BLUE)
        if t > 0.32:
            xs, x = [], 0.06
            for _ in range(9):
                xs.append(x); x = x - 0.08 * 2 * 2.1 * 2.1 * (x - 0.55)
            k = sub(t, 0.35, 0.92) * (len(xs) - 1)
            i = int(k); frac = k - i
            xa = xs[min(i, len(xs)-1)]; xb = xs[min(i+1, len(xs)-1)]
            bx = xa + (xb - xa) * frac
            by = bowl(bx)
            for j in range(min(i + 1, len(xs))):
                px, py = xs[j], bowl(xs[j])
                dr.ellipse([x0+px*w-5, y0+h-py*h-5, x0+px*w+5, y0+h-py*h+5], fill=BORD)
            cx, cy = x0 + bx * w, y0 + h - by * h
            dr.ellipse([cx-13, cy-13, cx+13, cy+13], fill=RED)
            dr.text((900, 300), "step against the gradient,", font=F(26), fill=INK)
            dr.text((900, 340), "over and over", font=F(26), fill=INK)
        if t > 0.88:
            dr.text((900, 420), "that is all training is", font=F(28, bold=True), fill=NAVY)
    return draw

def s_scatter_fit(kicker, title, wiggle=False):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        x0, y0, w, h = 160, 200, 620, 380
        axes(dr, x0, y0, w, h, "input x", "output y")
        r = random.Random(11)
        pts = [(x, clamp01(0.15 + 0.62 * x + r.uniform(-0.09, 0.09)))
               for x in [r.uniform(0.05, 0.95) for _ in range(16)]]
        for px, py in pts:
            dr.ellipse([x0+px*w-6, y0+h-py*h-6, x0+px*w+6, y0+h-py*h+6], fill=NAVY)
        a = ease(sub(t, 0.25, 0.7))
        w0, b0, w1, b1 = -0.3, 0.7, 0.62, 0.15
        wt, bt = w0 + (w1 - w0) * a, b0 + (b1 - b0) * a
        plot(dr, x0, y0, w, h, lambda x: wt * x + bt, 1.0, BLUE)
        if wiggle and t > 0.72:
            aw = ease(sub(t, 0.72, 0.95))
            plot(dr, x0, y0, w, h,
                 lambda x: clamp01(0.62*x + 0.15 + aw*0.13*math.sin(x*31)), 1.0, RED, 4)
            dr.text((900, 300), "the red model hits every", font=F(25), fill=INK)
            dr.text((900, 338), "training point — and loses", font=F(25), fill=INK)
            dr.text((900, 376), "on the next dataset", font=F(25), fill=INK)
            dr.text((900, 440), "fit ≠ memorize", font=F(30, bold=True), fill=RED)
        elif t > 0.5:
            dr.text((900, 320), "learning = sliding the line", font=F(25), fill=INK)
            dr.text((900, 358), "until the errors are small", font=F(25), fill=INK)
    return draw

def s_sigmoid(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        x0, y0, w, h = 160, 200, 620, 380
        axes(dr, x0, y0, w, h, "linear score wx+b", "probability")
        sig = lambda x: 1 / (1 + math.exp(-(x - 0.5) * 11))
        plot(dr, x0, y0, w, h, sig, sub(t, 0.08, 0.45), BLUE)
        if t > 0.5:
            thr = 0.5 + 0.16 * math.sin(sub(t, 0.5, 1.0) * math.tau)
            ty = y0 + h - thr * h
            dr.line([x0, ty, x0 + w, ty], fill=RED, width=4)
            dr.text((x0 + w - 16, ty - 30), f"threshold {thr:.2f}", font=F(20, med=True),
                    fill=RED, anchor="ra")
            dr.text((900, 320), "the curve is the model;", font=F(25), fill=INK)
            dr.text((900, 358), "the threshold is a choice", font=F(25), fill=INK)
        if t > 0.85:
            dr.text((900, 430), "you own the choice", font=F(28, bold=True), fill=NAVY)
    return draw

def s_matrix(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        cx, cy, cs = 260, 240, 150
        labels = [["TP  90", "FN  10"], ["FP  240", "TN  660"]]
        cols = [[GREEN, RED], [RED, mix(PAPER, MUT, 0.5)]]
        dr.text((cx + cs, cy - 44), "predicted +      predicted −", font=F(20, med=True), fill=MUT, anchor="ma")
        dr.text((cx - 24, cy + cs - 10), "actually +", font=F(20, med=True), fill=MUT, anchor="ra")
        dr.text((cx - 24, cy + cs + cs - 10), "actually −", font=F(20, med=True), fill=MUT, anchor="ra")
        order = [(0,0),(0,1),(1,0),(1,1)]
        for k,(i,j) in enumerate(order):
            at = 0.1 + k * 0.12
            if t > at:
                a = ease(sub(t, at, at + 0.1))
                x, y = cx + j * cs, cy + i * cs
                dr.rectangle([x, y, x + cs - 8, y + cs - 8],
                             fill=mix(CARD, mix(cols[i][j], CARD, 0.72), a), outline=BORD, width=2)
                dr.text((x + (cs-8)/2, y + (cs-8)/2 - 14), labels[i][j],
                        font=F(26, bold=True), fill=mix(PAPER, INK, a), anchor="ma")
        if t > 0.62:
            dr.text((760, 250), "accuracy: 75%  — sounds fine", font=F(26), fill=INK)
        if t > 0.74:
            dr.text((760, 300), "precision: 90/330 = 27%", font=F(26, med=True), fill=RED)
        if t > 0.86:
            dr.text((760, 350), "the matrix shows which way", font=F(25), fill=INK)
            dr.text((760, 388), "the model is wrong", font=F(25), fill=INK)
    return draw

def s_xor(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        x0, y0, s = 300, 210, 340
        dr.rectangle([x0, y0, x0 + s, y0 + s], outline=BORD, width=2, fill=CARD)
        pts = [(0.18, 0.18, False), (0.82, 0.82, False), (0.18, 0.82, True), (0.82, 0.18, True)]
        for px, py, pos in pts:
            cx, cy = x0 + px * s, y0 + s - py * s
            col = GREEN if pos else RED
            dr.ellipse([cx-16, cy-16, cx+16, cy+16], outline=col, width=6,
                       fill=CARD if not pos else mix(CARD, col, 0.25))
        if t < 0.55:
            ang = sub(t, 0.05, 0.55) * math.pi * 1.5
            cx, cy = x0 + s/2, y0 + s/2
            dx, dy = math.cos(ang) * s * 0.8, math.sin(ang) * s * 0.8
            dr.line([cx - dx, cy - dy, cx + dx, cy + dy], fill=MUT, width=5)
            dr.text((760, 300), "no single line separates", font=F(26), fill=INK)
            dr.text((760, 338), "the greens from the reds", font=F(26), fill=INK)
        else:
            a = ease(sub(t, 0.55, 0.75))
            # two lines parallel to the anti-diagonal: x+y=0.55 and x+y=1.45
            for c in (0.55, 1.45):
                x1, y1 = max(0.0, c - 1.0), min(1.0, c)      # in unit square
                x2, y2 = min(1.0, c), max(0.0, c - 1.0)
                X1, Y1 = x0 + x1 * s, y0 + s - y1 * s
                X2, Y2 = x0 + x2 * s, y0 + s - y2 * s
                dr.line([X1, Y1, X1 + (X2 - X1) * a, Y1 + (Y2 - Y1) * a], fill=BLUE, width=6)
            dr.text((760, 300), "two lines from one hidden", font=F(26), fill=INK)
            dr.text((760, 338), "layer isolate the stripe", font=F(26), fill=INK)
            if t > 0.8:
                dr.text((760, 400), "that bend is deep learning", font=F(27, bold=True), fill=NAVY)
    return draw

def s_tree(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        def node(x, y, txt, at, leaf=False, good=True):
            if t <= at: return
            a = ease(sub(t, at, at + 0.12))
            wd, ht = 220, 54
            col = (GREEN if good else RED) if leaf else NAVY
            dr.rounded_rectangle([x - wd/2, y, x + wd/2, y + ht], 6,
                                 outline=mix(PAPER, col, a), width=3,
                                 fill=mix(PAPER, CARD, a))
            dr.text((x, y + 14), txt, font=F(20, med=True), fill=mix(PAPER, col, a), anchor="ma")
        def edge(x1, y1, x2, y2, at):
            if t > at:
                a = ease(sub(t, at, at + 0.1))
                dr.line([x1, y1, x1 + (x2-x1)*a, y1 + (y2-y1)*a], fill=BORD, width=4)
        rootx, ry = 560, 210
        edge(rootx, ry+54, 340, 330, 0.3); edge(rootx, ry+54, 780, 330, 0.3)
        edge(340, 384, 230, 470, 0.55); edge(340, 384, 450, 470, 0.55)
        edge(780, 384, 670, 470, 0.55); edge(780, 384, 890, 470, 0.55)
        node(rootx, ry, "hours studied > 4 ?", 0.1)
        node(340, 330, "slept > 7h ?", 0.4); node(780, 330, "did exercises ?", 0.4)
        node(230, 470, "pass", 0.65, True, True); node(450, 470, "fail", 0.65, True, False)
        node(670, 470, "pass", 0.65, True, True); node(890, 470, "fail", 0.65, True, False)
        if t > 0.8:
            dr.text((80, 560), "each split chosen for maximum information gain — a model you can read aloud",
                    font=F(24), fill=INK)
    return draw

def s_clusters(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        x0, y0, w, h = 200, 200, 560, 400
        axes(dr, x0, y0, w, h)
        r = random.Random(5)
        centers = [(0.25, 0.3), (0.7, 0.7), (0.75, 0.22)]
        cols = [BLUE, GREEN, RED]
        pts = []
        for ci, (cx, cy) in enumerate(centers):
            for _ in range(14):
                pts.append((clamp01(cx + r.gauss(0, 0.09)), clamp01(cy + r.gauss(0, 0.09))))
        starts = [(0.5, 0.85), (0.1, 0.6), (0.5, 0.05)]
        k = ease(sub(t, 0.2, 0.85))
        cents = [(sx + (cx - sx) * k, sy + (cy - sy) * k)
                 for (sx, sy), (cx, cy) in zip(starts, centers)]
        for px, py in pts:
            d = [((px-cx)**2 + (py-cy)**2, i) for i, (cx, cy) in enumerate(cents)]
            ci = min(d)[1]
            col = cols[ci] if t > 0.2 else MUT
            X, Y = x0 + px * w, y0 + h - py * h
            dr.ellipse([X-6, Y-6, X+6, Y+6], fill=mix(CARD, col, 0.75))
        for i, (cx, cy) in enumerate(cents):
            X, Y = x0 + cx * w, y0 + h - cy * h
            dr.rectangle([X-11, Y-11, X+11, Y+11], fill=cols[i], outline=INK, width=2)
        dr.text((820, 300), "assign to nearest centroid,", font=F(25), fill=INK)
        dr.text((820, 338), "move centroid to the mean,", font=F(25), fill=INK)
        dr.text((820, 376), "repeat until still", font=F(25), fill=INK)
        if t > 0.88:
            dr.text((820, 440), "you chose k = 3", font=F(27, bold=True), fill=RED)
    return draw

def s_conv(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        g, cell = 8, 44
        gx, gy = 180, 210
        r = random.Random(9)
        vals = [[r.random() for _ in range(g)] for _ in range(g)]
        for i in range(g):
            for j in range(g):
                v = vals[i][j]
                dr.rectangle([gx + j*cell, gy + i*cell, gx + (j+1)*cell - 3, gy + (i+1)*cell - 3],
                             fill=mix(CARD, NAVY, v * 0.55))
        steps = (g - 2) * (g - 2)
        k = int(sub(t, 0.1, 0.9) * (steps - 1))
        ki, kj = divmod(k, g - 2)
        dr.rectangle([gx + kj*cell, gy + ki*cell, gx + (kj+3)*cell - 3, gy + (ki+3)*cell - 3],
                     outline=RED, width=5)
        ox, oy, oc = 700, 240, 40
        for idx in range(k + 1):
            oi, oj = divmod(idx, g - 2)
            s9 = sum(vals[oi+a][oj+b] for a in range(3) for b in range(3)) / 9
            dr.rectangle([ox + oj*oc, oy + oi*oc, ox + (oj+1)*oc - 3, oy + (oi+1)*oc - 3],
                         fill=mix(CARD, BLUE, s9 * 0.8))
        dr.text((gx, gy - 34), "image (8×8)", font=F(19, med=True), fill=MUT)
        dr.text((ox, oy - 34), "feature map (6×6)", font=F(19, med=True), fill=MUT)
        dr.text((700, 530), "nine shared weights scan every position —", font=F(24), fill=INK)
        dr.text((700, 566), "a pattern learned once is found anywhere", font=F(24), fill=INK)
    return draw

def s_vectors(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        cx, cy = 420, 480
        def arrow(ang_deg, ln, col, label, at):
            if t <= at: return
            a = ease(sub(t, at, at + 0.15))
            ang = math.radians(ang_deg)
            ex, ey = cx + math.cos(ang) * ln * a, cy - math.sin(ang) * ln * a
            dr.line([cx, cy, ex, ey], fill=col, width=6)
            dr.ellipse([ex-6, ey-6, ex+6, ey+6], fill=col)
            if a > 0.9:
                dr.text((ex + 12, ey - 12), label, font=F(23, med=True), fill=col)
        arrow(62, 300, BLUE, '"excellent"', 0.08)
        arrow(50, 285, GREEN, '"outstanding"', 0.22)
        arrow(8, 300, RED, '"terrible"', 0.38)
        if t > 0.55:
            dr.arc([cx-90, cy-90, cx+90, cy+90], -62, -50, fill=GREEN, width=5)
            dr.text((820, 240), "small angle:", font=F(24), fill=INK)
            dr.text((1000, 240), "cos ≈ 0.98", font=F(24, bold=True), fill=GREEN)
        if t > 0.7:
            dr.arc([cx-140, cy-140, cx+140, cy+140], -62, -8, fill=RED, width=5)
            dr.text((820, 286), "wide angle:", font=F(24), fill=INK)
            dr.text((1000, 286), "cos ≈ 0.57", font=F(24, bold=True), fill=RED)
        if t > 0.82:
            dr.text((820, 360), "meaning becomes geometry:", font=F(25), fill=INK)
            dr.text((820, 398), "similar words, nearby vectors", font=F(25), fill=INK)
    return draw

def s_fairbars(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        dr.text((80, 190), "one model · one accuracy number · qualified-candidate success rate by group:",
                font=F(24), fill=INK)
        mitig = t > 0.62
        vals = [("group A", 0.71, BLUE), ("group B", 0.44 if not mitig else 0.70, RED if not mitig else BLUE)]
        y = 270
        for name, v, col in vals:
            a = ease(sub(t, 0.15, 0.4))
            dr.text((200, y + 6), name, font=F(24, med=True), fill=INK, anchor="ra")
            dr.rectangle([220, y, 220 + 620, y + 44], outline=BORD, width=2, fill=CARD)
            dr.rectangle([220, y, 220 + 620 * v * a, y + 44], fill=mix(CARD, col, 0.75))
            dr.text([220 + 620 * v * a + 12, y + 8], f"{int(v*100)}%", font=F(22, med=True), fill=MUT)
            y += 90
        if not mitig and t > 0.45:
            dr.text((80, 470), "equally accurate overall — and not the same odds for equal candidates",
                    font=F(25, med=True), fill=RED)
        if mitig:
            dr.text((80, 470), "per-group thresholds can equalize this metric …", font=F(25, med=True), fill=GREEN)
        if t > 0.85:
            dr.text((80, 516), "… and arithmetic guarantees another fairness metric moved. Choosing is a values call.",
                    font=F(24), fill=INK)
    return draw

def s_gmm(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        x0, y0, w, h = 170, 210, 520, 340
        axes(dr, x0, y0, w, h)
        def bell(c, s_, amp):
            return lambda u: amp * math.exp(-0.5 * ((u - c) / s_) ** 2)
        fa, fb = bell(0.32, 0.10, 0.85), bell(0.62, 0.14, 0.60)
        if t > 0.08:
            plot(dr, x0, y0, w, h, fa, sub(t, 0.08, 0.32), BLUE)
        if t > 0.28:
            plot(dr, x0, y0, w, h, fb, sub(t, 0.28, 0.52), RED)
        if t > 0.58:
            u = 0.47
            X = x0 + u * w
            Y = y0 + h - 0.02 * h
            a = ease(sub(t, 0.58, 0.7))
            dr.ellipse([X-9, Y-9, X+9, Y+9], fill=mix(CARD, NAVY, a), outline=INK, width=2)
            da, db = fa(u), fb(u)
            pa = da / (da + db)
            dr.text((X - 60, Y - 56), "p = %.2f / %.2f" % (pa, 1 - pa),
                    font=F(22, med=True), fill=mix(PAPER, INK, a))
        steps = [("E-step: soft-assign every point", 0.36),
                 ("M-step: re-estimate each normal", 0.5),
                 ("repeat until nothing moves", 0.64)]
        for i, (txt, at) in enumerate(steps):
            if t > at:
                dr.text((760, 280 + i * 44), txt, font=F(25),
                        fill=mix(PAPER, INK, ease(sub(t, at, at + 0.1))))
        if t > 0.84:
            dr.text((760, 440), "k-means minus the certainty", font=F(27, bold=True), fill=NAVY)
    return draw

def s_wiring(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        ins = [("age", "raw number"), ("fare", "raw number"),
               ("sex", "one-hot"), ("class", "embed 3 x 4")]
        ys = [200, 290, 380, 470]
        for i, (nm, enc) in enumerate(ins):
            at = 0.04 + i * 0.06
            if t <= at:
                continue
            a = ease(sub(t, at, at + 0.1))
            y = ys[i]
            dr.rounded_rectangle([90, y, 240, y + 52], 8,
                                 fill=mix(PAPER, CARD, a), outline=mix(PAPER, NAVY, a), width=3)
            dr.text((108, y + 14), nm, font=F(22, med=True), fill=mix(PAPER, INK, a))
            dr.rounded_rectangle([272, y + 3, 480, y + 49], 8,
                                 fill=mix(PAPER, CARD, a), outline=mix(PAPER, BORD, a), width=2)
            dr.text((288, y + 14), enc, font=F(20), fill=mix(PAPER, MUT, a))
        if t > 0.34:
            a = ease(sub(t, 0.34, 0.48))
            for i in range(4):
                y = ys[i] + 26
                ty = 320 + i * 27
                dr.line([480, y, 480 + (556 - 480) * a, y + (ty - y) * a], fill=BORD, width=4)
        if t > 0.44:
            a = ease(sub(t, 0.44, 0.54))
            dr.rectangle([556, 290, 584, 460], fill=mix(PAPER, BLUE, a * 0.5),
                         outline=mix(PAPER, BLUE, a), width=3)
            dr.text((526, 252), "concat", font=F(20, med=True), fill=mix(PAPER, NAVY, a))
        if t > 0.54:
            a = ease(sub(t, 0.54, 0.66))
            dr.line([584, 375, 584 + 56 * a, 375], fill=BORD, width=4)
            dr.rounded_rectangle([640, 345, 852, 405], 8, fill=mix(PAPER, CARD, a),
                                 outline=mix(PAPER, NAVY, a), width=3)
            dr.text((658, 362), "Dense(16) relu", font=F(22, med=True), fill=mix(PAPER, INK, a))
            dr.text((640, 415), "(width + 1) x 16 params", font=F(18), fill=mix(PAPER, MUT, a))
        if t > 0.66:
            a = ease(sub(t, 0.66, 0.76))
            dr.line([852, 365, 852 + 46 * a, 365 - 60 * a], fill=BORD, width=4)
            dr.rounded_rectangle([900, 270, 1076, 322], 8, fill=mix(PAPER, CARD, a),
                                 outline=mix(PAPER, GREEN, a), width=3)
            dr.text((918, 284), "survival", font=F(22, med=True), fill=mix(PAPER, INK, a))
        if t > 0.78:
            a = ease(sub(t, 0.78, 0.88))
            dr.line([852, 385, 852 + 46 * a, 385 + 60 * a], fill=BORD, width=4)
            dr.rounded_rectangle([900, 428, 1076, 480], 8, fill=mix(PAPER, CARD, a),
                                 outline=mix(PAPER, RED, a), width=3)
            dr.text((918, 442), "fare bracket", font=F(22, med=True), fill=mix(PAPER, INK, a))
        if t > 0.88:
            dr.text((90, 560), "two heads, one trunk - the graph is the design document",
                    font=F(26, bold=True), fill=NAVY)
    return draw

def s_attend(kicker, title):
    def draw(dr, t, rng):
        kick(dr, kicker)
        dr.text((80, 112), title, font=F(38, bold=True), fill=NAVY)
        toks = ["The", "cat", "sat", "because", "it", "was", "tired"]
        weights = [0.06, 0.62, 0.10, 0.05, 0, 0.07, 0.10]
        xs, x = [], 150
        y = 420
        for tk in toks:
            wd = 40 + len(tk) * 17
            xs.append((x, wd)); x += wd + 26
        focus = 4
        for i, (tk, (tx, wd)) in enumerate(zip(toks, xs)):
            hot = i == focus and t > 0.15
            dr.rounded_rectangle([tx, y, tx + wd, y + 56], 8,
                                 fill=SKYBG if not hot else NAVY, outline=BORD, width=2)
            dr.text((tx + wd/2, y + 14), tk, font=F(26, med=True),
                    fill=INK if not hot else CARD, anchor="ma")
        fx = xs[focus][0] + xs[focus][1] / 2
        for i, (wgt, (tx, wd)) in enumerate(zip(weights, xs)):
            if i == focus or wgt == 0: continue
            at = 0.2 + i * 0.07
            if t <= at: continue
            a = ease(sub(t, at, at + 0.15))
            ox = tx + wd / 2
            midx, midy = (fx + ox) / 2, y - 60 - abs(fx - ox) * 0.22
            steps = 40
            pts = []
            for s_ in range(int(steps * a) + 1):
                u = s_ / steps
                bx = (1-u)**2 * fx + 2*(1-u)*u * midx + u*u * ox
                by = (1-u)**2 * y + 2*(1-u)*u * midy + u*u * y
                pts.append((bx, by))
            if len(pts) > 1:
                dr.line(pts, fill=mix(BORD, BLUE, wgt / 0.62), width=max(2, int(wgt * 22)))
        if t > 0.55:
            tx, wd = xs[1]
            dr.text((tx + wd/2, y - 130), "0.62", font=F(24, bold=True), fill=BLUE, anchor="ma")
        if t > 0.7:
            dr.text((80, 560), '"it" builds its meaning by attending to "cat" — long-range reference in one step',
                    font=F(24), fill=INK)
        if t > 0.88:
            dr.text((80, 600), "score, softmax, weighted sum. That is the mechanism under modern AI.",
                    font=F(24, med=True), fill=NAVY)
    return draw

# ------------------------------------------------------------ the 13 videos
def week(n): return f"week_{n:02d}"

VIDEOS = {
"week_01": [
 (s_title("Module 1 · Introduction and framing", "A model is a function",
          "fitted from data, not written by hand."),
  "Welcome to DATASCI 207. The whole field rests on one reframing: a prediction is a function — inputs go in, an output comes back. Programming writes that function by hand. Machine learning fits it from recorded examples. This module installs that mental model."),
 (s_scatter_fit("The fit", "Learning slides the line"),
  "Here is learning at its smallest. Points are recorded input-output pairs. The model is a line with two free parameters — slope and intercept. Fitting means sliding the line until its errors on the data are small. Every model in this course is this picture with more parameters."),
 (s_scatter_fit("The trap", "Memorizing is not learning", wiggle=True),
  "Now the course's standing warning. A flexible enough model can pass through every training point exactly — and fall apart on the first example it has not seen. That gap is overfitting, and it is why we always hold data back: the test set is the exam the model never studied for."),
 (s_bullets("This module", "Your six parts", [
   "Functions, data, and models — the vocabulary",
   "Loss: how good is the fit, as a number",
   "Train/test split: measuring generalization",
   "Overfitting: watch it happen on purpose"], closing=True),
  "In the notebook you will build all of it from scratch: functions as models, a loss to score them, the train-test split, and a deliberate overfit so you can recognize the failure for the rest of the course. Model, parameters, loss, generalization — four words, thirteen weeks."),
],
"week_02": [
 (s_title("Module 2 · Linear regression and gradient descent", "Learning is minimization",
          "define wrongness, then descend it."),
  "Module one said models have free parameters. This module answers how they are chosen. Define a loss — mean squared error, the average squared gap between prediction and truth. Now learning stops being mystical: it is minimization of that number, nothing more."),
 (s_descent("The algorithm", "Gradient descent"),
  "The minimizer is gradient descent. The gradient says which way the loss increases; step the opposite way, scaled by a learning rate, and repeat. Watch the ball: big confident steps on the steep slope, smaller ones as the floor flattens. Two lines of calculus, applied over and over."),
 (s_bullets("The dials", "What can go wrong", [
   "Learning rate too high — overshoot and diverge",
   "Too low — crawl for hours",
   "Batch: every example per step, stable, slow",
   "Stochastic and mini-batch: noisy, fast, standard"]),
  "The practical content is in the dials. A learning rate too large overshoots the valley and diverges; too small crawls. Batch descent uses all the data per step; stochastic uses one example; mini-batches are the working compromise everyone actually ships."),
 (s_bullets("This module", "Why it matters", [
   "Derive the update rule by hand",
   "Race batch against stochastic descent",
   "Same loop trains everything ahead"], closing=True),
  "In the notebook you derive the update rule yourself and race the three descent flavors on the same surface. Logistic regression, neural networks, transformers — everything ahead trains with exactly this loop. Different model, different loss, same descent."),
],
"week_03": [
 (s_title("Module 3 · Feature engineering", "Models see only numbers",
          "the gap between raw data and those numbers is yours."),
  "Models never see the world — they see the numbers you hand them. This module is about that handoff: turning raw, messy, mixed-type data into features a model can learn from. It decides more real projects than the choice of algorithm does."),
 (s_bullets("The craft", "Three unavoidable jobs", [
   "Scaling: level the loss surface for descent",
   "Missing values: drop, impute, or flag — each an assumption",
   "Categories: one-hot, because red=1 green=2 invents an order"]),
  "Three jobs come up in nearly every dataset. Features on wildly different scales tilt the loss surface, so descent zigzags — standardize them. Missing values must become numbers, and every choice encodes an assumption. And categories get one-hot vectors, because numbering colors invents an ordering that does not exist."),
 (s_bullets("More power", "Staying linear, fitting curves", [
   "Feature crosses: respond to combinations",
   "Bucketing: fit a curve in steps",
   "Matrix form: rows are examples, columns are features"]),
  "Two tricks add expressive power without leaving the linear world. A feature cross multiplies features so the model responds to combinations — location and size together. Bucketing turns a continuous value into ranges, so a straight-line model fits a curve in steps. And it all lives in matrix notation: rows are examples, columns are features."),
 (s_bullets("This module", "The habit to build", [
   "When a model underperforms, look at features first",
   "The notebook: scale, impute, encode, cross, bucket"], closing=True),
  "The habit this module builds: when a model disappoints, inspect the features before reaching for a fancier algorithm. Most of the fix usually lives there. The notebook walks scaling, imputation, encoding, crosses, and buckets on data designed to punish skipping them."),
],
"week_04": [
 (s_title("Module 4 · Logistic regression", "From numbers to categories",
          "squash the score, read a probability."),
  "Regression predicts a number; classification predicts a category. The honest first question is why not run regression on labels zero and one. The lecture starts exactly there, shows how it breaks, and lets the failure motivate the fix."),
 (s_sigmoid("The fix", "Sigmoid, boundary, threshold"),
  "The fix is the sigmoid: squash the linear score into zero-to-one and read it as a probability. Underneath, the geometry stays linear — where the probability crosses one half is a straight decision boundary. And the threshold that turns probability into decision is a dial you own, not a law of nature."),
 (s_bullets("The loss", "Why cross-entropy", [
   "Squared error goes non-convex through the sigmoid",
   "Gradients vanish when confidently wrong",
   "Cross-entropy punishes confident wrongness hard"]),
  "Training needs the right loss, and squared error is the wrong one here — pushed through the sigmoid it goes non-convex, and its gradients die exactly when the model is most confidently wrong. Cross-entropy fixes both: confident wrongness is punished hard, and the surface stays clean for the same descent loop."),
 (s_bullets("This module", "The pipeline", [
   "Sigmoid → probability → threshold → decision",
   "Where you set the threshold is an application question",
   "The template for every classifier ahead"], closing=True),
  "Sigmoid, boundary, threshold, cross-entropy — four pieces, one pipeline. In the notebook you will move the threshold and watch the trade-off: catch more positives, raise more false alarms. Medical screening and spam filtering want different settings of the same model. That decision belongs to you."),
],
"week_05": [
 (s_title("Module 5 · Multiclass and metrics", "Beyond accuracy",
          "measure which way the model is wrong."),
  "Two classes was a warm-up. Softmax generalizes the sigmoid to any number of classes — exponentiate the scores, normalize, read a distribution — and cross-entropy asks one thing: how much probability landed on the true class? The same training loop carries over unchanged."),
 (s_matrix("The ledger", "The confusion matrix"),
  "The heavier half of the module is measurement. Watch the matrix fill: ninety true positives, ten misses, two hundred forty false alarms. Accuracy reads seventy-five percent and sounds fine. Precision — of everything flagged, how much was real — reads twenty-seven. On imbalanced data, accuracy is a liar."),
 (s_bullets("The metrics", "Precision, recall, F1", [
   "Precision: of what I flagged, what was real?",
   "Recall: of what was real, what did I flag?",
   "They trade off through the threshold",
   "F1 collapses the pair — and hides which side is weak"]),
  "From the matrix come precision and recall — of what I flagged, how much was real; of what was real, how much did I catch. They pull against each other through last module's threshold. F1 averages them into one number for ranking models, at the cost of hiding which side is failing."),
 (s_bullets("This module", "The discipline", [
   "Never report one number for a classifier that matters",
   "Print the confusion matrix first",
   "The two error types almost never cost the same"], closing=True),
  "The discipline this module installs: never report a single number for a classifier that matters. Print the matrix first — it shows not just how often the model is wrong but which way, and the two directions almost never cost the same. The notebook builds softmax and every metric from scratch."),
],
"week_06": [
 (s_title("Module 6 · Feedforward networks", "When lines are not enough",
          "four points that defeat every straight line."),
  "Everything so far draws straight boundaries. This module opens with the four-point dataset that ends that era: XOR — output one when exactly one input is one. No line separates it. To go further, models need to bend."),
 (s_xor("The problem", "XOR, and the bend"),
  "Watch a single line try: every angle misclassifies at least one point. Now a hidden layer computes two linear functions, passes them through a nonlinearity, and combines them — the boundary bends into a V and the four points separate. That bend, stacked deep, is deep learning."),
 (s_bullets("The machinery", "How networks compute and learn", [
   "Layers: matrix multiply, then nonlinearity",
   "Without the nonlinearity, the stack collapses to one line",
   "ReLU won because its gradient survives depth",
   "Backprop: the chain rule, walked backwards"]),
  "Forward propagation is matrix multiplies with a nonlinearity between layers — remove the nonlinearity and the whole stack collapses back into one linear map. ReLU won the activation wars because its gradient survives depth. And backpropagation is the chain rule walked backwards from the loss, handing every weight its gradient for the same descent step you have run since module two."),
 (s_bullets("This module", "Scratch first, Keras second", [
   "Build forward and backward passes in NumPy",
   "Train a network to solve XOR",
   "Then watch Keras do it in three lines"], closing=True),
  "In the notebook you build the network in raw NumPy — forward pass, backward pass, training loop — and teach it XOR. Then you rebuild it in Keras, where it becomes three lines. Frameworks buy convenience, not understanding. That is why the scratch version comes first."),
],
"week_07": [
 (s_title("Module 7 · KNN, trees, and ensembles", "Questions, not equations",
          "two model families that never compute a gradient."),
  "This module breaks the pattern on purpose: not every model is a function trained by descent. K-nearest neighbors barely trains at all — store the data, and answer with the vote of the closest stored examples. Simple, surprisingly strong, and a lesson in what similarity can do."),
 (s_tree("The tree", "Learned questions"),
  "A decision tree asks learned questions and routes each example down branches to an answer. Learning is choosing the questions: entropy measures a group's impurity, and each split is picked for maximum information gain. The result is a model you can read aloud — and one that happily memorizes noise if grown deep."),
 (s_bullets("The ensembles", "Many weak trees, one strong model", [
   "Random forest: bootstrapped trees, averaged — variance tamed",
   "Gradient boosting: small trees fixing the errors so far — bias attacked",
   "Both beat a single tuned tree, routinely"]),
  "Ensembles fix the tree's flaws with statistics. A random forest trains many trees on resampled data and averages them — individually noisy, collectively stable. Boosting builds small trees in sequence, each correcting the errors of the sum so far. Both routinely beat any single tree you could tune."),
 (s_bullets("This module", "Where these win", [
   "On tabular data, tuned boosting is the baseline to beat",
   "Reach for it before a neural network on tables",
   "Networks earn their keep on images, text, audio"], closing=True),
  "The practical takeaway is scope. On tables of features — most business data — tuned gradient boosting is often the strongest model you can field, and it is the baseline to beat. Neural networks earn their keep on images, text, and audio, which is exactly where this course goes next."),
],
"week_08": [
 (s_title("Module 8 · Unsupervised learning", "No labels",
          "what structure do the inputs carry on their own?"),
  "Every module so far had labels — the right answers came with the data. This one takes them away and asks what structure the inputs carry on their own. Two answers: group the examples, or compress the dimensions."),
 (s_clusters("The loop", "K-means, converging"),
  "K-means at its most direct: place centroids, assign each point to the nearest, move each centroid to the mean of its points, repeat until nothing moves. Watch the squares find the clouds. The catch is honest — you chose k, and the elbow and silhouette methods are evidence for that choice, not proof."),
 (s_gmm("Soft borders", "Gaussian mixtures and EM"),
  "K-means draws hard borders; Gaussian mixtures admit what the borders hide. Model the data as overlapping normal distributions, and every point gets a probability of belonging to each. Expectation maximization fits it: softly assign, re-estimate, repeat. It is k-means minus the certainty — and the fitted mixture is generative: it can invent plausible new points."),
 (s_bullets("Compression", "PCA, trees, density, pictures", [
   "PCA: keep the directions of greatest variance",
   "SVD: the same machinery, general form",
   "Hierarchies read a tree; DBSCAN lets density decide k",
   "t-SNE: pictures to look at, never to measure"]),
  "PCA compresses instead of grouping: keep the directions of greatest variance, and hundreds of correlated columns become a handful of axes. S V D is the same machinery in general form. Hierarchical clustering reads a tree at any depth; DBSCAN lets density decide the number of clusters and calls sparse points noise. And t-SNE draws neighborhood-faithful pictures — for looking, never for measuring."),
 (s_bullets("This module", "The caveat that governs it", [
   "No labels → no ground truth",
   "“Four segments” is a reading, not a fact",
   "Unsupervised results are hypotheses"], closing=True),
  "The caveat that governs all of it: with no labels there is no ground truth. When the algorithm reports four customer segments, that is a defensible reading of the evidence, not a fact about the world. Treat unsupervised results as hypotheses — then test them. The notebook builds k-means and PCA from scratch."),
],
"week_09": [
 (s_title("Module 9 · Text and embeddings", "Text into numbers",
          "without losing what it means."),
  "Models eat numbers, and text is not numbers. Everything in this module answers one question: how do you turn language into vectors without losing the meaning? The answers run from counting words to learning geometry."),
 (s_bullets("Counting", "Bag-of-words and its patches", [
   "Bag-of-words: one column per word, order ignored",
   "TF-IDF: down-weight the everywhere-words",
   "N-grams: recover short-range order",
   "All sparse, all brittle: excellent ≠ outstanding"]),
  "The counting era first. Bag-of-words gives each vocabulary word a column and ignores order entirely. TF-IDF weights words down when they appear everywhere, so gradient starts mattering and the stops mattering. N-grams recover short phrases. All of it works — and all of it is blind: excellent and outstanding share no dimensions at all."),
 (s_vectors("Geometry", "Meaning as direction"),
  "Embeddings are the modern answer: give each word a short dense vector learned from the company it keeps. Now similarity is geometry — synonyms point the same way, and cosine similarity, the angle between vectors, measures relatedness that counting can never see. Watch: excellent and outstanding, ninety-eight hundredths. Terrible, fifty-seven."),
 (s_bullets("This module", "Where this road ends", [
   "Build bag-of-words, TF-IDF, cosine from scratch",
   "Average word vectors into document embeddings",
   "This pipeline, scaled a million-fold, feeds every transformer"], closing=True),
  "In the notebook you build the whole road from scratch — counts, TF-IDF, cosine similarity, document embeddings. Keep the destination in view: this token-and-vector pipeline, scaled a million-fold, is the front end of every transformer. Module thirteen finishes the road."),
],
"week_10": [
 (s_title("Module 10 · Convolutional networks", "Vision needs structure",
          "neighbors matter; flattening forgets them."),
  "Flatten a small image into a vector and a dense network needs tens of thousands of weights per neuron — and it forgets that pixel neighbors are neighbors. Convolutions fix both at once, and they are the reason computer vision works."),
 (s_conv("The mechanism", "A filter scans the image"),
  "A convolution slides a small filter across the image, computing a dot product at each position — watch the red window scan and the feature map fill. The same nine weights visit every location, so a pattern learned once is found anywhere, and parameters drop from millions to dozens."),
 (s_bullets("The stack", "From edges to objects", [
   "Pooling: keep the strongest response, buy shift-tolerance",
   "Early layers: edges → textures → parts → objects",
   "Output-size arithmetic: do it by hand once"]),
  "Pooling downsamples the feature maps, keeping the strongest responses and buying tolerance to small shifts. Stack convolution, nonlinearity, and pooling and the layers specialize: edges, then textures, then parts, then objects. The output-size arithmetic is unglamorous — do it by hand once, because every shape bug in Keras traces back to it."),
 (s_bullets("This module", "The two working tricks", [
   "Transfer learning: start from a pretrained network",
   "Augmentation: flip, crop, shift your data",
   "Between them, small-data vision gets done"], closing=True),
  "The two closing ideas are the ones practitioners lean on daily. Transfer learning: a network pretrained on millions of images has already learned edges and textures — start from it and retrain the top. Augmentation: flip, crop, and shift what you have. Between them, serious vision work gets done on small datasets."),
],
"week_11": [
 (s_title("Module 11 · Network architecture design", "Wiring, not stacking",
          "build the network around the data you have."),
  "Module eleven changes the question. Until now, every network was a stack — layers in a line. Real data is not a stack: a passenger manifest mixes numbers with categories. This module builds the network as a graph, wired around the data you actually have."),
 (s_wiring("The graph", "Four features, four front doors"),
  "Four features, four front doors. Raw numbers flow straight in. A number whose effect is not smooth gets bucketed into ranges. Categories become one-hot vectors — or embeddings, which buy geometry with parameters. Concatenate merges the paths into a shared trunk, and the trunk can feed two heads at once, each with its own loss. This graph is the Keras functional A P I, and the drawing is the design document."),
 (s_bullets("The ladder", "Five models on one manifest", [
   "Baselines first: majority class, then one rule",
   "Sequential, then Functional — same model, two APIs",
   "Buckets, named inputs, embeddings, two outputs",
   "Read the parameter count like a bill"]),
  "The lecture climbs five models on the Titanic manifest. Baselines first — the majority class, then one honest rule, because a network that cannot beat one rule is decoration. Then the same model in both A P Is, then bucketed inputs, then multiple named inputs, then embedded categoricals, then a second output head. At every rung, read the parameter count like a bill: every wiring choice prices in."),
 (s_bullets("This module", "Architecture is a claim", [
   "Every wire encodes a belief about the data",
   "Bucketing asserts: this effect is not smooth",
   "The notebook wires and trains the real thing"], closing=True),
  "One idea to carry out the door: architecture is a hypothesis. Bucketing age claims its effect is not smooth. An embedding claims the categories have geometry worth learning. Make the claims deliberately, and let the dev set judge them. The notebook wires the real thing — and the tuning thread continues in the assignments."),
],
"week_12": [
 (s_title("Module 12 · Fairness and responsible AI", "Accurate and unfair",
          "nothing malfunctioned. That is the point."),
  "A model trained on history learns history's patterns — including the ones a deployment should not repeat. This module makes it concrete: a hiring classifier, trained on biased outcomes, reproduces the bias with clean code and a good accuracy score. Nothing malfunctioned. That is the point."),
 (s_fairbars("The measurement", "Same accuracy, different odds"),
  "Fairness is measured with tools you already own — module five's confusion matrix, computed per group. Demographic parity asks whether groups are selected at equal rates. Equalized odds asks whether qualified candidates face the same true-positive rate regardless of group. Watch: one model, one accuracy, very different odds — until a per-group threshold equalizes it, and arithmetic moves the cost elsewhere."),
 (s_bullets("The theorem", "You must choose", [
   "When base rates differ, the fairness criteria conflict",
   "Not an engineering shortfall — arithmetic",
   "Choosing a definition is a values decision",
   "Made explicitly, or made by default"]),
  "Then the uncomfortable mathematics: when base rates differ between groups, several reasonable fairness criteria cannot all hold at once. That is not an engineering shortfall to be patched — it is arithmetic. Choosing which definition to satisfy is a decision about values, and it gets made either explicitly or by default."),
 (s_bullets("This module", "The professional posture", [
   "Audit by group, never only in aggregate",
   "Name the fairness definition you chose, and why",
   "“The model is accurate” is the start of evaluation"], closing=True),
  "The posture this module asks of you: audit by group, not in aggregate. Say which fairness definition you chose and why. And treat the model is accurate as the beginning of an evaluation, never the end. The systems you build will touch people who never saw the training data."),
],
"week_13": [
 (s_title("Module 13 · Transformers and attention", "Weighted looking",
          "the mechanism under modern AI, small enough to read."),
  "Module nine left text as averaged vectors — order ignored, context lost. The transformer fixed that, and it runs essentially every system currently called AI. This final module builds its core mechanism small enough to read."),
 (s_attend("The mechanism", "Attention"),
  "Attention is weighted looking. Each position computes a query, scores it against every position's key, softmaxes the scores into weights, and takes the weighted sum of values. Watch the word it assemble its meaning: sixty-two percent of its attention flows to cat, forty tokens or four — the distance does not matter."),
 (s_bullets("The refinements", "Three pieces complete the block", [
   "Causal masking: no peeking at the future — honest next-token training",
   "Multi-head: several attention patterns in parallel",
   "Positional encoding: order, injected — attention alone sees a bag"]),
  "Three refinements complete the block. Causal masking zeroes attention to future positions — that is what makes next-token prediction honest during training, and it is exactly how the chat models you use were trained. Multi-head runs several attention patterns in parallel. Positional encodings inject order, because attention alone treats a sentence as a bag of words."),
 (s_bullets("The course", "Where you have arrived", [
   "Attention + residuals + layer norm + feedforward = the block",
   "Stacked blocks ≈ the whole architecture",
   "It is matrix multiplies, softmax, and gradient descent",
   "You have now built every piece from scratch"], closing=True),
  "Assembled with residuals, layer norm, and a feedforward layer, this is the encoder block — and stacking such blocks is, to a first approximation, the whole architecture. The course ends on a deliberate note: the frontier of AI is matrix multiplies, softmax, and gradient descent. You have now implemented every piece of it from scratch."),
],
}

def narrate(pipe, text, out_wav, pad=0.45):
    chunks = [a for _, _, a in pipe(text, voice="af_heart")]
    audio = np.concatenate(chunks)
    audio = np.concatenate([audio, np.zeros(int(24000 * pad), dtype=audio.dtype)])
    sf.write(out_wav, audio, 24000)
    return len(audio) / 24000

def build(slug, pipe):
    scenes = VIDEOS[slug]
    total = len(scenes)
    modnum = int(slug.split("_")[1])
    wavs, durs = [], []
    for i, (_, say) in enumerate(scenes, 1):
        wav = f"{WORK}/{slug}-{i}.wav"
        d = narrate(pipe, say, wav)
        wavs.append(wav); durs.append(d)
    fdir = f"{WORK}/{slug}-frames"
    os.makedirs(fdir, exist_ok=True)
    for old in os.listdir(fdir): os.remove(f"{fdir}/{old}")
    n = 0
    for i, ((fn, _), d) in enumerate(zip(scenes, durs), 1):
        nf = int(round(d * FPS))
        rng = random.Random(i * 104729)
        for k in range(nf):
            t = (k + 0.5) / nf
            im, dr = frame_base(i, total, modnum)
            fn(dr, t, rng)
            im.save(f"{fdir}/f{n:05d}.png")
            n += 1
    allaudio = np.concatenate([sf.read(w)[0] for w in wavs])
    concat = f"{WORK}/{slug}-all.wav"
    sf.write(concat, allaudio, 24000)
    out = f"{ROOT}/video/{slug}.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", f"{fdir}/f%05d.png",
                    "-i", concat, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-r", "30", "-c:a", "aac", "-b:a", "96k", "-shortest", out],
                   capture_output=True, check=True)
    print(f"{out}  ({sum(durs)/60:.1f} min, {n} frames)", flush=True)

if __name__ == "__main__":
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for slug in VIDEOS:
        if only and slug != only:
            continue
        build(slug, pipe)
    print("DONE", flush=True)
