/* MSAI visual standard: shared drawing library for vertical artifacts.
   See ../STYLE.md. An artifact calls VIZ.run({duration, setup, render}).
   render(t, ctx) must draw the whole frame for time t (seconds) from scratch:
   no state carried between frames, so any frame can be rendered in any order
   and the exporter can step through time exactly. */
(function () {
  var P = {
    ground: "#07120f", groundWarm: "#13110d",
    teal: "#7fc4b2", amber: "#d8a65c", coral: "#e8836a",
    form: "#38351f", ink: "#e8e3d4", dim: "#8b887b", hair: "#24302c"
  };
  var W = 1080, H = 1920;

  function rgba(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return "rgba(" + (n >> 16 & 255) + "," + (n >> 8 & 255) + "," + (n & 255) + "," + a + ")";
  }
  function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
  function lerp(a, b, u) { return a + (b - a) * u; }
  // smoothstep, and the eased progress of a window [t0, t1] at time t
  function ease(u) { u = clamp(u, 0, 1); return u * u * (3 - 2 * u); }
  function span(t, t0, t1) { return ease((t - t0) / (t1 - t0)); }
  function rng(seed) {             // mulberry32: the same draws on every machine
    var a = seed >>> 0;
    return function () {
      a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function normal(r) { var u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  // drawing primitives: hairlines, dashed connectors, dots, filled forms, curves
  function line(ctx, x1, y1, x2, y2, color, w, dash) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = w || 2; ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash === true ? [10, 10] : dash);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore();
  }
  function dot(ctx, x, y, color, r) {
    ctx.save(); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r || 6, 0, 7); ctx.fill(); ctx.restore();
  }
  function ring(ctx, x, y, r, color, w) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = w || 2; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke(); ctx.restore();
  }
  // a polyline from an array of [x, y]; frac in [0, 1] draws only the first part
  function path(ctx, pts, color, w, frac, dash) {
    if (pts.length < 2) return;
    var n = frac === undefined ? pts.length : Math.max(2, Math.floor(pts.length * clamp(frac, 0, 1)));
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = w || 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash === true ? [10, 10] : dash);
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (var i = 1; i < n; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke(); ctx.restore();
  }
  function form(ctx, pts, alpha) {   // the muted filled silhouette: the object under study
    ctx.save(); ctx.fillStyle = rgba(P.form, alpha === undefined ? 1 : alpha);
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath(); ctx.fill(); ctx.restore();
  }
  function text(ctx, s, x, y, color, size, align, weight) {
    ctx.save(); ctx.fillStyle = color || P.dim; ctx.textAlign = align || "left";
    ctx.font = (weight || 500) + " " + (size || 28) + 'px "Inter","Helvetica Neue",Arial,sans-serif';
    ctx.fillText(s, x, y); ctx.restore();
  }

  // overlays: math and words are HTML on top of the canvas, typeset by KaTeX
  function tex(el, s) {
    if (el.__tex === s) return; el.__tex = s;
    if (window.katex) { el.innerHTML = s.replace(/\$([^$]+)\$/g, function (_, m) { return katex.renderToString(m, { throwOnError: false }); }); }
    else el.textContent = s.replace(/\$/g, "");
  }
  function el(cls, html, style) {
    var d = document.createElement("div"); d.className = cls;
    if (style) for (var k in style) d.style[k] = style[k];
    document.getElementById("stage").appendChild(d);
    if (html !== undefined) tex(d, html);
    return d;
  }
  function fade(d, a) { d.style.opacity = clamp(a, 0, 1); }

  function fit() {
    var s = document.getElementById("stage"), k = Math.min(innerWidth / W, innerHeight / H);
    s.style.transform = "translate(-50%,-50%) scale(" + k + ")";
  }

  function run(A) {
    var stage = document.getElementById("stage");
    var cv = document.createElement("canvas"); cv.width = W; cv.height = H; stage.insertBefore(cv, stage.firstChild);
    var ctx = cv.getContext("2d");
    var mark = el("mark"); mark.innerHTML = "<b>" + (A.mark || "MSAI") + "</b>" + (A.course || "");   // A.mark: another course's wordmark
    if (A.setup) A.setup();
    function frame(t) {
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, W, H);
      A.render(t, ctx);
    }
    // the exporter drives time itself: window.VIZ_FRAME(t) draws one frame
    window.VIZ_FRAME = frame;
    window.VIZ_META = { duration: A.duration, fps: A.fps || 30, title: A.title || document.title };
    window.VIZ_POSTER = A.poster;
    var exporting = /[?&]export\b/.test(location.search);
    if (exporting) { stage.style.transform = "none"; stage.style.left = "0"; stage.style.top = "0"; frame(0); return; }
    fit(); addEventListener("resize", fit);
    var reduce = matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    var t0 = performance.now();
    var hold = /[?&]t=([\d.]+)/.exec(location.search);
    if (hold) { frame(+hold[1]); return; }
    if (reduce) { frame(A.poster === undefined ? A.duration * 0.8 : A.poster); return; }
    (function loop(now) { frame(((now - t0) / 1000) % A.duration); requestAnimationFrame(loop); })(t0);
  }

  window.VIZ = { P: P, W: W, H: H, rgba: rgba, clamp: clamp, lerp: lerp, ease: ease, span: span, rng: rng, normal: normal,
    line: line, dot: dot, ring: ring, path: path, form: form, text: text, tex: tex, el: el, fade: fade, run: run };
})();

/* Added 2026-10-12 (207 module 8: dendrogram, choosing-k). VIZ.hulls(ctx,
   groups, pad, alpha): each group (an array of [x, y] stage points) drawn as
   its convex hull padded by pad pixels, filled with the form colour. All the
   groups of one call are drawn opaque off screen and laid down at one alpha,
   so a fading set of clusters shows no doubled edges where shapes overlap. */
(function () {
  var off = null;
  function convex(pts) {
    pts = pts.slice().sort(function (p, q) { return p[0] - q[0] || p[1] - q[1]; });
    function cross(o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); }
    var lo = [], up = [], i;
    for (i = 0; i < pts.length; i++) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], pts[i]) <= 0) lo.pop(); lo.push(pts[i]); }
    for (i = pts.length - 1; i >= 0; i--) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], pts[i]) <= 0) up.pop(); up.push(pts[i]); }
    var h = lo.slice(0, -1).concat(up.slice(0, -1));
    return h.length ? h : [pts[0]];
  }
  function hulls(ctx, groups, pad, alpha) {
    if (alpha === undefined) alpha = 1;
    if (alpha <= 0.005) return;
    if (!off) { off = document.createElement("canvas"); off.width = VIZ.W; off.height = VIZ.H; }
    var o = off.getContext("2d");
    o.setTransform(1, 0, 0, 1, 0, 0); o.clearRect(0, 0, off.width, off.height);
    o.fillStyle = o.strokeStyle = VIZ.P.form; o.lineWidth = 2 * (pad || 16); o.lineJoin = "round"; o.lineCap = "round";
    groups.forEach(function (g) {
      if (!g.length) return;
      var h = convex(g);
      o.beginPath(); o.moveTo(h[0][0], h[0][1]);
      for (var k = 1; k < h.length; k++) o.lineTo(h[k][0], h[k][1]);
      if (h.length === 1) o.lineTo(h[0][0] + 0.01, h[0][1]);
      o.closePath(); o.fill(); o.stroke();
    });
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha *= alpha; ctx.drawImage(off, 0, 0); ctx.restore();
  }
  VIZ.hulls = hulls;
})();
