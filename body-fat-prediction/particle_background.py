"""Full-page particle network background for Streamlit.

Ported from the "Aether Flow" React hero (21st.dev). Only the canvas
background is kept: the pill badge, gradient headline, filler copy and
"Explore the Engine" button were template content, not part of this app.

Changes from the original, all deliberate:
- Colours come from the app palette instead of hard-coded purple on pure black.
- Particle count is capped and link distance is a fixed pixel radius, so the
  O(n^2) link pass stays cheap on large screens (the original grew without bound
  and computed opacities below zero).
- Canvas is scaled for devicePixelRatio, so lines are crisp on HiDPI screens.
- prefers-reduced-motion draws one still frame and never animates.
- The loop pauses while the tab is hidden and cleans up its listeners when
  Streamlit replaces the iframe, so reruns do not stack animation loops.
- The canvas never takes pointer events, so it cannot block the app.
"""

import streamlit.components.v1 as components

_TEMPLATE = """
<canvas id="bg" aria-hidden="true"></canvas>
<style>html,body{margin:0;background:transparent;overflow:hidden}canvas{display:block}</style>
<script>
(() => {
  const DOT = "__DOT__", LINK = "__LINK__", NEAR = "__NEAR__";
  const MAX_PARTICLES = 110, LINK_DIST = 140, MOUSE_R = 170;

  // Pin this iframe behind the whole app and take it out of the page flow.
  const frame = window.frameElement;
  const host = window.parent;
  if (frame) {
    Object.assign(frame.style, {
      position: "fixed", inset: "0", width: "100vw", height: "100vh",
      zIndex: "-1", border: "0", pointerEvents: "none",
    });
    const box = frame.closest('[data-testid="stElementContainer"]');
    if (box) Object.assign(box.style, { position: "absolute", height: "0", margin: "0" });
  }

  const canvas = document.getElementById("bg");
  const ctx = canvas.getContext("2d");
  const reduce = host.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mouse = { x: null, y: null };
  let w = 0, h = 0, particles = [], raf = 0;

  function seed() {
    const n = Math.min(MAX_PARTICLES, Math.round((w * h) / 14000));
    particles = Array.from({ length: n }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.2 + 0.8,
    }));
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
    if (reduce) frameStep(false);
  }

  function frameStep(move) {
    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      if (move) {
        if (mouse.x !== null) {
          const dx = p.x - mouse.x, dy = p.y - mouse.y, d = Math.hypot(dx, dy);
          if (d < MOUSE_R && d > 0) {
            const f = (1 - d / MOUSE_R) * 1.2;
            p.x += (dx / d) * f; p.y += (dy / d) * f;
          }
        }
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = DOT;
      ctx.fill();
    }
    ctx.lineWidth = 1;
    for (let a = 0; a < particles.length; a++) {
      for (let b = a + 1; b < particles.length; b++) {
        const pa = particles[a], pb = particles[b];
        const d = Math.hypot(pa.x - pb.x, pa.y - pb.y);
        if (d > LINK_DIST) continue;
        const near = mouse.x !== null && Math.hypot(pa.x - mouse.x, pa.y - mouse.y) < MOUSE_R;
        ctx.globalAlpha = (1 - d / LINK_DIST) * (near ? 0.55 : 0.22);
        ctx.strokeStyle = near ? NEAR : LINK;
        ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  function loop() { frameStep(true); raf = requestAnimationFrame(loop); }
  function start() { if (!reduce && !raf) raf = requestAnimationFrame(loop); }
  function stop() { cancelAnimationFrame(raf); raf = 0; }

  const onMove = (e) => { mouse.x = e.clientX; mouse.y = e.clientY; };
  const onLeave = () => { mouse.x = null; mouse.y = null; };
  const onVisibility = () => (host.document.hidden ? stop() : start());

  host.addEventListener("mousemove", onMove, { passive: true });
  host.document.addEventListener("mouseleave", onLeave);
  host.document.addEventListener("visibilitychange", onVisibility);
  window.addEventListener("resize", resize);

  // Streamlit may swap this iframe on rerun: detach from the parent window.
  window.addEventListener("pagehide", () => {
    stop();
    host.removeEventListener("mousemove", onMove);
    host.document.removeEventListener("mouseleave", onLeave);
    host.document.removeEventListener("visibilitychange", onVisibility);
  });

  resize();
  start();
})();
</script>
"""


def render_particle_background(dot: str, link: str, near: str) -> None:
    """Draw the background. Call once, near the top of the script.

    dot   rgba() for particles
    link  CSS colour for connecting lines (alpha is applied per line)
    near  CSS colour for lines close to the cursor
    """
    html = (_TEMPLATE.replace("__DOT__", dot)
                     .replace("__LINK__", link)
                     .replace("__NEAR__", near))
    components.html(html, height=0)
