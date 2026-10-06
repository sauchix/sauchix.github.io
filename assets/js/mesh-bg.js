/**
 * mesh-bg.js — floating triangle-mesh background.
 *
 * Points drift slowly across the page and are re-triangulated every frame
 * (Delaunay, via the Delaunator library), so they always form a proper mesh.
 * The cursor joins the mesh as an extra vertex and brightens nearby edges.
 *
 * Turn it off: set `enable_mesh_background: false` in _config.yml.
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // Settings — change these to adjust the look and feel.
  // ---------------------------------------------------------------------------
  var CFG = {
    density: 4000,     // screen area (px²) per point. Smaller = more points
    maxPoints: 500,    // upper limit on points, so big screens stay smooth
    speed: 0.3,        // how fast points drift
    maxEdge: 170,      // px — edges longer than this are hidden
    cursorR: 220,      // px — how far the cursor's glow reaches
    dotR: 1.6,         // dot radius in px

    // Line colours (R, G, B)
    dark:  { r: 255, g: 255, b: 255 },  // dark mode: white
    light: { r: 60,  g: 60,  b: 60 },   // light mode: dark grey

    // Opacity (0 = invisible, 1 = solid)
    edgeAlpha: 0.14,       // edges at rest
    edgeNearAlpha: 0.45,   // extra brightness near the cursor
    dotAlpha: 0.3,         // dots at rest
    dotNearAlpha: 0.6,     // extra brightness near the cursor
  };

  if (typeof Delaunator === 'undefined') return; // library didn't load

  var reduceMotion = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------------------------------------------------------------------
  // Canvas
  // ---------------------------------------------------------------------------
  var canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  canvas.style.cssText =
    'position:fixed;top:0;left:0;width:100%;height:100%;' +
    'z-index:-1;pointer-events:none;display:block';
  document.body.insertBefore(canvas, document.body.firstChild);

  var ctx = canvas.getContext('2d');
  var W = 0, H = 0, dpr = 1;
  var points = [];
  var mouse = { x: -1e5, y: -1e5, on: false };

  function resize() {
    dpr = window.devicePixelRatio || 1;
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    var n = Math.min(CFG.maxPoints, Math.round((W * H) / CFG.density));
    // keep existing points where possible so resizing doesn't jump
    while (points.length < n) points.push(newPoint());
    points.length = n;
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      if (p.x > W + 40 || p.y > H + 40) { p.x = Math.random() * W; p.y = Math.random() * H; }
    }
    if (reduceMotion) draw();
  }

  function newPoint() {
    var a = Math.random() * Math.PI * 2;
    var s = CFG.speed * (0.4 + Math.random() * 0.6);
    return { x: Math.random() * W, y: Math.random() * H, vx: Math.cos(a) * s, vy: Math.sin(a) * s };
  }

  function colour() {
    var dark = document.documentElement.getAttribute('data-theme') === 'dark';
    var c = dark ? CFG.dark : CFG.light;
    return c.r + ',' + c.g + ',' + c.b;
  }

  function nearness(x, y) {
    if (!mouse.on) return 0;
    var d = Math.hypot(mouse.x - x, mouse.y - y);
    return d < CFG.cursorR ? 1 - d / CFG.cursorR : 0;
  }

  // ---------------------------------------------------------------------------
  // Draw one frame
  // ---------------------------------------------------------------------------
  function draw() {
    ctx.clearRect(0, 0, W, H);
    if (points.length < 3) return;

    var c = colour();
    var all = mouse.on ? points.concat([{ x: mouse.x, y: mouse.y }]) : points;
    var d = Delaunator.from(all, function (p) { return p.x; }, function (p) { return p.y; });
    var tri = d.triangles, half = d.halfedges;

    ctx.lineWidth = 1;
    for (var e = 0; e < tri.length; e++) {
      if (e < half[e]) continue; // draw each shared edge once
      var a = all[tri[e]];
      var b = all[tri[e % 3 === 2 ? e - 2 : e + 1]];
      var len = Math.hypot(a.x - b.x, a.y - b.y);
      if (len > CFG.maxEdge) continue;
      var near = nearness((a.x + b.x) / 2, (a.y + b.y) / 2);
      var alpha = (CFG.edgeAlpha + CFG.edgeNearAlpha * near) * (1 - len / CFG.maxEdge);
      ctx.strokeStyle = 'rgba(' + c + ',' + alpha.toFixed(3) + ')';
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      var da = CFG.dotAlpha + CFG.dotNearAlpha * nearness(p.x, p.y);
      ctx.fillStyle = 'rgba(' + c + ',' + da.toFixed(3) + ')';
      ctx.beginPath();
      ctx.arc(p.x, p.y, CFG.dotR, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function move() {
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < -30) { p.x = -30; p.vx = Math.abs(p.vx); }
      if (p.x > W + 30) { p.x = W + 30; p.vx = -Math.abs(p.vx); }
      if (p.y < -30) { p.y = -30; p.vy = Math.abs(p.vy); }
      if (p.y > H + 30) { p.y = H + 30; p.vy = -Math.abs(p.vy); }
    }
  }

  function tick() {
    requestAnimationFrame(tick);
    if (document.hidden) return; // don't work while the tab is in the background
    move();
    draw();
  }

  // ---------------------------------------------------------------------------
  // Events
  // ---------------------------------------------------------------------------
  window.addEventListener('resize', resize);
  window.addEventListener('mousemove', function (e) {
    mouse.x = e.clientX; mouse.y = e.clientY; mouse.on = true;
    if (reduceMotion) draw();
  });
  document.addEventListener('mouseleave', function () {
    mouse.on = false;
    if (reduceMotion) draw();
  });
  // redraw immediately when the light/dark toggle is used
  new MutationObserver(function () { if (reduceMotion) draw(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  resize();
  if (!reduceMotion) requestAnimationFrame(tick); // still image for "reduce motion" users
})();