---
layout: cv
permalink: /cv/
title: CV
nav: false
nav_order: 5
cv_pdf: /assets/pdf/example_pdf.pdf # you can also use external links here
cv_format: rendercv # options: rendercv, jsonresume
description: This is a description of the page. You can modify it in '_pages/cv.md'. You can also change or remove the top pdf download button.
toc:
  sidebar: left
---


{% if site.enable_mesh_background %}
  <!-- Floating mesh background -->
  <script defer src="https://cdn.jsdelivr.net/npm/delaunator@5.1.0/delaunator.min.js"></script>
  <script defer src="{{ '/assets/js/mesh-bg.js' | relative_url | bust_file_cache }}"></script>
{% endif %}