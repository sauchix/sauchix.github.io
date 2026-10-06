---
layout: page
title: submenus
nav: false
nav_order: 8
dropdown: true
children:
  - title: bookshelf
    permalink: /books/
  - title: divider
  - title: blog
    permalink: /blog/
---


{% if site.enable_mesh_background %}
  <!-- Floating mesh background -->
  <script defer src="https://cdn.jsdelivr.net/npm/delaunator@5.1.0/delaunator.min.js"></script>
  <script defer src="{{ '/assets/js/mesh-bg.js' | relative_url | bust_file_cache }}"></script>
{% endif %}