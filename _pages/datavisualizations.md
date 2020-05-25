---
layout: archive
permalink: /data-visualizations/
title: "Data Visualizations"
author_profile: true
header:
  image: "/images/dataviz.png"
---


{% for post in site.categories.datavisualization %}
      {% include archive-single.html %}
{% endfor %}