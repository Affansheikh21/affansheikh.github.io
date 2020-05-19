---
layout: archive
permalink: /data-visualizations/
title: "Data Visualizations"
author_profile: true
header:
  image: "/images/header.jpg"
---


{% for post in site.categories.datavisualization %}
    <li>{{ post.title }}</li>
      {% include archive-single.html %}
{% endfor %}