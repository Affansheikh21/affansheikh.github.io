---
layout: archive
permalink: /data-visualizations/
title: "Data Visualizations"
author_profile: true
header:
  image: "/images/header.jpg"
---


<ul>
{% for category in site.categories %}
  <li><a name="{{ category | first }}">{{ category | first }}</a>
    <ul>
    {% for post in category.last %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
    </ul>
  </li>
{% endfor %}
</ul>