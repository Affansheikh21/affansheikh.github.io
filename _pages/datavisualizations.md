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
	{% if page.categories contains "datavisualization" %}	
	  <li><a name="{{ category | first }}">{{ category | first }}</a>
	    <ul>
	    {% for post in category.last %}
	      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
	    {% endfor %}
	    </ul>
	  </li>
		{% endif %}
	{% endfor %}
	</ul>