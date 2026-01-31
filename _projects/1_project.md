---
layout: page
title: GenieEngine
description: A RAG + LLM-powered game search engine that combines mathematically precise search algorithms with the reasoning capabilities of an LLM, with a database of over 130000 games (currently all PC games).
img: assets/img/book_covers/genieengine.jpg
importance: 1
category: fun
related_publications: false
---

I launched GenieEngine (<a href="www.genieengine.ai/">Jekyll</a>)
A RAG + LLM-powered game search engine that combines mathematically precise search algorithms with the reasoning capabilities of an LLM, with a database of over 130000 games (currently all PC games).

Mainstream game stores such as Steam, App Store contain hundreds of thousands of games, effectively monopolizing their platforms. However, their game search systems are not so good... failing both the players and developers. In 2025, there are over 19,200 games being released on Steam, but reports indicate that nearly half of these new titles struggle to get even 10 user reviews, effectively launching into a void.

1. Store front pages are usually filled with massive AAA titles sitting on the leaderboard for months and even years. 
2. "Rich gets richer", these big game companies have giant amount of funds to promote their games compare to smaller developers/companies who also make quality games.
3. As new games come out they must take off immediately else they lose market momentum. This leaves thousands of high-quality games—released just months or a few years ago—completely buried.
4. Searching via categories is not so helpful - it is too broad and time consuming. From my observations people usually have very specific demands about the game they want to play at the giving moment. Someone was never able to search quickly "I want a cozy co-op game for me and my non-gamer partner, under $15, that runs on a low-end laptop."

This inspired me to create GenieEngine, an app that allows gamers to explore the ocean of beautiful games they would otherwise never find, while helping smaller game developers/companies get discovered.

Throughout the development, I have tested GenieEngine against modern LLMs like ChatGPT, I learned some important lessons on improving GenieEngine:

- Although LLMs nowadays are really powerful, and they can often take you straight to the games you described, they are still generalisations and usually biased towards popular, already well-known titles. GenieEngine uses Retrieval-Augmented Generation (RAG) which gives more grounded and statistically supported answers. Often you can find more niche yet still high quality games - "hidden gems" with GenieEngine.
- GenieEngine uses a "State of Thoughts" architecture, preserving the search states for conversations, unlike other usual LLMs they usually preserve the entire conversations. Since GenieEngine is a strict game searcher, preserving "State of Thoughts" for its previously recommended games is sufficient for the next output. This approach reduces token sizes and hallucinations, and results in higher accuracy.

If you like video games give it a try and let me know your thoughts!
www.genieengine.ai


<div class="row">
    
</div>
