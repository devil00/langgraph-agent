---
title: GAIA Agent
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

**Project Overview**
Developed an Agentic RAG system using LangGraph that orchestrates a multi-step workflow combining retrieval and reasoning capabilities. The agent integrates multiple search tools (Wikipedia, Arxiv, web search via Tavily), mathematical operations, and a Supabase vector database for semantic similarity search and question retrieval. For databse setup, run *supabase_sql_setup.sql*

**Evaluation Process**
The project was evaluated using the GAIA benchmark, specifically testing against 20 questions extracted from the level 1 validation set. This rigorous evaluation measured the agent's ability to handle complex, multi-step reasoning tasks. Performance was assessed through automated scoring, providing detailed metrics including overall accuracy percentage and correct answer counts.


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

