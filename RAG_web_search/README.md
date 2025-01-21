# Using Web Search for RAG

This example application demonstrates how to use results from a web search with RAG. The application can be used with any search engine or LLM provided the code is modified for the search engine and LLM.

To run the application

```
pip install -r requirements.txt
streamlit run app.py
```
To run the application on Windows

```
1) make sure python version is above 3.10
2) pip install -r requirements.txt
3) playwright install

4) Update app.py import part
    import asyncio
    import sys
    # ProactorEventLoopPolicy set for Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    import streamlit as st
    import ollama
    from openai import OpenAI
    from langchain_community.document_loaders import AsyncChromiumLoader
    from langchain_community.document_transformers import BeautifulSoupTransformer
    from duckduckgo_search import DDGS
    import re

5) streamlit run app.py
```

Code explanation at [dev.to](https://dev.to/spara_50/rag-with-web-search-2c3e).
