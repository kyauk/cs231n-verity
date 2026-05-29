"""Module 5: Judge UI — FastAPI server for blinded human proposal rating.

Unlike the library modules, this one is *run*, not imported:

    uvicorn pipeline.modules.judge_ui.server:app --port 8001

The FastAPI `app` lives in pipeline.modules.judge_ui.server; configuration is
read from the environment in pipeline.modules.judge_ui.config. Other modules
consume the Judge UI over HTTP (its GET /judge/ratings/export endpoint), never
by importing it — so nothing is re-exported here, which also keeps FastAPI off
the import path of code that doesn't need it.
"""
