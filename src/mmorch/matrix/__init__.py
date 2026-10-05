"""The earlier model x backend prototype: three domain models served by vLLM, TGI or TensorRT, behind `mmorch serve`.

It needs the 'matrix' extra, plus 'matrix-ml' for the DistilBERT domain classifier (without it, prompts are
classified by keywords). Importing this package loads none of fastapi, uvicorn, aiohttp, pydantic, torch or
transformers.
"""
