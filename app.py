from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
from gptcache import Cache, Config
from gptcache.manager.factory import manager_factory
from gptcache.adapter.api import get, put
from gptcache.embedding import Onnx
from langchain_community.cache import GPTCache
import hashlib
import uvicorn
from huggingface_hub import login
import torch
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()


# Create a function to get cache instance
def get_cache():
    cache = Cache()
    embedding = Onnx()
    cache_dir = "cache_similarity_search"

    cache.init(
        embedding_func=embedding.to_embeddings,
        data_manager=manager_factory(
            manager="map",
            data_dir=cache_dir,
            vector_params={"dimension": 768, "similarity_threshold": 0.85},
        ),
        config=Config(similarity_threshold=0.75),
    )
    return cache


# Initialize cache at module level
cache_instance = get_cache()


# Initialize the model
def init_model():
    login("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")
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )


# Initialize the model at module level
pipe = init_model()


@app.on_event("startup")
async def startup_event():
    """Initialize cache and model when the application starts"""
    global cache_instance
    if not cache_instance:
        cache_instance = get_cache()

    # Set up LangChain to use our cache
    def init_gptcache():
        return cache_instance

    import langchain

    langchain.llm_cache = GPTCache(init_gptcache)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/generate")
async def generateText(request: Request) -> JSONResponse:
    start_time = time.time()

    try:
        request_dict = await request.json()
        prompt = request_dict.pop("prompt")

        # Try to get from cache
        cached_result = get(prompt)

        if cached_result is not None:
            llmResponse = cached_result
            print(
                f"Cache hit! Found similar prompt. Using cached result: {llmResponse}"
            )
        else:
            print("Cache miss! Generating new text.")
            output = pipe(
                prompt, max_new_tokens=100, temperature=0.3, num_return_sequences=1
            )
            llmResponse = output[0]["generated_text"]

            # Store the result in cache
            put(prompt, llmResponse)
            print("Stored new result in cache")

        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency: {latency} seconds")

        ret = {"response": llmResponse, "latency": latency}
        return JSONResponse(ret)

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"An error occurred: {str(e)}"}
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)
