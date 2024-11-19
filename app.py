import langchain
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache
import hashlib
import uvicorn
from huggingface_hub import login
import torch
from transformers import pipeline

app = FastAPI()

login("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")

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


langchain.llm_cache = GPTCache(init_gptcache)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/generate")
async def generateText(request: Request) -> JSONResponse:
    start_time = time.time()

    request_dict = await request.json()
    prompt = request_dict.pop("prompt")

    # Check if the result is cached
    cache_manager = langchain.llm_cache.cache

    # Check if the result is cached
    cached_result = cache_manager.get(prompt)

    if cached_result:
        # If cached, use the cached result
        llmResponse = cached_result
        print("Cache hit! Using cached result.", llmResponse)
    else:
        # Otherwise, generate new text and store it in the cache
        print("Cache miss! Generating new text.")
        output = pipe(
            prompt, max_new_tokens=100, temperature=0.3, num_return_sequences=1
        )
        llmResponse = output[0]["generated_text"]
        # Store the result in the cache
        cache_manager.set(prompt, llmResponse)

    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency: {latency} seconds")

    print("Generated text:", llmResponse)
    ret = {"response": llmResponse, "latency": latency}
    return JSONResponse(ret)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)
