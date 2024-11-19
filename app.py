import langchain
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from gptcache.embedding import Onnx
from langchain_community.cache import GPTCache
import hashlib
import uvicorn
from huggingface_hub import login
import torch
from transformers import pipeline

app = FastAPI()

# Initialize cache with similarity search
cache = Cache()
embedding = Onnx()


def init_gptcache(cache_obj: Cache, llm: str):
    # Initialize a semantic cache for similarity search
    init_similar_cache(cache_obj, similarity_threshold=0.75)


# Initialize the cache
init_gptcache(cache, "llama-3b")
# Set up LangChain to use our cache
langchain.llm_cache = GPTCache(init_gptcache)

login("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")

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/generate")
async def generateText(request: Request) -> JSONResponse:
    start_time = time.time()

    request_dict = await request.json()
    prompt = request_dict.pop("prompt")

    try:
        cached_answer = cache.import_data(prompt).value
        print(f"Cache hit! Found similar prompt. Using cached result: {cached_answer}")
        llmResponse = cached_answer
    except:
        print("Cache miss! Generating new text.")
        output = pipe(
            prompt, max_new_tokens=100, temperature=0.3, num_return_sequences=1
        )
        llmResponse = output[0]["generated_text"]

        # Store the result in cache
        cache.export_data(prompt, llmResponse)
        print("Stored new result in cache")

    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency: {latency} seconds")

    print("Generated text:", llmResponse)
    ret = {"response": llmResponse, "latency": latency}
    return JSONResponse(ret)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)
