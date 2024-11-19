import langchain
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
from gptcache import Cache, Config
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache
import hashlib
import uvicorn
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
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


class CachedHuggingFacePipeline(HuggingFacePipeline):
    def __init__(self, pipeline, cache=langchain.llm_cache):
        super().__init__(pipeline)
        self.cache = cache

    def call(self, prompt, **kwargs):
        cached_result = self.cache.lookup(prompt)
        if cached_result is not None:
            return cached_result
        else:
            output = super().call(prompt, **kwargs)
            self.cache.update(prompt, output)
            return output


llm = CachedHuggingFacePipeline(pipe)


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

    # Use max_new_tokens instead of max_length
    output = llm(prompt, max_new_tokens=100, temperature=0.2, num_return_sequences=1)

    llmResponse = output[0]["generated_text"]

    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency: {latency} seconds")

    print("Generated text:", llmResponse)
    ret = {"response": llmResponse, "latency": latency}
    return JSONResponse(ret)
