import langchain
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import FastAPI
from langchain_community.llms import VLLM
import time
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache
import hashlib
import hashlib
import uvicorn
from huggingface_hub import login

app = FastAPI()

login("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")

cache_hit_status = {"last_hit": False}


def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)

    def check_cache_hit(prompt, *args, **kwargs):
        # Custom pre-embedding function to track cache hit
        result = get_prompt(prompt)

        # check if the prompt is in the cache
        if cache_obj.data_manager.get(result):
            cache_hit_status["last_hit"] = True
        else:
            cache_hit_status["last_hit"] = False

        return result

    cache_obj.init(
        pre_embedding_func=check_cache_hit,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


langchain.llm_cache = GPTCache(init_gptcache)
llm = VLLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=100,
    temperature=0.6,
    dtype="float16",
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/generate")
async def generateText(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    output = llm(prompt)

    from_cache = cache_hit_status["last_hit"]
    print("Generated text:", output, "| From cache:", from_cache)
    ret = {"response": output, "from_cache": from_cache}
    return JSONResponse(ret)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)
