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


def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


langchain.llm_cache = GPTCache(init_gptcache)
llm = VLLM(
    model="meta-llama/Llama-3.2-1B",
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=50,
    temperature=0.6,
    dtype="float16",
    gpu_memory_utilization=1.0,  # Increase utilization
    max_model_len=4096,  # Adjust to a lower value
    max_seq_len=4096,  # Match with reduced `max_model_len`
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/generate")
async def generateText(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    print(prompt)
    output = llm(prompt)
    print("Generated text:", output)
    ret = {"text": output}
    return JSONResponse(ret)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)
