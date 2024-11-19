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
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoConfig
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

llm = HuggingFacePipeline(pipeline=pipe)


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

    # output = llm.invoke(
    #     [prompt],
    #     max_new_tokens=1000,
    #     max_length=None,
    #     temperature=0.3,
    #     generate_kwargs={
    #         "max_new_tokens": 1000,
    #         "max_length": None,
    #     },  # Ensure no max_length interference
    # )

    # output = llm.generate(
    #     [prompt],
    #     max_new_tokens=1000,
    #     max_length=None,
    #     temperature=0.3,  # Ensure no max_length interference
    # )

    output = [{"generated_text": prompt}]

    print(f"Hit: {prompt}")

    config = AutoConfig.from_pretrained(model_id)
    print(f"Model's max position embeddings: {config.max_position_embeddings}")

    llmResponse = output[0]["generated_text"]

    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency: {latency} seconds")

    print("Generated text:", llmResponse)
    ret = {"response": llmResponse, "latency": latency}
    return JSONResponse(ret)
