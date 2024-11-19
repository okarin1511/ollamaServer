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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

login("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")

model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    max_new_tokens=100,
    temperature=0.2,
    tokenizer=tokenizer,
    do_sample=True,
    top_p=0.95,  # Use nucleus sampling to focus on top 95% of options
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    device_map="auto",
    return_full_text=False,
    clean_up_tokenization_spaces=True,
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

    print("PROMPT", prompt)

    formatted_prompt = f"[INST] {prompt} [/INST]"

    llmResponse = llm.invoke(formatted_prompt)

    # llmResponse = llm.invoke(formatted_prompt)
    print("Generated text:", llmResponse)

    end_time = time.time()
    latency = end_time - start_time
    print(f"Latency: {latency} seconds")

    ret = {"response": llmResponse, "latency": latency}
    return JSONResponse(ret)
