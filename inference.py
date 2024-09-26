from transformers import AutoModelForCausalLM, AutoTokenizer

def main(prompt: str = "I love", model: str = "./llama_model", tokenizer: str=None):
    if tokenizer==None:
        tokenizer = model
    model = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_new_tokens=16, pad_token_id=tokenizer.eos_token_id)
    res = tokenizer.decode(output[0])
    print(res)

if __name__ == "__main__":
    import fire
    fire.Fire(main)