from tqdm.notebook import tqdm, trange
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np


def naive_masking_score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input.cuda(), labels=labels.cuda()).loss
    return np.exp(loss.item())


def naive_t5_score(model, tokenizer, sentence):
    mask_token_id = tokenizer.encode("<extra_id_0>")[0]
    mask_token_id_1 = tokenizer.encode("<extra_id_1>")[0]

    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 1, 1)
    mask = torch.ones(tensor_input.size(-1)).diag()[:-1]
    masked_input = repeat_input.masked_fill(mask == 1, mask_token_id)

    n_seq = tensor_input[0][:-1].size(0)
    prefix = torch.full((n_seq,), mask_token_id)
    postfix = torch.full((n_seq,), mask_token_id_1)
    end = torch.full((n_seq,), 1)
    labels = torch.vstack([prefix, tensor_input[0][:-1], postfix, end]).T.contiguous()

    with torch.inference_mode():
        loss = model(masked_input.cuda(), labels=labels.cuda()).loss
    return loss.item()


def naive_gpt2_score(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return loss.item()


def naive_model_scores(model, tokenizer, sentences, scorer):
    scores = []
    for sent in tqdm(sentences):
        scores.append(scorer(model, tokenizer, sent))
    return np.array(scores)


def crows_pipeline(model, tokenizer, good_sentences, bad_sentences, scorer):
    bad_scores = naive_model_scores(model, tokenizer, bad_sentences, scorer)
    good_scores = naive_model_scores(model, tokenizer, good_sentences, scorer)

    return (bad_scores < good_scores).mean()

def stereo_pipeline(model, tokenizer, scorer, good, bad, unrelated):
    good_scores = naive_model_scores(model, tokenizer, good, scorer)
    bad_scores = naive_model_scores(model, tokenizer, bad, scorer)
    unrelated_scores = naive_model_scores(model, tokenizer, unrelated, scorer)

    lms = (good_scores < unrelated_scores).mean() / 2 + (bad_scores < unrelated_scores).mean() / 2
    ss = (bad_scores < good_scores).mean()
    icat = lms * min(ss, 1. - ss) * 2
    return {"lms": lms, "ss": ss, "icat": icat}


def winobias_pipeline(model, tokenizer, wb_data, scorer):
    result = {}
    for side in ["pro", "anti"]:
        good_scores = naive_model_scores(model, tokenizer, wb_data[side]["good"], scorer)
        bad_scores = naive_model_scores(model, tokenizer, wb_data[side]["bad"], scorer)
        result[side] = np.mean(good_scores < bad_scores)
    return result
