########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import types
import copy
import torch
from torch.nn import functional as F


from model import RWKV_RNN

np.set_printoptions(precision=4, suppress=True, linewidth=200)

##############################################################################################################

model = RWKV_RNN()


def sample_logits(out, temperature=1.0, top_p=None):
    probs = F.softmax(torch.tensor(out), dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)

    return torch.multinomial(probs, num_samples=1)[0]


def infer(
    prompt,
    trials=1,
    length_min=1,
    length_max=333,
    seperators=[".", "!", "?", "\n"],
    temperature=1.0,
    top_p=0.7,
):
    results = ["" for x in range(trials)]
    for trial in range(trials):
        ctx = [model.tokenizer.encode(prompt)][0]
        src_len = len(ctx)
        model.clear()

        if trial == 0:
            init_state = types.SimpleNamespace()
            for i in range(src_len):
                x = ctx[: i + 1]
                if i == src_len - 1:
                    init_state.out = model.run(x)
                else:
                    model.run(x)
                    model.save(init_state)
        else:
            model.load(init_state)

        out = init_state.out
        for i in range(src_len, src_len + length_max):
            x = ctx[: i + 1]
            x = x[-model.ctx_len :]

            if i == src_len:
                out = copy.deepcopy(init_state.out)
            else:
                out = model.run(x)

                out[0] = -999999999  # disable <|endoftext|>

                char = sample_logits(out, temperature=temperature, top_p=top_p)
                char = char.item()
                decoded = model.tokenizer.decode(char)
                results[trial] += decoded
                if (
                    seperators
                    and src_len + length_min < i
                    and any(sep in decoded for sep in seperators)
                ):
                    break
                ctx += [char]
    return results
