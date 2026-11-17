"""D13: does a method note help the model reason? Ceiling test: the type of each problem is given by the benchmark.

Two kinds of 'perfect' note, shown through the memory (note, 256 masked tokens, prompt), thinking enabled:
    card       a short written method for this type of problem (below)
    exemplars  worked examples of the same type (official chain-of-thought exemplars for BIG-Bench Hard, training
               problems with their solutions for MATH and GSM8K)

Compared with no note. A fourth condition, no note with the model's default repetition penalty (1.1, which every
earlier run inherited), measures what that setting costs on reasoning.

Benchmark questions are not written to the result file (BIG-Bench Hard asks that its data stay out of corpora)."""
import json
import random
import re
import time
import urllib.request

GAP = 256
BBH_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main"

CARDS = {
    "logical_deduction_five_objects": "List the objects. Rewrite every statement as a relation (A is left of B, A is older than B, A is third from the left). Start from the fixed positions, then place the others using the relations; when several placements are possible, try each and discard those that break a statement. Write the full order and check every statement against it. Only then read off what the question asks.",
    "tracking_shuffled_objects_five_objects": "Write a table 'person: item' with the starting assignment. Process the swaps one at a time, in the order given: after each swap, rewrite the two lines that changed. Never skip a swap and never do two at once. At the end, read the line of the person asked about.",
    "web_of_lies": "Go through the people in the order given, keeping one fact per person: tells the truth, or lies. If Y says X tells the truth, Y is truthful exactly when X is truthful. If Y says X lies, Y is truthful exactly when X lies. Carry the result forward to the last person. Answer Yes if that person tells the truth, No otherwise.",
    "boolean_expressions": "Evaluate the innermost parentheses first. Priority: not, then and, then or. Replace each sub-expression by True or False and rewrite the shorter expression at every step. not True = False. A and B is True only if both are True. A or B is False only if both are False.",
    "formal_fallacies": "Translate each premise into a rule (all A are B; no A is B; some A is B; X is A or B). Translate the conclusion the same way. The argument is valid only if the conclusion holds in every situation where all premises hold. Try to build a counterexample, a situation where the premises are true and the conclusion false. If one exists, answer invalid; if none can exist, answer valid. Use no outside knowledge.",
    "navigate": "Track a position (x, y) starting at (0, 0) and a facing direction. 'Turn left', 'turn right' and 'turn around' change the direction only. 'Take N steps' moves N in the facing direction. When told to always face forward, steps left, right, backward and forward move along the matching axis without turning. Update after every instruction. Answer Yes if the final position is (0, 0), otherwise No.",
    "temporal_sequences": "List every interval during which the person was seen somewhere else: they are busy then. Note the waking time and the closing time of the place. The answer is the only interval between waking up and closing that no busy interval covers. Choose the option equal to that free interval.",
    "date_understanding": "First fix today's date from the statement, being careful with phrases such as 'yesterday was' or 'in two days it will be'. Then apply the shift asked for (a week ago, ten days later, one month from now), carrying across month and year ends with the right number of days per month; February has 29 days in a leap year. Write the result as MM/DD/YYYY and match it to an option.",
    "object_counting": "Write the items as a list with their quantities ('a' or 'an' counts as 1). Keep only the items that belong to the category asked for. Add the quantities one by one, showing the running total. Answer with the final number.",
    "causal_judgement": "These questions ask how a typical person would answer. Identify the outcome and the action in question. People usually say an action caused the outcome when the outcome would not have happened without it, especially when the action broke a rule; if two actions were both needed, the one that broke the rule is blamed. For 'intentionally': people say Yes when the agent knew a harmful side effect would follow and went ahead, No for a helpful side effect the agent did not care about. Answer Yes or No.",
    "penguins_in_a_table": "Rewrite the table with one line per animal and its name, age, height and weight. Apply any stated change (a line added or deleted) before answering. Then filter or sort on the column the question is about, checking each line in turn, and count or pick the matching line.",
    "reasoning_about_colored_objects": "Write the objects in the order given, each with its colour, numbered from left to right. Apply any stated removal. Then read the answer off the list: the position of an object, the colour of the object at a position, what is to the left or right of another, or how many objects are, or are not, a given colour.",
    "word_sorting": "Write the words one per line. Sort by the first letter; for words with the same first letter compare the second letter, then the third, and so on. A shorter word comes before a longer word that starts with it. Write the final list in order, separated by spaces, with every word exactly once.",
    "multistep_arithmetic_two": "Compute each parenthesis separately. Inside one, do the multiplications first, then additions and subtractions from left to right, watching the signs: subtracting a negative adds, and the product of two negatives is positive. Write every intermediate value. Then combine the results with the outer operation.",
    "algebra": "Name the unknown, translate the statement into equations, and isolate the unknown step by step, doing the same operation on both sides. For quadratics try factoring, otherwise use the quadratic formula; for systems use substitution or elimination. Check the solution in the original equation and against any condition (positive, integer). Give the exact value, simplified.",
    "counting_and_probability": "Decide whether order matters (arrangements, n!/(n-k)!) or not (combinations, n choose k). Split into disjoint cases and add; chain independent choices and multiply. For 'at least one', compute 1 minus the probability of none. Probability is favourable outcomes over equally likely outcomes. Watch for overcounting identical objects or symmetric arrangements. Simplify the final fraction.",
    "geometry": "Label the known lengths and angles. Look for right triangles (Pythagoras), similar triangles (equal ratios) and the standard formulas: triangle area is base times height over 2, circle area is pi r squared, circumference is 2 pi r, the angles of a triangle sum to 180 degrees, an inscribed angle is half the central angle. Set up one equation at a time. Keep the answer exact (radicals, pi) unless told otherwise.",
    "intermediate_algebra": "For polynomials use Vieta's formulas (sum and product of the roots), the remainder and factor theorems, and substitutions that lower the degree. For inequalities, find the zeros and test each interval; AM-GM gives the minimum of a sum with a fixed product. For complex numbers use the modulus and the conjugate. For functional equations, plug in simple values (0, 1, x = y). State domain restrictions and reject extraneous solutions.",
    "number_theory": "Factor the numbers into primes. The gcd takes the smallest exponents, the lcm the largest. For remainders work modulo n: reduce each term first, and find the cycle of the powers. The number of divisors is the product of (exponent + 1). To change base, divide repeatedly by the base. To solve ax = b mod n, use the inverse of a when gcd(a, n) = 1. Check with small cases.",
    "prealgebra": "Read what is asked and in which units. Follow the order of operations: parentheses, powers, multiplication and division, then addition and subtraction. For fractions, add with a common denominator, multiply straight across, and simplify. Turn percentages into decimals. Write ratios as fractions of the total. Estimate to check that the answer is reasonable.",
    "precalculus": "For trigonometry use the unit-circle values, sin squared plus cos squared equals 1, and the angle-addition and double-angle formulas. For vectors: a.b = |a||b|cos(theta), the cross product is perpendicular to both, and the projection of a on b is (a.b / b.b) b. For matrices: multiply row by column; the determinant of a 2 by 2 matrix is ad - bc. For complex numbers use the polar form and De Moivre's formula. Keep the answer exact.",
    "gsm8k": "List the quantities given, with their units, and what is asked. Compute one step at a time, writing each intermediate result and what it represents. Watch the words 'each', 'per', 'more than', 'fewer than', 'half', 'twice'. Check that every number in the statement was used or is irrelevant, then give the final number.",
}
BBH_TASKS = [k for k in CARDS if k not in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus", "gsm8k")]
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def boxed(text):
    """Content of the last \\boxed{...} in a text, or None."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def norm_math(s):
    s = s.strip().strip("$").strip()
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\ ", ""), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""), ("\\$", ""), ("dfrac", "frac")):
        s = s.replace(a, b)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mbox\{[^}]*\}", "", s)
    s = s.replace(" ", "").rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    s = re.sub(r"^[a-zA-Z]=", "", s)
    return s


def norm_plain(s):
    return re.sub(r"\s+", " ", s.lower()).strip(" *.()")


def run(bench: str, n_per_type: int = 40, budget_scale: float = 1.0, seed: int = 0):
    import torch
    from datasets import load_dataset

    from .common import load_model, load_tokenizer

    t_start = time.time()
    rng = random.Random(seed)
    tok = load_tokenizer()
    model = load_model()
    dev = "cuda"
    bos, pad, eos = tok.bos_token_id, tok.pad_token_id, tok.eos_token_id

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def get(url):
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "lfm3-memory-research/0.1"}), timeout=60) as r:
            return r.read().decode("utf-8")

    # ---- problems, by type, with their exemplars
    items = []  # {type, content, target, kind}
    exemplars = {}
    if bench == "bbh":
        for task in BBH_TASKS:
            data = json.loads(get(f"{BBH_URL}/bbh/{task}.json"))["examples"]
            rng.shuffle(data)
            for x in data[:n_per_type]:
                items.append({"type": task, "content": f"{x['input']}\n\nFinish with a line of the form 'Answer: X'.", "target": x["target"], "kind": "plain"})
            cot = get(f"{BBH_URL}/cot-prompts/{task}.txt").split("-----", 1)[-1].strip()
            parts = cot.split("\n\nQ: ")
            exemplars[task] = (parts[0] + "".join("\n\nQ: " + p for p in parts[1:3])).strip()  # task description + 2 worked examples
        budget, answer_budget, prefix = int(768 * budget_scale), 24, "Answer:"
    elif bench == "math":
        for subject in MATH_SUBJECTS:
            test = [x for x in load_dataset("EleutherAI/hendrycks_math", subject, split="test") if x["level"] in ("Level 3", "Level 4", "Level 5") and boxed(x["solution"])]
            train = [x for x in load_dataset("EleutherAI/hendrycks_math", subject, split="train") if x["level"] in ("Level 3", "Level 4", "Level 5") and len(x["solution"]) < 900]
            rng.shuffle(test)
            rng.shuffle(train)
            for x in test[:n_per_type]:
                items.append({"type": subject, "content": f"{x['problem']}\n\nPut the final answer in \\boxed{{}}.", "target": boxed(x["solution"]), "kind": "math"})
            exemplars[subject] = "\n\n".join(f"Problem: {x['problem']}\nSolution: {x['solution']}" for x in train[:2])
        budget, answer_budget, prefix = int(1536 * budget_scale), 48, "The final answer is \\boxed{"
    else:
        test = list(load_dataset("openai/gsm8k", "main", split="test"))
        train = [x for x in load_dataset("openai/gsm8k", "main", split="train") if len(x["answer"]) < 500]
        rng.shuffle(test)
        rng.shuffle(train)
        for x in test[: n_per_type * 4]:
            items.append({"type": "gsm8k", "content": f"{x['question']}\n\nFinish with a line of the form 'Answer: <number>'.", "target": x["answer"].split("####")[-1].strip().replace(",", ""), "kind": "number"})
        exemplars["gsm8k"] = "\n\n".join(f"Problem: {x['question']}\nSolution: {re.sub(r'<<[^>]*>>', '', x['answer']).replace('####', 'Answer:')}" for x in train[:2])
        budget, answer_budget, prefix = int(768 * budget_scale), 24, "Answer:"

    def note(kind, t):
        if kind == "card":
            text = f"Reference note: a method for this kind of problem. Use it if it helps.\n{CARDS[t]}"
        else:
            text = f"Reference note: worked examples of the same kind of problem. Use them if they help.\n{exemplars[t]}"
        return [bos] + enc(text)[:900]

    def chat_ids(content):
        ids = enc(tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False))
        return ids if ids[0] == bos else [bos] + ids

    close_think = enc("\n</think>\n\n")

    def generate(seqs, max_new, kw):
        width = max(len(x[0]) for x in seqs)
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, (x, m) in enumerate(seqs):
            ids[r, width - len(x) :], att[r, width - len(x) :] = torch.tensor(x), torch.tensor(m)
        chunk = max(64, 130_000 // len(seqs))
        extra = {"prefill_chunk_size": chunk} if width > chunk else {}
        with torch.inference_mode():
            out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, **kw, **extra)
        return [out[r, width:].tolist() for r in range(len(seqs))]

    def solve(condition, batch=140):
        torch.manual_seed(seed)
        kw = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.1 if condition == "no_note_default_penalty" else 1.0}
        st = []
        for it in items:
            block = note(condition, it["type"]) if condition in ("card", "exemplars") else None
            prompt = chat_ids(it["content"])
            st.append({"block": block, "prompt": prompt, "gen": [], "think_open": "<think>" in tok.decode(prompt[-6:])})

        def seq(s):
            tail = s["prompt"] + s["gen"]
            if s["block"] is None:
                return tail, [1] * len(tail)
            return s["block"] + [pad] * GAP + tail, [1] * len(s["block"]) + [0] * GAP + [1] * len(tail)

        for i in range(0, len(st), batch):
            group = st[i : i + batch]
            for s, new in zip(group, generate([seq(s) for s in group], budget, kw)):
                s["done"] = eos in new
                s["gen"] = new[: new.index(eos)] if eos in new else new
        forced = [s for s in st if not s["done"] and "</think>" not in tok.decode(s["gen"], skip_special_tokens=True)]
        for s in forced:
            s["gen"] += close_think + enc(prefix)
            s["forced"] = True
        cut_in_answer = [s for s in st if not s["done"] and not s.get("forced")]
        for group_all in (forced, cut_in_answer):
            for i in range(0, len(group_all), batch):
                group = group_all[i : i + batch]
                for s, new in zip(group, generate([seq(s) for s in group], answer_budget, kw)):
                    s["gen"] += new[: new.index(eos)] if eos in new else new
        outs = []
        for s in st:
            text = tok.decode(s["gen"], skip_special_tokens=True)
            outs.append({"final": text.split("</think>")[-1] if "</think>" in text else text, "tokens": len(s["gen"]), "forced": bool(s.get("forced"))})
        return outs

    def correct(final, it):
        if it["kind"] == "math":
            got = boxed(final)
            return got is not None and norm_math(got) == norm_math(it["target"])
        tail = final.split("Answer:")[-1] if "Answer:" in final else final
        if it["kind"] == "number":
            nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
            try:
                return bool(nums) and abs(float(nums[0 if "Answer:" in final else -1].replace(",", "").rstrip(".")) - float(it["target"])) < 1e-6
            except ValueError:
                return False
        got, want = norm_plain(tail.strip().split("\n")[0]), norm_plain(it["target"])
        if len(want) == 1 and want.isalpha():  # multiple choice letter
            return got[:1] == want and (len(got) == 1 or not got[1].isalnum())
        return got == want or got.startswith(want + " ") or got.startswith(want + ".") or got.startswith(want + ",")

    result = {"bench": bench, "n": len(items), "thinking_budget": budget, "conditions": {}, "by_type": {}}
    oks = {}
    for condition in ("no_note", "no_note_default_penalty", "card", "exemplars"):
        outs = solve(condition)
        ok = [correct(o["final"], it) for o, it in zip(outs, items)]
        oks[condition] = ok
        result["conditions"][condition] = {
            "accuracy": round(sum(ok) / len(items), 4), "thinking_cut_at_budget": round(sum(o["forced"] for o in outs) / len(items), 4),
            "mean_tokens": round(sum(o["tokens"] for o in outs) / len(items)),
        }
    for condition in ("no_note_default_penalty", "card", "exemplars"):
        result["conditions"][condition]["fixed"] = sum(a and not b for a, b in zip(oks[condition], oks["no_note"]))
        result["conditions"][condition]["broken"] = sum(b and not a for a, b in zip(oks[condition], oks["no_note"]))
    for t in sorted({it["type"] for it in items}):
        idx = [k for k, it in enumerate(items) if it["type"] == t]
        result["by_type"][t] = {"n": len(idx), **{c: round(sum(oks[c][k] for k in idx) / len(idx), 3) for c in oks}}
    result["note_tokens"] = {"card_mean": round(sum(len(note("card", t)) for t in exemplars) / len(exemplars)), "exemplars_mean": round(sum(len(note("exemplars", t)) for t in exemplars) / len(exemplars))}
    result["seconds_total"] = round(time.time() - t_start)
    return result
