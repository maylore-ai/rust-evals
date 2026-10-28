import re

FENCE = re.compile(r"```[A-Za-z0-9_+-]*[^\n]*\n(.*?)^[ \t]*```", re.S | re.M)
THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
OUTER = re.compile(r"[ \t]*(///|#\[)[^\n]*\n")


def function_name(prompt):
    return re.findall(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", prompt)[-1]


def indent(code):
    return "".join("    " + line if line.strip() else line for line in code.splitlines(keepends=True))


def matching_brace(code, start):
    depth = 0
    for i in range(start, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def body_and_items(code, name):
    match = re.search(rf"\bfn\s+{name}\b", code)
    if not match:
        return None
    start = code.find("{", match.end())
    end = matching_brace(code, start) if start >= 0 else None
    if end is None:
        return None
    line_start = code.rfind("\n", 0, match.start()) + 1
    head = code[:line_start]
    while True:
        prev = head.rfind("\n", 0, len(head) - 1) + 1
        if not OUTER.fullmatch(head[prev:]):
            break
        head = head[:prev]
    return code[start + 1:end], head + code[end + 1:]


def extract(prompt, reply):
    reply = THINK.sub("", reply)
    match = FENCE.search(reply)
    code = match.group(1) if match else reply
    found = body_and_items(code, function_name(prompt))
    if found:
        body, items = found
        completion = indent(items.strip() + "\n") + body if items.strip() else body
    else:
        completion = code
        if completion.count("}") > completion.count("{"):
            completion = completion[:completion.rindex("}")]
    return prompt + indent(completion.strip("\n") + "\n")
