from extract import extract

PROMPT = "/// Add two numbers.\nfn add(a: isize, b: isize) -> isize {"
BODY = PROMPT + "        a + b\n"


def test_takes_body_from_code_block():
    reply = "Here you go:\n```rust\nfn add(a: isize, b: isize) -> isize {\n    a + b\n}\n```\nDone."
    assert extract(PROMPT, reply) == BODY


def test_moves_other_items_into_body_and_drops_doc_and_attributes():
    reply = ("```rust\nuse std::cmp::max;\n\nfn twice(x: isize) -> isize {\n    x * 2\n}\n\n"
             "/// Add two numbers.\n#[inline]\nfn add(a: isize, b: isize) -> isize {\n    twice(a) + b\n}\n```")
    assert extract(PROMPT, reply) == (PROMPT + "        use std::cmp::max;\n\n        fn twice(x: isize) -> isize {\n"
                                      "            x * 2\n        }\n\n        twice(a) + b\n")


def test_skips_think_block():
    reply = ("<think>```rust\nfn add(a: isize, b: isize) -> isize { 0 }\n```</think>"
             "```rust\nfn add(a: isize, b: isize) -> isize {\n    a + b\n}\n```")
    assert extract(PROMPT, reply) == BODY


def test_drops_closing_brace_of_bare_body():
    assert extract(PROMPT, "    a + b\n}\n") == BODY
