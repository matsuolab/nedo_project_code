def char_is_hiragana(c):
    return u'\u3040' <= c <= u'\u309f'


def contains_hiragana(s):
    return any(char_is_hiragana(c) for c in s)


def check(s):
    if not contains_hiragana(s):
        return ""

    else:
        return s
