import re
from lxml import etree as ET


def remove_xref_and_wrapped_brackets(text, proximity=10):
    try:
        # Parse the text within a root element to ensure proper structure
        root = ET.fromstring(f"<root>{text}</root>")

        # Iterate over xref elements and remove them with their brackets
        xrefs = list(root.xpath(".//xref"))
        for xref in xrefs:
            parent = xref.getparent()
            if parent is None:
                continue

            full_text = (
                ET.tostring(parent, method="text", encoding="unicode", with_tail=False)
                or ""
            )

            # Calculate positions relative to the parent text
            xref_start = full_text.find(xref.text) if xref.text else -1
            xref_end = xref_start + len(xref.text) if xref.text else -1

            # Extract text slices around the xref to check for brackets
            pre_slice = (
                full_text[max(0, xref_start - proximity) : xref_start]
                if xref_start != -1
                else ""
            )
            post_slice = (
                full_text[xref_end : xref_end + proximity] if xref_end != -1 else ""
            )

            # Regex to find the closest brackets around xref
            pre_bracket_match = (
                re.search(r"(\(|\[)[^\(\)\[\]]*$", pre_slice) if pre_slice else None
            )
            post_bracket_match = (
                re.search(r"^[^\(\)\[\]]*(\)|\])", post_slice) if post_slice else None
            )

            if pre_bracket_match and post_bracket_match:
                pre_bracket = pre_bracket_match.group(1)
                post_bracket = post_bracket_match.group(1)

                # Check if the brackets are matching pairs
                if (pre_bracket, post_bracket) in [("(", ")"), ("[", "]")]:
                    # If brackets match, adjust text slices to remove brackets
                    pre_bracket_index = pre_slice.rfind(pre_bracket)
                    post_bracket_index = post_slice.find(post_bracket)

                    new_pre_slice = (
                        pre_slice[:pre_bracket_index]
                        + pre_slice[pre_bracket_index + 1 :]
                    )
                    new_post_slice = (
                        post_slice[:post_bracket_index]
                        + post_slice[post_bracket_index + 1 :]
                    )

                    # Update the parent text to exclude the xref and the brackets
                    parent.text = (
                        (parent.text or "")[: max(0, xref_start - proximity)]
                        + new_pre_slice
                        + (parent.text or "")[xref_start:]
                        + new_post_slice
                    )
            else:
                # If no matching brackets, clean the xref text only
                if xref.tail:
                    xref.tail = xref.text + xref.tail if xref.text else xref.tail
                else:
                    if xref.getprevious() is not None:
                        xref.getprevious().tail = (xref.getprevious().tail or "") + (
                            xref.text or ""
                        )
                    else:
                        parent.text = (parent.text or "") + (xref.text or "")

            # Remove the xref element from the tree
            parent.remove(xref)

        # Serialize the modified XML tree back to a string, preserving all tags
        cleaned_text = ET.tostring(root, encoding="unicode", method="xml")
        cleaned_text = "".join(cleaned_text.splitlines())[
            6:-7
        ].strip()  # Remove artificial root element
        return cleaned_text

    except Exception as e:
        print(f"An error occurred: {e}")
        # Return original text if parsing fails
        return text


def remove_specific_tags(element, remove_list):
    # この要素自体が削除対象であれば、何も返さない
    if element.tag in remove_list:
        return

    # 子要素を探索
    if hasattr(element, "__iter__"):
        for e in list(element):
            if e.tag in remove_list:
                element.remove(e)
            else:
                yield from remove_specific_tags(e, remove_list)

    # 要素のテキストを返す
    if element.text:
        yield element.text

    # 要素のテイルを返す
    if element.tail:
        yield element.tail


def remove_unnecessary_elements(xml_string, remove_list):
    try:
        parser = ET.XMLParser(recover=True)
        root = ET.fromstring(xml_string, parser=parser)
        # 抽出されたテキストを結合して返す
        return "".join(remove_specific_tags(root, remove_list))
    except ET.ParseError as e:
        print(f"Failed to parse XML: {e}")
        return ""


def remove_incomplete_sentence_at_start(text):
    text = text.lstrip()
    # Continue until text starts with an uppercase letter
    while text and not text[0].isupper():
        position = text.find(". ")
        if position == -1:
            return ""  # No valid starting point found
        text = text[position + 2 :]
        text = text.lstrip()
    return text


def remove_dirty_text(text):
    # 空の丸括弧 () を除去
    text = re.sub(r"\(\s*\)", "", text)

    # 丸括弧内にセミコロン ; またはカンマ , が含まれる場合のテキストを除去
    text = re.sub(r"\([^)]*[;,][^)]*\)", "", text)

    # 丸括弧内に 'Fig' という文字列が含まれる場合のテキストを除去
    text = re.sub(r"\([^)]*Fig[^)]*\)", "", text)

    # 丸括弧内に 'al' という文字列が含まれる場合のテキストを除去
    text = re.sub(r"\([^)]*al[^)]*\)", "", text)

    # 空の角括弧 [] を除去
    text = re.sub(r"\[\s*\]", "", text)

    # 角括弧内にセミコロン ; またはカンマ , が含まれる場合のテキストを除去
    text = re.sub(r"\[[^]]*[;,][^]]*\]", "", text)

    return text


def extract_abstract_and_body_text(xml_string):
    try:
        parser = ET.XMLParser(recover=True)
        root = ET.fromstring(xml_string, parser=parser)

        # Extracting abstract and body elements
        abstract_elements = root.xpath(".//front/article-meta/abstract")
        body_elements = root.xpath(".//body")

        # Using ET.tostring to keep the text with inner tags before cleaning
        abstract_text = " ".join(
            ET.tostring(element, method="xml", encoding="unicode")
            for element in abstract_elements
        )
        body_text = " ".join(
            ET.tostring(element, method="xml", encoding="unicode")
            for element in body_elements
        )

        # Removing xref and text in brackets
        abstract_text = remove_xref_and_wrapped_brackets(abstract_text)
        body_text = remove_xref_and_wrapped_brackets(body_text)

        # Removing unnecessary elements
        remove_list = [
            "title",
            "xref",
            "fig",
            "table-wrap",
            "inline-formula",
            "disp-formula",
        ]
        abstract_text = remove_unnecessary_elements(
            f"<root>{abstract_text}</root>", remove_list
        )
        body_text = remove_unnecessary_elements(
            f"<root>{body_text}</root>", remove_list
        )

        # Remove dirty text
        abstract_text = remove_dirty_text(abstract_text)
        body_text = remove_dirty_text(body_text)

        # Remove incomplete sentence at the start if needed
        abstract_text = remove_incomplete_sentence_at_start(abstract_text)
        body_text = remove_incomplete_sentence_at_start(body_text)

        return abstract_text, body_text
    except ET.ParseError as e:
        print(f"Failed to parse XML: {e}")
        return "", ""


def generate_record(xml_string):
    abstract_text, body_text = extract_abstract_and_body_text(xml_string)
    return f"{abstract_text} {body_text}".strip()
