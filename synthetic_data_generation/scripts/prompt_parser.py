
import re
import spacy
from spacy.matcher import PhraseMatcher

SPATIAL_RELATIONS = {
    "left of": ["left of", "to the left of"],
    "right of": ["right of", "to the right of"],
    "in front of": ["in front of", "ahead of"],
    "behind": ["behind", "at the back of", "in back of"],
    "next to": ["next to", "beside", "near", "close to", "adjacent to"],
    "on top of": ["on top of", "on", "over"],
    "under": ["under", "below", "beneath", "underneath"],
    "above": ["above", "over"],
    "bottom": ["bottom", "bottom of", "at the bottom of"],
}


class PromptParser:
    def __init__(self):
        print("[INFO] Loading spaCy NLP model...")
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        patterns = [self.nlp.make_doc(phrase)
                    for phrases in SPATIAL_RELATIONS.values()
                    for phrase in phrases]
        self.matcher.add("SPATIAL_RELATION", patterns)

    def extract_size_keyword(self, prompt_lower):
        size_map = {
            "tiny": 20, "small": 40, "medium": 80,
            "large": 120, "huge": 160, "gigantic": 200
        }
        for word, size in size_map.items():
            if word in prompt_lower:
                return size
        return None

    def extract_size_modifier(self, prompt_lower):
        for word in ["tiny", "small", "medium", "large", "huge", "gigantic"]:
            if word in prompt_lower:
                return word
        return None

    def extract_relationship(self, prompt: str):
        if not prompt:
            return {}

        prompt_lower = prompt.lower()
        doc = self.nlp(prompt_lower)
        matches = self.matcher(doc)

        if not matches:
            raise ValueError("No known spatial relation found in prompt.")

        match_spans = [doc[start:end] for _, start, end in matches]
        relation_span = max(match_spans, key=len)
        relation_text = relation_span.text

        relation_key = next(
            (key for key, phrases in SPATIAL_RELATIONS.items() if relation_text in phrases),
            relation_text
        )

        print("Noun chunks:")
        for chunk in doc.noun_chunks:
            print(f"  '{chunk.text}' ({chunk.start}, {chunk.end})")
        print(f"Relation '{relation_text}' span: ({relation_span.start}, {relation_span.end})")

        target_obj = None
        reference_obj = None


        before_chunks = [chunk for chunk in doc.noun_chunks if chunk.end <= relation_span.start]
        if before_chunks:
            target_obj = max(before_chunks, key=lambda c: c.end).root.text
        else:
            candidates = [chunk for chunk in doc.noun_chunks if chunk.end < relation_span.start + 2]
            if candidates:
                target_obj = max(candidates, key=lambda c: c.end).root.text


        after_chunks = [chunk for chunk in doc.noun_chunks if chunk.start >= relation_span.end]
        if after_chunks:
            reference_obj = min(after_chunks, key=lambda c: c.start).root.text
        else:
            candidates = [chunk for chunk in doc.noun_chunks if chunk.start > relation_span.end - 2]
            if candidates:
                reference_obj = min(candidates, key=lambda c: c.start).root.text

        nouns = [chunk.root.text for chunk in doc.noun_chunks]
        if not target_obj and nouns:
            target_obj = nouns[0]
        if not reference_obj and len(nouns) > 1:
            reference_obj = nouns[-1]

        if not target_obj or not reference_obj:
            raise ValueError("Unable to identify both target and reference objects.")

        is_text_prompt = any(kw in prompt_lower for kw in ["write", "text", "say", "display", "label"])
        font_size = None
        bold = None
        font_color = None
        size_modifier = None

        if is_text_prompt:
            size_match = re.search(r'font size (\d+)', prompt_lower)
            if size_match:
                font_size = int(size_match.group(1))
            else:
                size_kw = self.extract_size_keyword(prompt_lower)
                if size_kw:
                    font_size = size_kw
                    for k, v in {
                        "tiny": 20, "small": 40, "medium": 80,
                        "large": 120, "huge": 160, "gigantic": 200
                    }.items():
                        if v == size_kw:
                            size_modifier = k
                            break

            bold = "bold" in prompt_lower

            if "red" in prompt_lower:
                font_color = (0, 0, 255)
            elif "green" in prompt_lower:
                font_color = (0, 255, 0)
            elif "blue" in prompt_lower:
                font_color = (255, 0, 0)
            elif "black" in prompt_lower:
                font_color = (0, 0, 0)
            else:
                font_color = (0, 0, 0)


        orientation = "horizontal"
        orientation_keywords = {
        "vertical": ["vertical"],
        "rotated_90": ["rotate 90", "rotated 90", "rotate ninety", "rotate 90 degrees"],
        "rotated_180": ["rotate 180", "rotated 180", "upside down"],
        "rotated_270": ["rotate 270", "rotated 270", "rotate two hundred seventy"],
        "angled": ["angled", "slanted", "diagonal"],
        "tilted": ["tilted", "leaning"]
    }


        for ori, keywords in orientation_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                orientation = ori
                break

        size_modifier = self.extract_size_modifier(prompt_lower)

        return {
            "target_object": target_obj,
            "reference_object": reference_obj,
            "spatial_relation": relation_key,
            "font_size": font_size,
            "bold": bold,
            "font_color": font_color,
            "orientation": orientation,
            "target_is_text": bool(is_text_prompt),
            "size_modifier": size_modifier
        }


# ---------------------------------------------------------
# Prompt Testing Block
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = PromptParser()

    test_prompts = [
        "Place the cup to the left of the laptop with font size 100 and bold red text.",
        "Put the bottle on the table with font size 50 green.",
        "Set the phone next to the keyboard in bold blue font size 120.",
        "Add the flower pot behind the sofa.",
        "Put the vase close to the lamp with font size 60.",
        "Place the book beneath the shelf bold.",
        "Set the plate above the table blue font size 90.",
        "Put the text bold black vertically on the vase",
        "Write Hello rotated 90 degrees on the wall",
        "Write text upside down on the wall",
        "Put the apple horizontal on the vase",
        "Put the text bold black vertically on the vase",
        "Write large text on the cup",
        "Display small text next to the phone",
        "Say medium font size on the book",
        "Put tiny text under the table",
        "Set huge font size on the wall",
        "Put a gigantic donut beside the cake",
        "Place a medium apple beside the vase",
        "Place a medium tie rotate 180 of cup"
    ]

    for prompt in test_prompts:
        try:
            result = parser.extract_relationship(prompt)
            print(f"Prompt: {prompt}")
            print(f"Parsed: {result}")
        except ValueError as e:
            print(f"Prompt: {prompt} - Error: {e}")
        print("-" * 40)
