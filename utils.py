import random

class ColorCollection:
    def __init__(self, random_seed=27):
        random.seed(random_seed)

    @staticmethod
    def get_random_color():
        color_list = [
            "snow",
            "ghost white",
            "gainsboro",
            "old lace",
            "linen",
            "antique white",
            "papaya whip",
            "blanched almond",
            "bisque",
            "peach puff",
            "navajo white",
            "lemon chiffon",
            "mint cream",
            "azure",
            "alice blue",
            "lavender",
            "lavender blush",
            "misty rose",
            "turquoise", 
            "aquamarine", 
            "powder blue", 
            "sky blue", 
            "steel blue", 
            "cadet blue", 
            "deep sky blue", 
            "dodger blue", 
            "cornflower blue", 
            "medium aquamarine", 
            "medium turquoise", 
            "light sea green", 
            "medium sea green"
        ]
        return random.choice(color_list)