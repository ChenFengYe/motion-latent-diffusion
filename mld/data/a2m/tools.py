import os
import string


def parse_info_name(path):
    name = os.path.splitext(os.path.split(path)[-1])[0]
    info = {}
    current_letter = None
    for letter in name:
        if letter in string.ascii_letters:
            info[letter] = []
            current_letter = letter
        else:
            info[current_letter].append(letter)
    for key in info.keys():
        info[key] = "".join(info[key])
    return info


