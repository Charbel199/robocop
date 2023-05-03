import os
import re


def get_next_number(directory):
    max_number = 0
    pattern = re.compile(r'^(\D*)(\d+)(\D*)(\.\w+)$')
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            try:
                number = int(match.group(2))
                if number > max_number:
                    max_number = number
            except ValueError:
                pass
    return max_number + 1
