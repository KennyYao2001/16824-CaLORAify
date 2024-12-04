import json

def parse_ingredients(s: str):
    number_characters = "0123456789/ "
    now = 0
    result = []
    while now < len(s):
        number = ""
        for i in range(now, len(s)):
            if s[i] not in number_characters:
                number = s[now:i]
                break
        if number == "":
            now = len(s)
            continue
        ingred = ""
        for i in range(now+len(number), len(s)):
            if s[i] in number_characters[:-1]:
                ingred = s[now+len(number):i]
                break 
        if ingred == "":
            ingred = s[now+len(number):]
        if ingred == "":
            now = len(s)
            continue
        now = now + len(number) + len(ingred)
        number = number.strip()
        number = number.replace(" ", "+")
        try:
            number = eval(number)
        except SyntaxError:
            continue
        ingred = ingred.strip(" ,.")
        ingred = ingred.replace("of ", "")
        result.append((number, ingred))
    return result

if __name__ == "__main__":

    with open("test_result.json", "r") as f:
        output = json.load(f)

    for idx in range(90, 110):
        print(idx)
        print(output[idx]["image_path"])
        print(output[idx]["GT"])
        print(parse_ingredients(output[idx]["pred"]))
        print()


