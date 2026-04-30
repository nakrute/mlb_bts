import requests
from bs4 import BeautifulSoup


def decode_secret_message(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    points = {}

    for row in soup.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]

        if len(cells) != 3:
            continue

        try:
            x = int(cells[0])
            char = cells[1]
            y = int(cells[2])
        except ValueError:
            continue

        if char == "░":
            char = " "

        points[(x, y)] = char

    max_x = max(x for x, y in points)
    max_y = max(y for x, y in points)

    for y in range(max_y + 1):
        line = ""
        for x in range(max_x + 1):
            line += points.get((x, y), " ")
        print(line)


decode_secret_message(
    "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
)
