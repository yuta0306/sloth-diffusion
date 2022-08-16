import base64
import os
import time
from typing import Dict, Final, Generator

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class Crawler:

    GOOGLE_SEARCH_URL: Final[str] = "https://www.google.com/search"
    GOOGLE_IMAGE_EVENT: Final[str] = "https://www.google.com/imgevent"
    session: requests.Session
    use_splash: bool

    def __init__(self, use_splash: bool = False) -> None:
        self.session = requests.session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"
            }
        )
        self.use_splash = use_splash

    def search(
        self, keyword: str, max_num: int = 100, start: int = 0
    ) -> Dict[str, Dict[str, str]]:
        print(f"Search `{keyword}` image for Google")

        res, total = {}, 0

        for i, param in enumerate(self.queries(keyword=keyword, start=start), 1):
            if self.use_splash:
                url = (
                    self.GOOGLE_SEARCH_URL
                    + "?"
                    + "&".join([key + "=" + value for key, value in param.items()])
                )
                response = self.session.get(
                    "http://localhost:8050/render.html",
                    params={"url": url, "wait": 2.0, "viewport": "full"},
                )
                html = response.text
            else:
                response = self.session.get(self.GOOGLE_SEARCH_URL, params=param)
                html = response.text
                time.sleep(2.0)

            # print(response.status_code)
            # print(response.headers)
            print(response.url)

            start = html.index("<")
            end = "".join(list(reversed(html))).index(">")
            html = html[start : len(html) - end]

            soup = BeautifulSoup(html, "lxml")

            elements = soup.select(".ivg-i")
            print("Elements:", len(elements))
            image_tags = [
                [elm.get("data-ved"), elm.select_one("img.rg_i")] for elm in elements
            ]
            print("Image Tags:", len(image_tags))

            def get_src(tag):
                if (src := tag.get("src")) is not None:
                    return src
                return tag.get("data-src")

            images = {
                id_: get_src(img)
                for [id_, img] in image_tags
                # if elm.get("data-ri") is not None and elm.get("data-id") is not None
            }
            # images = {key: data for key, data in images.items() if data is not None}

            if len(images) > 0:
                prev_total = total
                res.update(images)
                # total += len(images)
                total = len(res)
                if prev_total == total:
                    break
                if total > max_num:
                    # total = max_num
                    # res = res[:max_num]
                    break
            else:
                break
            if i % 2 == 0:
                print(f"{i} LOOPs: Fetched {total} images")

        print(f"Fetched {total} images")

        # postprocess
        res = {
            key: {
                "url": data if data.startswith("http") else None,
                "data": data if data.startswith("data:") else None,
            }
            for key, data in res.items()
        }
        return res

    def queries(self, keyword: str, start: int) -> Generator:
        page = start
        while True:
            # https://www.google.com/search?q=sloth&oq=sloth&tbm=isch&ijn=3&asearch=ichunk&async=_id:rg_s,_pms:s,_fmt:pc&sourceid=chrome&ie=UTF-8
            params = {
                "q": keyword,
                "tbm": "isch",
                "ijn": str(page),
                "asearch": "ichunk",
                "async": "_id:rg_s,_pms:s,_fmt:pc",
                "sourceid": "chrome",
                "ie": "UTF-8",
                "hl": "en",
            }
            yield params
            page += 1

    def save(self, url: str, save_dir: str, filename: str) -> bool:
        res = requests.get(url=url)
        if res.status_code != 200:
            return False

        if res.content.startswith(b"\xff\xd8"):
            filename = f"{filename}.jpg"
        elif res.content.startswith(b"\x89PNG"):
            filename = f"{filename}.png"
        else:
            return False

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)

        with open(path, "wb") as f:
            f.write(res.content)

        return True

    def save_by_base64(self, data: str, save_dir: str, filename: str) -> bool:
        fmt, data = data.split(";")
        fmt = fmt.split("/")[-1]
        if fmt == "jpeg":
            fmt = "jpg"
        if fmt == "gif":
            return False
        data = data.replace("base64,", "")

        filename = f"{filename}.{fmt}"

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)

        with open(path, "wb") as f:
            f.write(base64.decodebytes(data.encode()))

        return True


if __name__ == "__main__":
    crawler = Crawler(use_splash=False)
    res = crawler.search(keyword="sloth", max_num=10000)
    save_dir = "images"

    for filename, data in tqdm(res.items()):
        if (url := data.get("url")) is not None:
            crawler.save(url, save_dir=save_dir, filename=filename)
        elif (image := data.get("data")) is not None:
            crawler.save_by_base64(image, save_dir=save_dir, filename=filename)
