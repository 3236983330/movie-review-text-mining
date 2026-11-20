import requests
import csv
import time
import random
import json

# 配置
movie_id = 1212592
save_path = os.path.join("data", "maoyan_raw_xhs_city.csv")
target_count = 465
base_api = "https://m.maoyan.com/mmdb/comments/movie/{}.json?_v_=yes&offset={}"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

ua_list = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 16_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15"
]

# 修改表头，点赞数 → 用户城市
field_names = ["用户昵称", "评分", "评论内容", "用户城市", "评论时间"]

with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()
    count = 0
    offset = 0

    while count < target_count:
        url = base_api.format(movie_id, offset)
        headers = {"User-Agent": random.choice(ua_list)}
        try:
            res = requests.get(url, headers=headers, timeout=12)
            if not res.text.strip():
                print(f"offset={offset}，返回空，休眠等待")
                time.sleep(random.uniform(4,7))
                offset +=15
                continue
            data = res.json()
            cmts = data.get("cmts", [])
            if not cmts:
                print("没有更多评论，提前结束")
                break
            for c in cmts:
                row = {
                    "用户昵称": c.get("nickName", ""),
                    "评分": c.get("score", ""),
                    "评论内容": c.get("content", ""),
                    "用户城市": c.get("cityName", ""),  # 城市字段
                    "评论时间": c.get("time", "")
                }
                writer.writerow(row)
                count += 1
                if count >= target_count:
                    break
            print(f"已爬取 {count} 条")
            offset += 15
            time.sleep(random.uniform(2.5,5))
        except json.JSONDecodeError:
            print(f"offset={offset} JSON解析失败，风控拦截")
            time.sleep(random.uniform(4,7))
            offset +=15
        except Exception as e:
            print("出错：", e)
            time.sleep(3)

print(f"✅ 任务完成！一共爬取 {count} 条，保存路径：{save_path}")
