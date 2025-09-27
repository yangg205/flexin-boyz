#!/usr/bin/env python3
"""
get_map.py

Cách dùng:
  # 1) Truyền token và map_type bằng tham số:
  python get_map.py --token YOUR_TEAM_TOKEN --map_type map_z

  # 2) Hoặc lưu token vào biến môi trường TEAM_TOKEN:
  export TEAM_TOKEN=YOUR_TEAM_TOKEN
  python get_map.py --map_type map_z

Theo đề bài: endpoint GET /api/maps/get_active_map/?token=[token]&map_type=[map_type].
(map_type: map_a / map_b / map_z). Tham khảo: tài liệu Hackathon (API examples). 
"""

import os
import sys
import argparse
import json
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DOMAIN = "https://hackathon2025-dev.fpt.edu.vn"
ENDPOINT = "/api/maps/get_active_map/"

def create_session_with_retries(total_retries=3, backoff_factor=0.5, status_forcelist=(429,500,502,503,504)):
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        method_whitelist=frozenset(["GET", "POST"])   # dùng key cũ
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch_map(token: str, map_type: str, timeout=10):
    if not token:
        raise ValueError("Token is required")
    params = {"token": token, "map_type": map_type}
    url = DOMAIN.rstrip("/") + ENDPOINT
    session = create_session_with_retries()
    resp = session.get(url, params=params, timeout=timeout)
    # Raise for HTTP errors (4xx/5xx)
    resp.raise_for_status()
    return resp.json()

def save_map_json(data, map_type: str, out_dir="maps"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{map_type}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(out_path)

def summarize_map_json(data):
    # Tùy cấu trúc map — cố gắng rút ra thông tin phổ biến: nodes, edges...
    info = {}
    if isinstance(data, dict):
        if "nodes" in data and isinstance(data["nodes"], list):
            info["nodes_count"] = len(data["nodes"])
        elif "vertices" in data and isinstance(data["vertices"], list):
            info["nodes_count"] = len(data["vertices"])
        # detect more fields if needed
        if "map_type" in data:
            info["map_type"] = data["map_type"]
    return info

def main():
    p = argparse.ArgumentParser(description="Kéo map từ server và lưu ra file JSON")
    p.add_argument("--token", type=str, help="Team token (hoặc set environment TEAM_TOKEN)")
    p.add_argument("--map_type", type=str, default="map_z", choices=["map_a","map_b","map_z"],
                   help="Loại map: map_a / map_b / map_z (mặc định: map_z)")
    p.add_argument("--out_dir", type=str, default="maps", help="Thư mục lưu file JSON")
    args = p.parse_args()

    token = args.token or os.environ.get("TEAM_TOKEN") or os.environ.get("TEAM'S_TOKEN")  # fallback
    if not token:
        print("Error: Token không được cung cấp. Truyền --token hoặc set env TEAM_TOKEN.", file=sys.stderr)
        sys.exit(2)

    try:
        print(f"Lấy map '{args.map_type}' từ server...")
        data = fetch_map(token=token, map_type=args.map_type)
    except requests.HTTPError as e:
        print(f"HTTP error khi lấy map: {e} (status {getattr(e.response,'status_code',None)})", file=sys.stderr)
        try:
            print("Response body:", e.response.text, file=sys.stderr)
        except Exception:
            pass
        sys.exit(3)
    except Exception as e:
        print(f"Lỗi khi kết nối/parse: {e}", file=sys.stderr)
        sys.exit(4)

    out_path = save_map_json(data, args.map_type, out_dir=args.out_dir)
    print(f"Đã lưu map vào: {out_path}")

    summary = summarize_map_json(data)
    if summary:
        print("Tóm tắt map:", json.dumps(summary, ensure_ascii=False))
    else:
        print("Không tìm thấy thông tin nodes/summary tự động trong JSON (kiểm tra file).")

if __name__ == "__main__":
    main()
