#!/usr/bin/env python3
"""자율주행 트랙 매트 디지털 트윈 도면 생성기.

출력 (이 파일과 같은 폴더)
  track_texture.png  위에서 본 매트 이미지. 1px = 1mm, 흰 테두리 포함 2355(동서) x 2400(남북)
                     -> Isaac Sim / MuJoCo 바닥 평면에 텍스처로 입힌다 (평면 크기 2.355m x 2.4m)
  track_layout.json  도로 중심선, 교차로, 횡단보도, 노면 표시, 소품 자리(파란 원) 좌표
                     단위 m, 원점 = 매트 중심, x = 동(이미지 오른쪽), y = 북(이미지 위쪽)

방향 기준: 빨간 실선 두 개가 있는 도로를 북쪽, 매트 이음매가 지나가는 가운데 도로를 동서로 둔다.

치수 출처 (2026-10-14)
  [측정] 줄자로 잰 값
  [사진] 트랙 사진을 원근 보정해 위에서 본 모습으로 편 뒤 잰 값 (오차 약 +-2cm)
  [추정] 위 둘로 정하지 못해 맞춘 값
"""
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent
SS = 2  # 슈퍼샘플링 배율 (가장자리를 부드럽게 그린 뒤 축소)

# ---------------- 치수 (mm) ----------------
# 인쇄 영역(초록 끝~초록 끝)은 228 x 223.5 직사각형. 짧은 쪽을 동서(이음매와 나란한 방향)로 둔 근거:
# 원근 보정 사진에서 순환로 바깥 테두리가 동서로 약 6cm 짧고, 블록 동서 폭이 약 63.6cm로 나옴
PRINT_X = 2235          # [측정] 동서 전체 (세로 223~224cm)
PRINT_Y = 2280          # [측정] 남북 전체 (가로 228cm)
OUTER_ROAD = 260        # [측정] 바깥 순환로 폭 (양쪽 노란선 포함)
CENTER_ROAD_EW = 185    # [측정] 동서 방향 가운데 도로 (이음매가 지나가는 도로)
CENTER_ROAD_NS = 210    # [측정] 남북 방향 가운데 도로 (2026-10-30 줄자 21cm, 사진 약 207mm)
BLOCK_Y = 650           # [측정] 안쪽 초록 블록 남북 길이
BAND_X = 125            # [측정] 동서 끝 바깥 초록 띠 (줄자 사진 9의 12.5cm)
# 나머지로 계산: 블록 동서 폭 627.5 (사진 약 636), 남북 끝 초록 띠 137.5
BLOCK_X = (PRINT_X - 2 * BAND_X - 2 * OUTER_ROAD - CENTER_ROAD_NS) / 2
BAND_Y = (PRINT_Y - 2 * OUTER_ROAD - 2 * BLOCK_Y - CENTER_ROAD_EW) / 2

STRIPE_W, STRIPE_GAP, STRIPE_LEN = 20, 11.5, 120  # [측정] 횡단보도 흰 줄 폭, 간격, 길이
STRIPE_N = 6                                      # [사진] 횡단보도 줄 수
CW_CENTER_GAP = 105     # [사진] 가운데 교차로 가장자리 ~ 횡단보도
CW_OUTER_GAP = 50       # [사진] T자 교차로 가장자리 ~ 횡단보도

BLOCK_R = 105           # [사진] 블록 모서리 반경 (모서리 5곳 104~107)
ROAD_R = 325            # [사진] 순환로 바깥 모서리 반경 (모서리 3곳 322~333)
GREEN_R = 370           # [사진] 초록 띠 바깥 모서리 반경 (모서리 2곳 367~380)
SPOT_D = 105            # [사진] 파란 원 지름
SPOT_INSET = 100        # [사진] 블록 가장자리 ~ 파란 원 중심
PINK_X_FROM_EDGE = 455  # [사진] 동서 끝 ~ 남북 방향 분홍 점선 (3곳 440~459)
PINK_Y_FROM_EDGE = 490  # [사진] 남북 끝 ~ 동서 방향 분홍 점선 (4곳 477~503)
PINK_LEN, RED_LEN = 200, 160          # [사진] 분홍 점선 / 빨간 실선 길이 (도로 가운데 정렬)
RED_FROM_T = (225, 515)               # [사진] 북쪽 T자 교차로 동쪽 끝 ~ 빨간 실선 두 개
DASH, DASH_GAP, MARK_W = 12, 6, 3     # [추정] 점선 모양, 표시선 두께
LINE_W = 3              # [추정] 노란선 두께
EDGE_W = 4              # [추정] 초록 바깥 테두리의 진한 선 두께
BORDER = 60             # [추정] 흰 테두리 폭 -> 전체 판 2355 x 2400
SEAM = True             # 매트 2장 이음매 (동서 가운데 도로 한가운데). 카메라가 중앙선으로 오인할 수 있어 넣어 둠

COLORS = {
    "white": (244, 244, 242),
    "green": (78, 178, 72),
    "edge": (38, 58, 46),
    "road": (30, 32, 32),
    "yellow": (238, 222, 42),
    "stripe": (240, 240, 238),
    "blue": (62, 168, 236),
    "pink": (228, 140, 210),
    "red": (222, 52, 48),
    "seam": (150, 150, 150, 70),
}


def axis_spans(band, block, center, total):
    """한 축을 따라 (바깥 도로, 블록, 가운데 도로, 블록, 바깥 도로) 구간. 인쇄 영역 mm."""
    edges = [band]
    for w in (OUTER_ROAD, block, center, block, OUTER_ROAD):
        edges.append(edges[-1] + w)
    assert abs(edges[-1] + band - total) < 1e-6
    keys = ("road_lo", "block_lo", "center", "block_hi", "road_hi")
    return {k: (edges[i], edges[i + 1]) for i, k in enumerate(keys)}


# 인쇄 영역 좌표: 왼쪽 위 원점, x 오른쪽(동), y 아래쪽(남)
X = axis_spans(BAND_X, BLOCK_X, CENTER_ROAD_NS, PRINT_X)
Y = axis_spans(BAND_Y, BLOCK_Y, CENTER_ROAD_EW, PRINT_Y)


def world(px, py):
    """인쇄 영역 좌표(mm) -> 월드 좌표(m, 매트 중심 원점, y 위쪽)."""
    return [round((px - PRINT_X / 2) / 1000, 4), round((PRINT_Y / 2 - py) / 1000, 4)]


class Canvas:
    def __init__(self):
        self.size = (PRINT_X + 2 * BORDER, PRINT_Y + 2 * BORDER)
        self.img = Image.new("RGB", (round(self.size[0] * SS), round(self.size[1] * SS)), COLORS["white"])
        self.draw = ImageDraw.Draw(self.img, "RGBA")

    @staticmethod
    def _px(v):
        return round((v + BORDER) * SS)

    def _box(self, x0, y0, x1, y1):
        return [self._px(x0), self._px(y0), self._px(x1) - 1, self._px(y1) - 1]

    def rrect(self, x0, y0, x1, y1, r, fill):
        self.draw.rounded_rectangle(self._box(x0, y0, x1, y1), radius=round(r * SS), fill=fill)

    def rect(self, x0, y0, x1, y1, fill):
        self.draw.rectangle(self._box(x0, y0, x1, y1), fill=fill)

    def circle(self, cx, cy, d, fill):
        self.draw.ellipse(self._box(cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2), fill=fill)

    def save(self, path):
        self.img.resize(self.size, Image.LANCZOS).save(path, optimize=True)


def draw_base(cv):
    """초록 띠, 순환로, 블록, 노란선."""
    cv.rrect(0, 0, PRINT_X, PRINT_Y, GREEN_R, COLORS["edge"])
    cv.rrect(EDGE_W, EDGE_W, PRINT_X - EDGE_W, PRINT_Y - EDGE_W, GREEN_R - EDGE_W, COLORS["green"])
    # 도로 전체(바깥 경계 안쪽)를 검게 칠하고 블록을 위에 얹는다
    x0, x1 = X["road_lo"][0], X["road_hi"][1]
    y0, y1 = Y["road_lo"][0], Y["road_hi"][1]
    cv.rrect(x0, y0, x1, y1, ROAD_R, COLORS["yellow"])
    cv.rrect(x0 + LINE_W, y0 + LINE_W, x1 - LINE_W, y1 - LINE_W, ROAD_R - LINE_W, COLORS["road"])
    for bx in (X["block_lo"], X["block_hi"]):
        for by in (Y["block_lo"], Y["block_hi"]):
            cv.rrect(bx[0], by[0], bx[1], by[1], BLOCK_R, COLORS["yellow"])
            cv.rrect(bx[0] + LINE_W, by[0] + LINE_W, bx[1] - LINE_W, by[1] - LINE_W,
                     BLOCK_R - LINE_W, COLORS["green"])


def draw_crosswalk(cv, axis, across, along0, name, road, layout):
    """axis='x': 도로가 동서로 뻗음(흰 줄도 동서). across=(시작, 끝) 도로 폭 구간."""
    total = STRIPE_N * STRIPE_W + (STRIPE_N - 1) * STRIPE_GAP
    start = (across[0] + across[1] - total) / 2
    for k in range(STRIPE_N):
        a = start + k * (STRIPE_W + STRIPE_GAP)
        if axis == "x":
            cv.rect(along0, a, along0 + STRIPE_LEN, a + STRIPE_W, COLORS["stripe"])
        else:
            cv.rect(a, along0, a + STRIPE_W, along0 + STRIPE_LEN, COLORS["stripe"])
    mid_across = (across[0] + across[1]) / 2
    mid_along = along0 + STRIPE_LEN / 2
    center = world(mid_along, mid_across) if axis == "x" else world(mid_across, mid_along)
    layout["crosswalks"].append({
        "id": name, "road": road, "road_direction": "east-west" if axis == "x" else "north-south",
        "center": center, "length_along_road": STRIPE_LEN / 1000, "width_across_road": round(total / 1000, 4),
    })


def draw_crosswalks(cv, layout):
    g, o, L = CW_CENTER_GAP, CW_OUTER_GAP, STRIPE_LEN
    xc, yc = X["center"], Y["center"]
    # 가운데 교차로 네 방향
    draw_crosswalk(cv, "y", xc, yc[0] - g - L, "center-N", "center_ns", layout)
    draw_crosswalk(cv, "y", xc, yc[1] + g, "center-S", "center_ns", layout)
    draw_crosswalk(cv, "x", yc, xc[0] - g - L, "center-W", "center_ew", layout)
    draw_crosswalk(cv, "x", yc, xc[1] + g, "center-E", "center_ew", layout)
    # 바깥 순환로: 시계 방향으로 돌 때 T자 교차로 바로 앞 (바람개비 배치)
    draw_crosswalk(cv, "x", Y["road_lo"], xc[0] - o - L, "outer-N", "outer", layout)
    draw_crosswalk(cv, "y", X["road_hi"], yc[0] - o - L, "outer-E", "outer", layout)
    draw_crosswalk(cv, "x", Y["road_hi"], xc[1] + o, "outer-S", "outer", layout)
    draw_crosswalk(cv, "y", X["road_lo"], yc[1] + o, "outer-W", "outer", layout)


def draw_transverse(cv, axis, at, across, length, color, dashed, name, kind, layout):
    """도로를 가로지르는 선 (도로 가운데 정렬). axis='x': 동서 도로 위의 남북 방향 선 (x=at)."""
    mid = (across[0] + across[1]) / 2
    a0, a1 = mid - length / 2, mid + length / 2
    segs = []
    if dashed:
        a = a0
        while a < a1:
            segs.append((a, min(a + DASH, a1)))
            a += DASH + DASH_GAP
    else:
        segs.append((a0, a1))
    for s0, s1 in segs:
        if axis == "x":
            cv.rect(at - MARK_W / 2, s0, at + MARK_W / 2, s1, color)
        else:
            cv.rect(s0, at - MARK_W / 2, s1, at + MARK_W / 2, color)
    p0, p1 = (world(at, a0), world(at, a1)) if axis == "x" else (world(a0, at), world(a1, at))
    layout["road_marks"].append({"id": name, "kind": kind, "p0": p0, "p1": p1})


def draw_marks(cv, layout):
    west, east = PINK_X_FROM_EDGE, PRINT_X - PINK_X_FROM_EDGE
    north, south = PINK_Y_FROM_EDGE, PRINT_Y - PINK_Y_FROM_EDGE
    pink = lambda axis, at, road, name: draw_transverse(
        cv, axis, at, road, PINK_LEN, COLORS["pink"], True, name, "pink_dashed", layout)
    # 분홍 점선: 네 모서리 곡선 양 끝. 북동 모서리의 북쪽 끝만 빨간 실선으로 대체됨
    pink("x", west, Y["road_lo"], "pink-NW-N")
    pink("x", west, Y["road_hi"], "pink-SW-S")
    pink("x", east, Y["road_hi"], "pink-SE-S")
    pink("y", north, X["road_lo"], "pink-NW-W")
    pink("y", south, X["road_lo"], "pink-SW-W")
    pink("y", north, X["road_hi"], "pink-NE-E")
    pink("y", south, X["road_hi"], "pink-SE-E")
    for k, off in enumerate(RED_FROM_T):
        draw_transverse(cv, "x", X["center"][1] + off, Y["road_lo"], RED_LEN, COLORS["red"], False,
                        f"red-N-{k + 1}", "red_solid", layout)


def draw_spots(cv, layout):
    """파란 원 = 표지판·신호등·나무를 꽂는 자리."""
    for by, ny in ((Y["block_lo"], "N"), (Y["block_hi"], "S")):
        for bx, nx in ((X["block_lo"], "W"), (X["block_hi"], "E")):
            for cy, cyn in ((by[0] + SPOT_INSET, "N"), (by[1] - SPOT_INSET, "S")):
                for cx, cxn in ((bx[0] + SPOT_INSET, "W"), (bx[1] - SPOT_INSET, "E")):
                    cv.circle(cx, cy, SPOT_D, COLORS["blue"])
                    layout["prop_spots"].append({"id": f"block-{ny}{nx}-{cyn}{cxn}", "on": "block",
                                                 "center": world(cx, cy), "diameter": SPOT_D / 1000})
    # 바깥 띠의 원은 변 가운데가 아니라 옆 블록의 파란 원 줄에 맞춰 시계 방향으로 비켜 있음
    band = (
        (X["center"][1] + SPOT_INSET, BAND_Y / 2, "N"),
        (PRINT_X - BAND_X / 2, Y["center"][1] + SPOT_INSET, "E"),
        (X["center"][0] - SPOT_INSET, PRINT_Y - BAND_Y / 2, "S"),
        (BAND_X / 2, Y["center"][0] - SPOT_INSET, "W"),
    )
    for cx, cy, tag in band:
        cv.circle(cx, cy, SPOT_D, COLORS["blue"])
        layout["prop_spots"].append({"id": f"band-{tag}", "on": "outer_green_band",
                                     "center": world(cx, cy), "diameter": SPOT_D / 1000})


def draw_seam(cv):
    mid = PRINT_Y / 2
    cv.rect(-BORDER, mid - 0.75, PRINT_X + BORDER, mid + 0.75, COLORS["seam"])


def rounded_rect_path(hx, hy, r, step=20):
    """중심 원점, 반폭 hx/hy, 모서리 반경 r인 둥근 사각형 경로 (시계 방향, 북쪽 가운데에서 출발)."""
    pts = []
    sx, sy = hx - r, hy - r

    def line(x0, y0, x1, y1):
        n = max(1, round(math.hypot(x1 - x0, y1 - y0) / step))
        for i in range(n):
            pts.append((x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n))

    def arc(cx, cy, a0, a1):
        n = max(1, round(abs(a1 - a0) * r / step))
        for i in range(n):
            a = a0 + (a1 - a0) * i / n
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    line(0, hy, sx, hy)
    arc(sx, sy, math.pi / 2, 0)
    line(hx, sy, hx, -sy)
    arc(sx, -sy, 0, -math.pi / 2)
    line(sx, -hy, -sx, -hy)
    arc(-sx, -sy, -math.pi / 2, -math.pi)
    line(-hx, -sy, -hx, sy)
    arc(-sx, sy, math.pi, math.pi / 2)
    return [[round(x / 1000, 4), round(y / 1000, 4)] for x, y in pts]


def build_layout():
    hx = PRINT_X / 2 - (BAND_X + OUTER_ROAD / 2)   # 동서 순환로 중심선까지
    hy = PRINT_Y / 2 - (BAND_Y + OUTER_ROAD / 2)   # 남북 순환로 중심선까지
    m = lambda v: round(v / 1000, 4)
    return {
        "meta": {
            "units": "m",
            "origin": "mat center",
            "axes": "x = east (image right), y = north (image up), z = up",
            "orientation": "north road has the two red lines; mat seam runs along the east-west center road",
            "texture": "track_texture.png, 1px = 1mm, covers the whole board",
            "board_size": [m(PRINT_X + 2 * BORDER), m(PRINT_Y + 2 * BORDER)],
            "printed_size": [m(PRINT_X), m(PRINT_Y)],
            "measured_tape": ["printed_size", "outer_road_width", "center_road_ew_width",
                              "block_size north-south", "outer green band east/west",
                              "crosswalk stripe width/gap/length"],
            "from_rectified_photo": ["center_road_ns_width", "corner radii", "crosswalk positions",
                                     "prop spot positions", "pink/red mark positions",
                                     "which axis is the short one"],
            "computed": ["block_size east-west", "outer green band north/south"],
            "estimated": ["white border", "line widths"],
        },
        "dims": {
            "outer_road_width": m(OUTER_ROAD),
            "center_road_ew_width": m(CENTER_ROAD_EW),
            "center_road_ns_width": m(CENTER_ROAD_NS),
            "block_size_xy": [m(BLOCK_X), m(BLOCK_Y)],
            "outer_green_band_east_west": m(BAND_X),
            "outer_green_band_north_south": m(BAND_Y),
            "block_corner_radius": m(BLOCK_R),
            "outer_road_corner_radius": m(ROAD_R),
        },
        "centerlines": {
            "outer_loop_clockwise": rounded_rect_path(hx, hy, ROAD_R - OUTER_ROAD / 2),
            "center_north_south": [[0.0, m(hy)], [0.0, -m(hy)]],
            "center_east_west": [[-m(hx), 0.0], [m(hx), 0.0]],
        },
        "intersections": {
            "center_4way": [0.0, 0.0],
            "T_north": [0.0, m(hy)],
            "T_east": [m(hx), 0.0],
            "T_south": [0.0, -m(hy)],
            "T_west": [-m(hx), 0.0],
        },
        "crosswalks": [],
        "road_marks": [],
        "prop_spots": [],
    }


def main():
    layout = build_layout()
    cv = Canvas()
    draw_base(cv)
    draw_crosswalks(cv, layout)
    draw_marks(cv, layout)
    draw_spots(cv, layout)
    if SEAM:
        draw_seam(cv)
    cv.save(OUT_DIR / "track_texture.png")
    (OUT_DIR / "track_layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2))
    print(f"printed {PRINT_X} x {PRINT_Y} mm, block {BLOCK_X} x {BLOCK_Y} mm, "
          f"green band E/W {BAND_X} mm, N/S {BAND_Y} mm")
    print(f"{len(layout['crosswalks'])} crosswalks, {len(layout['road_marks'])} road marks, "
          f"{len(layout['prop_spots'])} prop spots")


if __name__ == "__main__":
    main()
