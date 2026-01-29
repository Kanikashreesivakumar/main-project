from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

Point2i = Tuple[int, int]


@dataclass
class Block:
    block_id: int
    center: Point2i
    size: int
    color_bgr: Tuple[int, int, int]


class BlockWorld:
    def __init__(self) -> None:
        self._next_id = 1
        self.blocks: List[Block] = []
        self.selected_id: Optional[int] = None

    def create_block(self, center: Tuple[float, float], size: int = 70) -> Block:
        cx, cy = int(center[0]), int(center[1])
        # Deterministic pleasant colors
        palette = [
            (80, 190, 255),
            (120, 220, 120),
            (255, 170, 80),
            (180, 140, 255),
            (120, 200, 240),
        ]
        color = palette[(self._next_id - 1) % len(palette)]

        block = Block(block_id=self._next_id, center=(cx, cy), size=size, color_bgr=color)
        self._next_id += 1
        self.blocks.append(block)
        return block

    def _get_block(self, block_id: int) -> Optional[Block]:
        for b in self.blocks:
            if b.block_id == block_id:
                return b
        return None

    def clear_selection(self) -> None:
        self.selected_id = None

    def select_nearest(self, point: Tuple[float, float], max_dist_px: float = 120.0) -> Optional[Block]:
        px, py = float(point[0]), float(point[1])

        best: Optional[Block] = None
        best_d = 1e18
        for b in self.blocks:
            bx, by = b.center
            d = float(np.hypot(px - bx, py - by))
            if d < best_d:
                best = b
                best_d = d

        if best is None or best_d > max_dist_px:
            self.selected_id = None
            return None

        self.selected_id = best.block_id
        return best

    def move_selected_to(self, point: Tuple[float, float]) -> None:
        if self.selected_id is None:
            return
        b = self._get_block(self.selected_id)
        if b is None:
            self.selected_id = None
            return
        b.center = (int(point[0]), int(point[1]))

    def draw(self, frame_bgr: np.ndarray) -> None:
        for b in self.blocks:
            cx, cy = b.center
            half = b.size // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half

            is_selected = self.selected_id == b.block_id
            thickness = 3 if is_selected else 2

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), b.color_bgr, thickness)
            if is_selected:
                cv2.rectangle(frame_bgr, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 2)

            cv2.putText(
                frame_bgr,
                f"#{b.block_id}",
                (x1 + 6, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                f"#{b.block_id}",
                (x1 + 6, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
