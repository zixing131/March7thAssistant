from __future__ import annotations

import argparse
import base64
import contextlib
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mcp.server.fastmcp import FastMCP, Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
if sys.platform == "win32":
    os.environ.setdefault("ProgramData", r"C:\ProgramData")
    os.environ.setdefault("ProgramFiles", r"C:\Program Files")

CoordinateMode = Literal["screenshot", "relative", "window_relative", "absolute"]

_PROJECT_MODULES: dict[str, Any] | None = None
_AUTOMATION_LOCK = threading.RLock()


def _load_project_modules() -> dict[str, Any]:
    global _PROJECT_MODULES
    if _PROJECT_MODULES is not None:
        return _PROJECT_MODULES

    # MCP stdio uses stdout for JSON-RPC. Some project modules print banners on
    # import, so silence import-time stdout to keep the protocol and logs clean.
    with contextlib.redirect_stdout(io.StringIO()):
        from module.automation import auto
        from module.automation.screenshot import Screenshot
        from module.config import cfg
        from module.ocr import ocr

    _PROJECT_MODULES = {
        "auto": auto,
        "Screenshot": Screenshot,
        "cfg": cfg,
        "ocr": ocr,
    }
    return _PROJECT_MODULES


def _crop(crop_x: float, crop_y: float, crop_w: float, crop_h: float) -> tuple[float, float, float, float]:
    values = (float(crop_x), float(crop_y), float(crop_w), float(crop_h))
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("crop_w and crop_h must be greater than 0")
    return values


def _png_bytes(image: Any) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _ensure_screenshot() -> Any:
    modules = _load_project_modules()
    auto = modules["auto"]
    if getattr(auto, "screenshot", None) is None:
        auto.take_screenshot()
    return auto


def _window_region() -> tuple[int, int, int, int]:
    modules = _load_project_modules()
    cfg = modules["cfg"]
    Screenshot = modules["Screenshot"]
    if cfg.get_value("cloud_game_enable", False):
        return (0, 0, 1920, 1080)

    window = Screenshot.get_window(cfg.get_value("game_title_name"))
    if not window:
        raise RuntimeError(f"Game window not found: {cfg.get_value('game_title_name')}")
    return Screenshot.get_window_region(window)


def _to_absolute_xy(x: float, y: float, coordinate_mode: CoordinateMode) -> tuple[int, int]:
    auto = _ensure_screenshot()
    mode = coordinate_mode

    if mode == "absolute":
        return int(round(x)), int(round(y))

    if mode == "window_relative":
        left, top, width, height = _window_region()
        return int(round(left + x * width)), int(round(top + y * height))

    if mode == "relative":
        image_width, image_height = auto.screenshot.size
        x = x * image_width
        y = y * image_height
        mode = "screenshot"

    if mode == "screenshot":
        scale = getattr(auto, "screenshot_scale_factor", 1) or 1
        pos = getattr(auto, "screenshot_pos", None)
        if pos is None:
            raise RuntimeError("No screenshot position is available")
        return int(round(pos[0] + x / scale)), int(round(pos[1] + y / scale))

    raise ValueError(f"Unknown coordinate_mode: {coordinate_mode}")


def _box_from_screenshot_coords(top_left: tuple[int, int], bottom_right: tuple[int, int]) -> dict[str, Any]:
    left, top = top_left
    right, bottom = bottom_right
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    abs_center = _to_absolute_xy(center_x, center_y, "screenshot")
    abs_top_left = _to_absolute_xy(left, top, "screenshot")
    abs_bottom_right = _to_absolute_xy(right, bottom, "screenshot")
    return {
        "found": True,
        "screenshot_box": {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
            "center_x": center_x,
            "center_y": center_y,
        },
        "absolute_box": {
            "left": abs_top_left[0],
            "top": abs_top_left[1],
            "right": abs_bottom_right[0],
            "bottom": abs_bottom_right[1],
            "center_x": abs_center[0],
            "center_y": abs_center[1],
        },
    }


def _scale_range(scale_min: float, scale_max: float) -> tuple[float, float] | None:
    if scale_min <= 0 or scale_max <= 0:
        return None
    if scale_min > scale_max:
        raise ValueError("scale_min must be less than or equal to scale_max")
    if scale_min == 1.0 and scale_max == 1.0:
        return None
    return float(scale_min), float(scale_max)


def _template_path(template_path: str) -> str:
    path = Path(template_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Template image not found: {path}")
    return str(path)


def create_server(host: str = "127.0.0.1", port: int = 8000, log_level: str = "WARNING") -> FastMCP:
    mcp = FastMCP(
        "March7thAssistant",
        instructions=(
            "Control March7th Assistant game automation through MCP. "
            "Use capture_screen first, inspect the image, then send mouse or keyboard actions. "
            "Coordinates default to pixels in the latest screenshot."
        ),
        host=host,
        port=port,
        log_level=log_level,
    )

    @mcp.tool()
    def get_status() -> dict[str, Any]:
        """Return current game automation status, window title, and window region when available."""
        with _AUTOMATION_LOCK:
            modules = _load_project_modules()
            cfg = modules["cfg"]
            Screenshot = modules["Screenshot"]
            title = cfg.get_value("game_title_name")
            cloud = bool(cfg.get_value("cloud_game_enable", False))
            window_found = True
            region = None
            if cloud:
                region = (0, 0, 1920, 1080)
            else:
                window = Screenshot.get_window(title)
                window_found = bool(window)
                if window:
                    region = Screenshot.get_window_region(window)
            return {
                "game_title_name": title,
                "cloud_game_enable": cloud,
                "window_found": window_found,
                "window_region": region,
                "use_background_screenshot": cfg.get_value("use_background_screenshot", None),
                "latest_screenshot_size": getattr(modules["auto"].screenshot, "size", None),
            }

    @mcp.tool()
    def capture_screen(
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_w: float = 1.0,
        crop_h: float = 1.0,
        use_background_screenshot: bool | None = None,
        prefer_frame_screenshot: bool = True,
    ) -> Image:
        """Capture the game screen and return it as a PNG image for visual AI recognition."""
        with _AUTOMATION_LOCK:
            modules = _load_project_modules()
            auto = modules["auto"]
            screenshot, _, _ = auto.take_screenshot(
                _crop(crop_x, crop_y, crop_w, crop_h),
                use_background_screenshot=use_background_screenshot,
                prefer_frame_screenshot=prefer_frame_screenshot,
            )
            return Image(data=_png_bytes(screenshot), format="png")

    @mcp.tool()
    def capture_screen_data(
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_w: float = 1.0,
        crop_h: float = 1.0,
        use_background_screenshot: bool | None = None,
        prefer_frame_screenshot: bool = True,
    ) -> dict[str, Any]:
        """Capture the game screen and return PNG bytes as base64 plus coordinate metadata."""
        with _AUTOMATION_LOCK:
            modules = _load_project_modules()
            auto = modules["auto"]
            screenshot, pos, scale = auto.take_screenshot(
                _crop(crop_x, crop_y, crop_w, crop_h),
                use_background_screenshot=use_background_screenshot,
                prefer_frame_screenshot=prefer_frame_screenshot,
            )
            png = _png_bytes(screenshot)
            return {
                "format": "png",
                "width": screenshot.width,
                "height": screenshot.height,
                "screenshot_pos": pos,
                "screenshot_scale_factor": scale,
                "image_base64": base64.b64encode(png).decode("ascii"),
            }

    @mcp.tool()
    def ocr_screen(
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_w: float = 1.0,
        crop_h: float = 1.0,
        use_background_screenshot: bool | None = None,
        prefer_frame_screenshot: bool = True,
    ) -> dict[str, Any]:
        """Run March7th Assistant OCR on the current game screen and return text boxes."""
        with _AUTOMATION_LOCK:
            modules = _load_project_modules()
            auto = modules["auto"]
            ocr = modules["ocr"]
            auto.take_screenshot(
                _crop(crop_x, crop_y, crop_w, crop_h),
                use_background_screenshot=use_background_screenshot,
                prefer_frame_screenshot=prefer_frame_screenshot,
            )
            results = []
            for box, (text, confidence) in ocr.recognize_multi_lines(np.array(auto.screenshot)) or []:
                results.append({
                    "text": text,
                    "confidence": float(confidence),
                    "box": _plain(box),
                })
            return {
                "width": auto.screenshot.width,
                "height": auto.screenshot.height,
                "results": results,
            }

    @mcp.tool()
    def mouse_click(
        x: float,
        y: float,
        coordinate_mode: CoordinateMode = "screenshot",
        press_duration: float = 0.0,
    ) -> dict[str, Any]:
        """Click the game. By default x/y are pixels in the latest screenshot."""
        with _AUTOMATION_LOCK:
            modules = _load_project_modules()
            auto = modules["auto"]
            abs_x, abs_y = _to_absolute_xy(x, y, coordinate_mode)
            if press_duration > 0:
                auto.mouse_down(abs_x, abs_y)
                time.sleep(float(press_duration))
                auto.mouse_up()
            else:
                auto.mouse_click(abs_x, abs_y)
            return {"ok": True, "x": abs_x, "y": abs_y}

    @mcp.tool()
    def mouse_move(x: float, y: float, coordinate_mode: CoordinateMode = "screenshot") -> dict[str, Any]:
        """Move the mouse. By default x/y are pixels in the latest screenshot."""
        with _AUTOMATION_LOCK:
            modules = _load_project_modules()
            abs_x, abs_y = _to_absolute_xy(x, y, coordinate_mode)
            modules["auto"].mouse_move(abs_x, abs_y)
            return {"ok": True, "x": abs_x, "y": abs_y}

    @mcp.tool()
    def mouse_down(x: float, y: float, coordinate_mode: CoordinateMode = "screenshot") -> dict[str, Any]:
        """Press and hold the left mouse button at the given coordinate."""
        with _AUTOMATION_LOCK:
            modules = _load_project_modules()
            abs_x, abs_y = _to_absolute_xy(x, y, coordinate_mode)
            modules["auto"].mouse_down(abs_x, abs_y)
            return {"ok": True, "x": abs_x, "y": abs_y}

    @mcp.tool()
    def mouse_up() -> dict[str, Any]:
        """Release the left mouse button."""
        with _AUTOMATION_LOCK:
            _load_project_modules()["auto"].mouse_up()
            return {"ok": True}

    @mcp.tool()
    def mouse_scroll(count: int, direction: int = -1, pause: bool = True) -> dict[str, Any]:
        """Scroll the mouse wheel. direction=-1 scrolls down, direction=1 scrolls up."""
        with _AUTOMATION_LOCK:
            _load_project_modules()["auto"].mouse_scroll(int(count), int(direction), bool(pause))
            return {"ok": True, "count": int(count), "direction": int(direction)}

    @mcp.tool()
    def press_key(key: str, duration: float = 0.2) -> dict[str, Any]:
        """Press and release a keyboard key, for example w, a, s, d, f, esc, enter, or space."""
        with _AUTOMATION_LOCK:
            _load_project_modules()["auto"].press_key(key, float(duration))
            return {"ok": True, "key": key, "duration": float(duration)}

    @mcp.tool()
    def key_down(key: str) -> dict[str, Any]:
        """Press and hold a keyboard key."""
        with _AUTOMATION_LOCK:
            _load_project_modules()["auto"].press_key_down(key)
            return {"ok": True, "key": key}

    @mcp.tool()
    def key_up(key: str) -> dict[str, Any]:
        """Release a keyboard key."""
        with _AUTOMATION_LOCK:
            _load_project_modules()["auto"].press_key_up(key)
            return {"ok": True, "key": key}

    @mcp.tool()
    def type_text(text: str, interval: float = 0.05) -> dict[str, Any]:
        """Type text into the game or launcher using the active input backend."""
        with _AUTOMATION_LOCK:
            _load_project_modules()["auto"].secretly_write(text, float(interval))
            return {"ok": True, "length": len(text), "interval": float(interval)}

    @mcp.tool()
    def wait(seconds: float = 1.0) -> dict[str, Any]:
        """Wait for a short time between observation and action steps."""
        time.sleep(max(0.0, float(seconds)))
        return {"ok": True, "seconds": max(0.0, float(seconds))}

    @mcp.tool()
    def find_text(
        text: str,
        include: bool = True,
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_w: float = 1.0,
        crop_h: float = 1.0,
        max_retries: int = 1,
    ) -> dict[str, Any]:
        """Find text on screen using OCR and return screenshot and absolute coordinates."""
        with _AUTOMATION_LOCK:
            modules = _load_project_modules()
            match = modules["auto"].find_element(
                text,
                "text",
                max_retries=max(1, int(max_retries)),
                crop=_crop(crop_x, crop_y, crop_w, crop_h),
                relative=True,
                include=bool(include),
            )
            if not match:
                return {"found": False}
            return _box_from_screenshot_coords(match[0], match[1])

    @mcp.tool()
    def click_text(
        text: str,
        include: bool = True,
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_w: float = 1.0,
        crop_h: float = 1.0,
        max_retries: int = 1,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> dict[str, Any]:
        """Find text with OCR and click its center plus an optional screenshot-pixel offset."""
        with _AUTOMATION_LOCK:
            result = find_text(text, include, crop_x, crop_y, crop_w, crop_h, max_retries)
            if not result.get("found"):
                return result
            box = result["screenshot_box"]
            click_result = mouse_click(
                box["center_x"] + offset_x,
                box["center_y"] + offset_y,
                "screenshot",
            )
            return {**result, "clicked": True, "click": click_result}

    @mcp.tool()
    def find_image(
        template_path: str,
        threshold: float = 0.8,
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_w: float = 1.0,
        crop_h: float = 1.0,
        max_retries: int = 1,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
    ) -> dict[str, Any]:
        """Find a template image on screen and return screenshot and absolute coordinates."""
        with _AUTOMATION_LOCK:
            modules = _load_project_modules()
            match = modules["auto"].find_element(
                _template_path(template_path),
                "image",
                threshold=float(threshold),
                max_retries=max(1, int(max_retries)),
                crop=_crop(crop_x, crop_y, crop_w, crop_h),
                relative=True,
                scale_range=_scale_range(scale_min, scale_max),
            )
            if not match:
                return {"found": False}
            return _box_from_screenshot_coords(match[0], match[1])

    @mcp.tool()
    def click_image(
        template_path: str,
        threshold: float = 0.8,
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_w: float = 1.0,
        crop_h: float = 1.0,
        max_retries: int = 1,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> dict[str, Any]:
        """Find a template image and click its center plus an optional screenshot-pixel offset."""
        with _AUTOMATION_LOCK:
            result = find_image(
                template_path,
                threshold,
                crop_x,
                crop_y,
                crop_w,
                crop_h,
                max_retries,
                scale_min,
                scale_max,
            )
            if not result.get("found"):
                return result
            box = result["screenshot_box"]
            click_result = mouse_click(
                box["center_x"] + offset_x,
                box["center_y"] + offset_y,
                "screenshot",
            )
            return {**result, "clicked": True, "click": click_result}

    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="March7th Assistant MCP server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args(argv)

    server = create_server(host=args.host, port=args.port, log_level=args.log_level)
    server.run(args.transport)


if __name__ == "__main__":
    main()
