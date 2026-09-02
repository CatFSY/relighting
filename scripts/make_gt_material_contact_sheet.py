#!/usr/bin/env python3
"""Make an annotated contact sheet for all GT RGB/albedo/roughness views."""

from pathlib import Path
import argparse

from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def make_sheet(frame_names, base: Path, out: Path, start: int, end: int,
               thumb_w: int, thumb_h: int, label_w: int) -> None:
    header_h = 76
    row_h = thumb_h + 30
    width = label_w + 3 * thumb_w
    height = header_h + (end - start) * row_h
    canvas = Image.new("RGB", (width, height), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    title_font = font(28, bold=True)
    label_font = font(18, bold=True)
    row_font = font(15)

    headers = ["View / frame", "GT RGB", "GT Albedo", "GT Roughness"]
    x_positions = [0, label_w, label_w + thumb_w, label_w + 2 * thumb_w]
    for x, text in zip(x_positions, headers):
        draw.rectangle((x, 0, x + (label_w if x == 0 else thumb_w), header_h), fill="#263238")
        box = draw.textbbox((0, 0), text, font=title_font)
        draw.text((x + 12, (header_h - (box[3] - box[1])) // 2 - box[1]), text,
                  fill="white", font=title_font)

    for row, frame in enumerate(frame_names[start:end]):
        y = header_h + row * row_h
        fill = "#ffffff" if row % 2 == 0 else "#e4e8eb"
        draw.rectangle((0, y, width, y + row_h), fill=fill)
        label = f"view {start + row:03d}\n{frame.stem}"
        draw.multiline_text((10, y + 12), label, fill="#111111", font=row_font, spacing=5)

        paths = [
            base / "images" / f"{frame.stem}.jpg",
            base / "basecolor" / f"{frame.stem}.png",
            base / "roughness" / f"{frame.stem}.png",
        ]
        for col, path in enumerate(paths):
            x = label_w + col * thumb_w
            try:
                image = fit_image(path, (thumb_w - 8, thumb_h - 8))
                ix = x + (thumb_w - image.width) // 2
                iy = y + 4 + (thumb_h - image.height) // 2
                canvas.paste(image, (ix, iy))
                draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h), outline="#9aa3a8", width=1)
            except FileNotFoundError:
                draw.text((x + 10, y + 12), f"missing\n{path.name}", fill="#b00020", font=row_font)

    canvas.save(out, quality=95, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=25)
    args = parser.parse_args()

    frames = sorted((args.dataset / "images").glob("frame_*.jpg"))
    if not frames:
        raise SystemExit(f"no frames found under {args.dataset / 'images'}")

    args.output.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h, label_w = 320, 178, 300
    make_sheet(frames, args.dataset, args.output / "gt_rgb_albedo_roughness_all_views.jpg",
               0, len(frames), thumb_w, thumb_h, label_w)
    for start in range(0, len(frames), args.page_size):
        end = min(start + args.page_size, len(frames))
        make_sheet(frames, args.dataset, args.output / f"gt_views_{start:03d}_{end - 1:03d}.jpg",
                   start, end, thumb_w, thumb_h, label_w)
    print(f"created {len(frames)} views in {args.output}")


if __name__ == "__main__":
    main()
