import asyncio
import numpy as np
import os
import sys
import tqdm
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw


DESKTOP_DIR_PATH = (
	os.path.join(os.environ['USERPROFILE'], 'Desktop')
	if os.name == 'nt' else
	os.path.join(os.environ['HOME'], 'Desktop'))


def init_progress_bar(total, update_iter=1) -> None:
	global PROGRESS_BAR
	PROGRESS_BAR = tqdm.tqdm(total=total, mininterval=0, miniters=update_iter)
	PROGRESS_BAR.refresh()
	sys.stdout.flush()


def update_progress_bar(step=1) -> None:
	PROGRESS_BAR.update(step)
	sys.stdout.flush()


def close_progress_bar() -> None:
	PROGRESS_BAR.close()
	sys.stdout.flush()


async def swirl_img(image: Image.Image, strength: float, radius: int) -> Image.Image:
	img = np.array(image)
	h, w = img.shape[:2]
	cx, cy = w//2, h//2
	Y, X = np.indices((h, w))
	Xc = X - cx
	Yc = Y - cy
	r = np.sqrt(Xc**2 + Yc**2)
	theta = np.arctan2(Yc, Xc)
	swirl_strength = strength * (radius - r) / radius
	swirl_strength[~(r < radius)] = 0
	theta_new = theta + swirl_strength
	Xs = np.clip((r * np.cos(theta_new) + cx).astype(int), 0, w - 1)
	Ys = np.clip((r * np.sin(theta_new) + cy).astype(int), 0, h - 1)
	swirled = np.zeros_like(img)
	swirled[Y, X] = img[Ys, Xs]
	return Image.fromarray(swirled)


async def draw_arc(
	img: Image.Image,
	x: int,
	y: int,
	radius: int,
	color: tuple[int, int, int],
	angles: int,
	i: int
) -> None:
	deg = angles * i
	deg_end = deg+angles
	draw = ImageDraw.Draw(img, img.mode)
	while radius > 0:
		draw.arc((x - radius, y - radius, x + radius, y + radius), deg, deg_end, color, angles)
		radius -= 1
	update_progress_bar()


async def main_coro(size: int = 1024):
	colors = (
		(255, 0, 0),
		(255, 127, 0),
		(255, 255, 0),
		(127, 255, 0),
		(0, 255, 0),
		(0, 255, 127),
		(0, 255, 255),
		(0, 127, 255),
		(0, 0, 255),
		(127, 0, 255),
		(255, 0, 255),
		(255, 0, 127),
	)
	colors = colors*5
	colors_len = len(colors)
	gif_path = os.path.join(DESKTOP_DIR_PATH, 'image.gif')
	gif_fps = 30
	gif: list[Image.Image] = []
	angles = 360//colors_len

	with Image.new('RGB', (size*2, size*2), (0, 0, 0)) as img:
		radius = size//2
		init_progress_bar(colors_len)
		tasks = tuple(
			draw_arc(img, size, size, radius, colors[i], angles, i)
			for i in range(0, colors_len))
		await asyncio.gather(*tasks)
		img = await swirl_img(img, 2, radius)
		close_progress_bar()

		init_progress_bar(colors_len)
		for i in range(colors_len):
			gif.append(img
				.rotate(360//colors_len * i, resample=Image.Resampling.BICUBIC)
				.resize((size, size), resample=Image.Resampling.LANCZOS)
				.quantize(method=Image.Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG))
			update_progress_bar()
		close_progress_bar()

	gif.pop(0).save(
		gif_path,
		'GIF',
		append_images=gif,
		disposal=1,
		duration=1000//gif_fps,
		optimize=False,
		loop=0,
		save_all=True)
	sys.stdout.write(f'Saved GIF as: {gif_path}\n')
	sys.stdout.flush()
	os.system(f'start "" "{gif_path}"')


def main() -> int:
	asyncio.run(main_coro(512), debug=False)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
