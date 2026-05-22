# TODO : should be able to not open all row data of large tiles versions
import sys
import os
import os.path
import argparse
import pathlib
from collections import namedtuple
import heapq
from PIL import Image, ImageOps
from multiprocessing import Process, Queue, cpu_count


Image.MAX_IMAGE_PIXELS = (
    1620 * 1920 * (4**4)
)  # Up the maximum size limit allowed by PIL

# DEFAULT parameters
DEFAULT_TILE_SIZE = 50  # height/width of mosaic tiles in pixels
DEFAULT_TILE_MATCH_RES = 5  # tile matching resolution (higher values give better fit but require more processing)
DEFAULT_ENLARGEMENT = (
    8  # the mosaic image will be this many times wider and taller than the original
)

DEFAULT_WORKER_COUNT = max(cpu_count(), 2)
DEFAULT_OUT_FILE = "mosaic.jpeg"
DEFAULT_VARIETY = 40
EOQ_VALUE = None

# Parameter structure
Parameters = namedtuple(
    "Parameters",
    [
        "source_image",
        "tiles_directory",
        "output_file",
        "tile_size",
        "match_res",
        "enlargement",
        "tile_block_size",
        "worker_count",
        "variety",
    ],
    defaults=[
        None,
        None,
        DEFAULT_OUT_FILE,
        DEFAULT_TILE_SIZE,
        DEFAULT_TILE_MATCH_RES,
        DEFAULT_ENLARGEMENT,
        None,
        DEFAULT_WORKER_COUNT,
        DEFAULT_VARIETY,
    ],
)


class TileProcessor:
    def __init__(self, parameters):
        self.params = parameters

    def __process_tile(self, tile_path):
        try:
            img = Image.open(tile_path)
            img = ImageOps.exif_transpose(img)

            # tiles must be square, so get the largest square that fits inside the image
            w = img.size[0]
            h = img.size[1]
            min_dimension = min(w, h)
            w_crop = (w - min_dimension) / 2
            h_crop = (h - min_dimension) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))

            large_tile_img = img.resize(
                (self.params.tile_size, self.params.tile_size), Image.LANCZOS
            )
            small_tile_img = img.resize(
                (
                    int(self.params.tile_size / self.params.tile_block_size),
                    int(self.params.tile_size / self.params.tile_block_size),
                ),
                Image.LANCZOS,
            )

            return (large_tile_img.convert("RGB"), small_tile_img.convert("RGB"))
        except:
            return (None, None)

    def read_tiles(self, parse_queue, parsed_queue):
        while True:
            try:
                root, tile_name = parse_queue.get(True)
                if root == EOQ_VALUE:
                    break
                tile_path = os.path.join(root, tile_name)
                print("Reading {:40.40}".format(tile_name), flush=True, end="\r")
                large_tile, small_tile = self.__process_tile(tile_path)
                if large_tile and small_tile:
                    parsed_queue.put(
                        (
                            large_tile,
                            list(small_tile.get_flattened_data()),
                        )
                    )
            except KeyboardInterrupt:
                pass
        parsed_queue.put((EOQ_VALUE, EOQ_VALUE))
        return

    def get_tiles(self):
        # Multi core images parsing
        parse_queue = Queue()
        parsed_queue = Queue()
        processes = []
        try:
            large_tiles = []
            small_tiles = []

            print("Reading tiles from {}...".format(self.params.tiles_directory))

            # start the reader processes
            for _ in range(self.params.worker_count):
                p = Process(target=self.read_tiles, args=(parse_queue, parsed_queue))
                processes.append(p)
                p.start()

            # search the tiles directory recursively
            for root, _, files in os.walk(self.params.tiles_directory):
                for tile_name in files:
                    parse_queue.put((root, tile_name))

            # Send stop signal to readers
            for _ in range(self.params.worker_count):
                parse_queue.put((EOQ_VALUE, EOQ_VALUE))

            # collect images
            workingParsers = self.params.worker_count
            while workingParsers > 0:
                large_t, small_t = parsed_queue.get()
                if large_t == EOQ_VALUE:
                    workingParsers -= 1
                    continue
                large_tiles.append(large_t)
                small_tiles.append(small_t)

        except KeyboardInterrupt:
            print("\nStopping parse processes")
            for p in processes:
                p.kill()
            parse_queue.cancel_join_thread()
            parsed_queue.cancel_join_thread()
            sys.exit(130)  # terminate with exit from SIGKILL

        print("Processed {} tiles.".format(len(large_tiles)), flush=True)

        return (large_tiles, small_tiles)


class TargetImage:
    def __init__(self, parameters):
        self.params = parameters

    def get_data(self):
        print("Processing main image...")
        img = Image.open(self.params.source_image)
        w = img.size[0] * self.params.enlargement
        h = img.size[1] * self.params.enlargement
        new_size = w * h
        if new_size > Image.MAX_IMAGE_PIXELS:
            print(
                "Main image is too heavy after resizing ({} pixels for "
                "a max size of {} pixels), consider "
                'retrying with a smaller "enlarge" parameter'.format(
                    new_size, Image.MAX_IMAGE_PIXELS
                )
            )
            sys.exit(1)
            raise Image.DecompressionBombError("Too large after resize")
        large_img = img.resize((w, h), Image.LANCZOS)
        w_diff = (w % self.params.tile_size) / 2
        h_diff = (h % self.params.tile_size) / 2

        # if necessary, crop the image slightly so we use a whole number of tiles horizontally and vertically
        if w_diff or h_diff:
            large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))

        small_img = large_img.resize(
            (
                int(w / self.params.tile_block_size),
                int(h / self.params.tile_block_size),
            ),
            Image.LANCZOS,
        )

        image_data = (large_img.convert("RGB"), small_img.convert("RGB"))

        print("Main image processed.")

        return image_data


class TileFitHeap:
    """
    A class to keep a heap of best tile index based on diff score
    elements are : (diff_val, tile_index)
    n is the maximum number of tiles
    """

    def __init__(self, n):
        self.size = n
        self.heap = []

    def add(self, score_element):
        if len(self.heap) < self.size:
            heapq.heappush_max(self.heap, score_element)
        else:
            if self.heap[0][0] > score_element[0]:
                heapq.heapreplace_max(self.heap, score_element)

    def max_score(self):
        if len(self.heap) < self.size:
            return sys.maxsize
        else:
            return self.heap[0][0]

    def get_ordered_best_fits(self):
        return sorted(self.heap)


class TileFitter:
    def __init__(self, tiles_data, parameters):
        self.tiles_data = tiles_data
        self.params = parameters

    def __get_tile_diff(self, t1, t2, bail_out_value):
        diff = 0
        for i in range(len(t1)):
            # diff += (abs(t1[i][0] - t2[i][0]) + abs(t1[i][1] - t2[i][1]) + abs(t1[i][2] - t2[i][2]))
            diff += (
                (t1[i][0] - t2[i][0]) ** 2
                + (t1[i][1] - t2[i][1]) ** 2
                + (t1[i][2] - t2[i][2]) ** 2
            )
            #             m1 = (t1[i][0]+t1[i][1]+t1[i][2])/3
            #             m2 = (t2[i][0]+t2[i][1]+t2[i][2])/3
            #             diff += abs(m1-m2)
            if diff > bail_out_value:
                # we know already that this isn't going to be the best fit, so no point continuing with this tile
                return diff
        return diff

    def __get_tile_diff_lum(self, t1, t2, bail_out_value):
        """
        get best fit tile based on luminance value
        """
        diff = 0
        for i in range(len(t1)):
            diff += (
                (0.2126 * t1[i][0] + 0.7152 * t1[i][1] + 0.0722 * t1[i][2])
                - (0.2126 * t2[i][0] + 0.7152 * t2[i][1] + 0.0722 * t2[i][2])
            ) ** 2
            if diff > bail_out_value:
                # we know already that this isn't going to be the best fit, so no point continuing with this tile
                return diff
        return diff

    def __get_tile_diff_lum_color(self, t1, t2, bail_out_value):
        """
        get best fit tile based on limunance value
        with a small color factor
        """
        diff = 0
        for i in range(len(t1)):
            diff += (
                (
                    (0.2126 * t1[i][0] + 0.7152 * t1[i][1] + 0.0722 * t1[i][2])
                    - (0.2126 * t2[i][0] + 0.7152 * t2[i][1] + 0.0722 * t2[i][2])
                )
                ** 2
            ) / 2  # luminance part
            diff += (
                abs(t1[i][0] - t2[i][0]) ** 2
                + abs(t1[i][1] - t2[i][1]) ** 2
                + abs(t1[i][2] - t2[i][2]) ** 2
            )  # color part

            if diff > bail_out_value:
                # we know already that this isn't going to be the best fit, so no point continuing with this tile
                return diff
        return diff

    def get_best_fit_tile(self, img_data):
        best_fit_tile_index = None
        min_diff = sys.maxsize
        tile_index = 0

        # go through each tile in turn looking for the best match for the part of the image represented by 'img_data'
        for tile_data in self.tiles_data:
            #             diff = self.__get_tile_diff(img_data, tile_data, min_diff)
            #             diff = self.__get_tile_diff_lum(img_data, tile_data, min_diff)
            diff = self.__get_tile_diff_lum_color(img_data, tile_data, min_diff)
            if diff < min_diff:
                min_diff = diff
                best_fit_tile_index = tile_index
            tile_index += 1

        return best_fit_tile_index

    def get_best_fit_tiles(self, img_data):
        max_heap = TileFitHeap(self.params.variety)
        tile_index = 0

        # go through each tile in turn looking for the best overall matches
        # bail out value is the value of the wors image in the heap
        for tile_data in self.tiles_data:
            diff = self.__get_tile_diff_lum_color(
                img_data, tile_data, max_heap.max_score()
            )
            if diff < max_heap.max_score():
                max_heap.add((diff, tile_index))
            tile_index += 1

        return [e[1] for e in max_heap.get_ordered_best_fits()]


def fit_tiles(work_queue, result_queue, tiles_data, parameters):
    # this function gets run by the worker processes, one on each CPU core
    tile_fitter = TileFitter(tiles_data, parameters)

    while True:
        try:
            img_data, img_coords = work_queue.get(True)
            if img_data == EOQ_VALUE:
                break
            # tile_index = tile_fitter.get_best_fit_tile(img_data)
            tiles_bests = tile_fitter.get_best_fit_tiles(img_data)
            # result_queue.put((img_coords, tile_index))
            result_queue.put((img_coords, tiles_bests))
        except KeyboardInterrupt:
            pass

    # let the result handler know that this worker has finished everything
    result_queue.put((EOQ_VALUE, EOQ_VALUE))


class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.counter = 0

    def update(self):
        self.counter += 1
        print(
            "Progress: {:04.1f}%".format(100 * self.counter / self.total),
            flush=True,
            end="\r",
        )


class MosaicImage:
    def __init__(self, original_img, parameters):
        self.params = parameters
        self.image = Image.new(original_img.mode, original_img.size)
        self.x_tile_count = int(original_img.size[0] / self.params.tile_size)
        self.y_tile_count = int(original_img.size[1] / self.params.tile_size)
        self.total_tiles = self.x_tile_count * self.y_tile_count

    def add_tile(self, tile_data, coords):
        img = Image.new("RGB", (self.params.tile_size, self.params.tile_size))
        img.putdata(tile_data.get_flattened_data())
        self.image.paste(img, coords)

    def save(self, path):
        self.image.save(path)


def build_mosaic(result_queue, all_tile_data_large, original_img_large, parameters):
    mosaic = MosaicImage(original_img_large, parameters)
    active_workers = parameters.worker_count
    while True:
        try:
            img_coords, best_fit_tile_index = result_queue.get()

            if img_coords == EOQ_VALUE:
                active_workers -= 1
                if not active_workers:
                    break
            else:
                tile_data = all_tile_data_large[best_fit_tile_index]
                mosaic.add_tile(tile_data, img_coords)

        except KeyboardInterrupt:
            pass

    mosaic.save(parameters.output_file)
    print("\nFinished, output is in", parameters.output_file)


def build_mosaic_multi(
    mosaic, result_queue, all_tile_data_large, original_img_large, parameters
):
    active_workers = parameters.worker_count
    usage_tab = [
        0 for _ in range(len(all_tile_data_large))
    ]  # number of times a tile is used
    progress = ProgressCounter(mosaic.x_tile_count * mosaic.y_tile_count)
    while True:
        try:
            img_coords, best_list = result_queue.get()

            if img_coords == EOQ_VALUE:
                active_workers -= 1
                if not active_workers:
                    break
            else:
                # find minimal used tile
                index_in_best = sorted(
                    [(usage_tab[best_list[i]], i) for i in range(len(best_list))]
                )[0][1]
                tile_data = all_tile_data_large[best_list[index_in_best]]
                usage_tab[best_list[index_in_best]] += 1
                mosaic.add_tile(tile_data, img_coords)
                progress.update()

        except KeyboardInterrupt:
            break
    mosaic.save(parameters.output_file)
    print("\nFinished, output is in", parameters.output_file)


def compose(original_img, tiles, parameters):
    print("Building mosaic, press Ctrl-C to abort...")
    original_img_large, original_img_small = original_img
    tiles_large, tiles_small = tiles

    mosaic = MosaicImage(original_img_large, parameters)

    # all_tile_data_large = [list(tile.get_flattened_data()) for tile in tiles_large]
    # all_tile_data_small = [list(tile.get_flattened_data()) for tile in tiles_small]
    all_tile_data_large = tiles_large
    all_tile_data_small = tiles_small

    work_queue = Queue(parameters.worker_count)
    result_queue = Queue()

    try:
        # start the worker processes that will build the mosaic image
        Process(
            target=build_mosaic_multi,
            args=(
                mosaic,
                result_queue,
                all_tile_data_large,
                original_img_large,
                parameters,
            ),
        ).start()

        # start the worker processes that will perform the tile fitting
        for _ in range(parameters.worker_count):
            Process(
                target=fit_tiles,
                args=(work_queue, result_queue, all_tile_data_small, parameters),
            ).start()

        for x in range(mosaic.x_tile_count):
            for y in range(mosaic.y_tile_count):
                large_box = (
                    x * parameters.tile_size,
                    y * parameters.tile_size,
                    (x + 1) * parameters.tile_size,
                    (y + 1) * parameters.tile_size,
                )
                small_box = (
                    x * parameters.tile_size / parameters.tile_block_size,
                    y * parameters.tile_size / parameters.tile_block_size,
                    (x + 1) * parameters.tile_size / parameters.tile_block_size,
                    (y + 1) * parameters.tile_size / parameters.tile_block_size,
                )
                work_queue.put(
                    (
                        list(original_img_small.crop(small_box).get_flattened_data()),
                        large_box,
                    )
                )

    except KeyboardInterrupt:
        print("\nHalting, saving partial image please wait...")

    finally:
        # put these special values onto the queue to let the workers know they can terminate
        for _ in range(parameters.worker_count):
            work_queue.put((EOQ_VALUE, EOQ_VALUE))


def show_error(msg):
    print("ERROR: {}".format(msg))


def mosaic(parameters):
    image_data = TargetImage(parameters).get_data()
    tiles_data = TileProcessor(parameters).get_tiles()
    if tiles_data[0]:
        compose(image_data, tiles_data, parameters)
    else:
        show_error(
            "No images found in tiles directory '{}'".format(parameters.tiles_directory)
        )


def main(argv):
    parser = argparse.ArgumentParser(
        prog="mosaic", description="A simple mosaic creator programm"
    )
    parser.add_argument("image", type=str, help="Input image to transform")
    parser.add_argument(
        "tiles_directory", type=pathlib.Path, help="Directory for the tiles data"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="The output image", default=DEFAULT_OUT_FILE
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        help=f"The number of threads to use (computed default is {DEFAULT_WORKER_COUNT})",
        default=DEFAULT_WORKER_COUNT,
    )
    parser.add_argument(
        "--tilesize",
        "-ts",
        type=int,
        help=f"The size (in pixels) of the tiles (default is {DEFAULT_TILE_SIZE})",
        default=DEFAULT_TILE_SIZE,
    )
    parser.add_argument(
        "--tileres",
        "-tr",
        type=int,
        help=f"Tile matching resolution (default is {DEFAULT_TILE_MATCH_RES})",
        default=DEFAULT_TILE_MATCH_RES,
    )
    parser.add_argument(
        "--enlarge",
        "-r",
        type=int,
        help=f"The size of the resulting image X times the original (default is {DEFAULT_ENLARGEMENT})",
        default=DEFAULT_ENLARGEMENT,
    )
    parser.add_argument(
        "--variety",
        "-vr",
        type=int,
        help=f"The variety parameter to avoid reusing the same tile (default is {DEFAULT_VARIETY})",
        default=DEFAULT_VARIETY,
    )

    args = parser.parse_args(argv)
    # print(args)

    # sets up parameters
    params = Parameters(
        source_image=args.image,
        tiles_directory=args.tiles_directory,
        output_file=args.output,
        tile_size=args.tilesize,
        match_res=args.tileres,
        enlargement=args.enlarge,
        tile_block_size=args.tilesize / max(min(args.tileres, DEFAULT_TILE_SIZE), 1),
        worker_count=max((args.threads) - 1, 1),
        variety=args.variety,
    )
    # print(params)

    if not os.path.isfile(params.source_image):
        show_error("Unable to find image file '{}'".format(params.source_image))
    elif not os.path.isdir(params.tiles_directory):
        show_error("Unable to find tile directory '{}'".format(params.tiles_directory))
    else:
        mosaic(params)


if __name__ == "__main__":
    main(sys.argv[1:])
