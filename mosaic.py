import sys
import os
import os.path
import argparse
import pathlib
from PIL import Image, ImageOps
from multiprocessing import Process, Queue, cpu_count

# DEFAULT parameters
DEFAULT_TILE_SIZE      = 50     # height/width of mosaic tiles in pixels
DEFAULT_TILE_MATCH_RES = 5      # tile matching resolution (higher values give better fit but require more processing)
DEFAULT_ENLARGEMENT    = 8      # the mosaic image will be this many times wider and taller than the original

DEFAULT_WORKER_COUNT = max(cpu_count(), 2)
DEFAULT_OUT_FILE = 'mosaic.jpeg'
EOQ_VALUE = None

# GLOBAL PARAMETERS
G_TILE_SIZE = None
G_TILE_MATCH_RES = None
G_ENLARGEMENT = None
G_WORKER_COUNT = None
G_OUT_FILE = None

G_TILE_BLOCK_SIZE = None

class TileProcessor:
    def __init__(self, tiles_directory):
        self.tiles_directory = tiles_directory

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

            large_tile_img = img.resize((G_TILE_SIZE, G_TILE_SIZE), Image.LANCZOS)
            small_tile_img = img.resize((int(G_TILE_SIZE/G_TILE_BLOCK_SIZE), int(G_TILE_SIZE/G_TILE_BLOCK_SIZE)), Image.LANCZOS)

            return (large_tile_img.convert('RGB'), small_tile_img.convert('RGB'))
        except:
            return (None, None)

    def __read_tiles(self, parse_queue, parsed_queue):
        while True:
            try:
                root, tile_name = parse_queue.get(True)
                if root == EOQ_VALUE:
                    break
                tile_path = os.path.join(root, tile_name)
                print('Reading {:40.40}'.format(tile_name), flush=True, end='\r')
                large_tile, small_tile = self.__process_tile(tile_path)
                if large_tile:
                    parsed_queue.put((large_tile, small_tile))
            except KeyboardInterrupt:
                pass
        parsed_queue.put((EOQ_VALUE, EOQ_VALUE))
        return

    def get_tiles(self):
        try:
            large_tiles = []
            small_tiles = []

            # Multi core images parsing
            parse_queue = Queue()
            parsed_queue = Queue()

            print('Reading tiles from {}...'.format(self.tiles_directory))

            # start the reader processes
            processes = []
            for _ in range(G_WORKER_COUNT):
                p = Process(target=self.__read_tiles, args=(parse_queue, parsed_queue))
                processes.append(p)
                p.start()

            # search the tiles directory recursively
            for root, subFolders, files in os.walk(self.tiles_directory):
                for tile_name in files:
                    parse_queue.put((root, tile_name))

            # Send stop signal to readers
            for _ in range(G_WORKER_COUNT):
                parse_queue.put((EOQ_VALUE, EOQ_VALUE))

            # collect images
            workingParsers = G_WORKER_COUNT
            while workingParsers > 0:
                large_t, small_t = parsed_queue.get()
                if large_t == EOQ_VALUE:
                    workingParsers -= 1
                    continue
                large_tiles.append(large_t)
                small_tiles.append(small_t)

        except KeyboardInterrupt:
            print('\nStopping parse processes')
            for p in processes:
                p.kill()
            parse_queue.cancel_join_thread()
            parsed_queue.cancel_join_thread()
            sys.exit(130)  # terminate with exit from SIGKILL

        print('Processed {} tiles.'.format(len(large_tiles)), flush=True)

        return (large_tiles, small_tiles)


class TargetImage:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_data(self):
        print('Processing main image...')
        img = Image.open(self.image_path)
        w = img.size[0] * G_ENLARGEMENT
        h = img.size[1] * G_ENLARGEMENT
        large_img = img.resize((w, h), Image.LANCZOS)
        w_diff = (w % G_TILE_SIZE)/2
        h_diff = (h % G_TILE_SIZE)/2

        # if necessary, crop the image slightly so we use a whole number of tiles horizontally and vertically
        if w_diff or h_diff:
            large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))

        small_img = large_img.resize((int(w/G_TILE_BLOCK_SIZE), int(h/G_TILE_BLOCK_SIZE)), Image.LANCZOS)

        image_data = (large_img.convert('RGB'), small_img.convert('RGB'))

        print('Main image processed.')

        return image_data


class TileFitter:
    def __init__(self, tiles_data):
        self.tiles_data = tiles_data

    def __get_tile_diff(self, t1, t2, bail_out_value):
        diff = 0
        for i in range(len(t1)):
            #diff += (abs(t1[i][0] - t2[i][0]) + abs(t1[i][1] - t2[i][1]) + abs(t1[i][2] - t2[i][2]))
            diff += ((t1[i][0] - t2[i][0])**2 + (t1[i][1] - t2[i][1])**2 + (t1[i][2] - t2[i][2])**2)
#             m1 = (t1[i][0]+t1[i][1]+t1[i][2])/3
#             m2 = (t2[i][0]+t2[i][1]+t2[i][2])/3
#             diff += abs(m1-m2)
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
            diff = self.__get_tile_diff(img_data, tile_data, min_diff)
            if diff < min_diff:
                min_diff = diff
                best_fit_tile_index = tile_index
            tile_index += 1

        return best_fit_tile_index


def fit_tiles(work_queue, result_queue, tiles_data):
    # this function gets run by the worker processes, one on each CPU core
    tile_fitter = TileFitter(tiles_data)

    while True:
        try:
            img_data, img_coords = work_queue.get(True)
            if img_data == EOQ_VALUE:
                break
            tile_index = tile_fitter.get_best_fit_tile(img_data)
            result_queue.put((img_coords, tile_index))
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
        print("Progress: {:04.1f}%".format(100 * self.counter / self.total), flush=True, end='\r')


class MosaicImage:
    def __init__(self, original_img):
        self.image = Image.new(original_img.mode, original_img.size)
        self.x_tile_count = int(original_img.size[0] / G_TILE_SIZE)
        self.y_tile_count = int(original_img.size[1] / G_TILE_SIZE)
        self.total_tiles  = self.x_tile_count * self.y_tile_count

    def add_tile(self, tile_data, coords):
        img = Image.new('RGB', (G_TILE_SIZE, G_TILE_SIZE))
        img.putdata(tile_data)
        self.image.paste(img, coords)

    def save(self, path):
        self.image.save(path)


def build_mosaic(result_queue, all_tile_data_large, original_img_large):
    mosaic = MosaicImage(original_img_large)

    active_workers = G_WORKER_COUNT
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

    mosaic.save(G_OUT_FILE)
    print('\nFinished, output is in', G_OUT_FILE)


def compose(original_img, tiles):
    print('Building mosaic, press Ctrl-C to abort...')
    original_img_large, original_img_small = original_img
    tiles_large, tiles_small = tiles

    mosaic = MosaicImage(original_img_large)

    all_tile_data_large = [list(tile.getdata()) for tile in tiles_large]
    all_tile_data_small = [list(tile.getdata()) for tile in tiles_small]

    work_queue   = Queue(G_WORKER_COUNT)
    result_queue = Queue()

    try:
        # start the worker processes that will build the mosaic image
        Process(target=build_mosaic, args=(result_queue, all_tile_data_large, original_img_large)).start()

        # start the worker processes that will perform the tile fitting
        for n in range(G_WORKER_COUNT):
            Process(target=fit_tiles, args=(work_queue, result_queue, all_tile_data_small)).start()

        progress = ProgressCounter(mosaic.x_tile_count * mosaic.y_tile_count)
        for x in range(mosaic.x_tile_count):
            for y in range(mosaic.y_tile_count):
                large_box = (x * G_TILE_SIZE, y * G_TILE_SIZE, (x + 1) * G_TILE_SIZE, (y + 1) * G_TILE_SIZE)
                small_box = (x * G_TILE_SIZE/G_TILE_BLOCK_SIZE, y * G_TILE_SIZE/G_TILE_BLOCK_SIZE, (x + 1) * G_TILE_SIZE/G_TILE_BLOCK_SIZE, (y + 1) * G_TILE_SIZE/G_TILE_BLOCK_SIZE)
                work_queue.put((list(original_img_small.crop(small_box).getdata()), large_box))
                progress.update()

    except KeyboardInterrupt:
        print('\nHalting, saving partial image please wait...')

    finally:
        # put these special values onto the queue to let the workers know they can terminate
        for n in range(G_WORKER_COUNT):
            work_queue.put((EOQ_VALUE, EOQ_VALUE))


def show_error(msg):
    print('ERROR: {}'.format(msg))


def mosaic(img_path, tiles_path):
    image_data = TargetImage(img_path).get_data()
    tiles_data = TileProcessor(tiles_path).get_tiles()
    if tiles_data[0]:
        compose(image_data, tiles_data)
    else:
        show_error("No images found in tiles directory '{}'".format(tiles_path))

def main(argv):
    parser = argparse.ArgumentParser(
            prog="mosaic",
            description="A simple mosaic creator programm"
            )
    parser.add_argument('image', type=str, help="Input image to transform")
    parser.add_argument('tiles_directory', type=pathlib.Path, help="Directory for the tiles data")
    parser.add_argument('--output', '-o', type=str, help="The output image", default=DEFAULT_OUT_FILE)
    parser.add_argument('--threads', '-t', type=int, help="The number of threads to use", default=DEFAULT_WORKER_COUNT)
    parser.add_argument('--tilesize', '-ts', type=int, help="The size (in pixels) of the tiles", default=DEFAULT_TILE_SIZE)
    parser.add_argument('--tileres', '-tr', type=int, help="Tile matching resolution (the level of detail used for tile matching)", default=DEFAULT_TILE_MATCH_RES)
    parser.add_argument('--enlarge', '-r', type=int, help="The size of the resulting image (X times the original)", default=DEFAULT_ENLARGEMENT)

    args = parser.parse_args(argv)
    print(args)

    # setting globals
    global G_TILE_SIZE
    G_TILE_SIZE = args.tilesize
    global G_TILE_MATCH_RES
    G_TILE_MATCH_RES = args.tileres
    global G_ENLARGEMENT
    G_ENLARGEMENT = args.enlarge

    global G_TILE_BLOCK_SIZE
    G_TILE_BLOCK_SIZE = G_TILE_SIZE / max(min(G_TILE_MATCH_RES, DEFAULT_TILE_SIZE), 1)

    global G_WORKER_COUNT
    G_WORKER_COUNT = max((args.threads) - 1, 1)
    global G_OUT_FILE
    G_OUT_FILE = args.output

    source_image = args.image
    tile_dir = args.tiles_directory

    if not os.path.isfile(source_image):
        show_error("Unable to find image file '{}'".format(source_image))
    elif not os.path.isdir(tile_dir):
        show_error("Unable to find tile directory '{}'".format(tile_dir))
    else:
        mosaic(source_image, tile_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
