#!/usr/bin/env python3
import re
import sys
import math
import logging
import argparse
import hashlib
import json

from pathlib import Path
from enum import Enum, auto
from functools import cached_property, lru_cache

from pypdf import PdfReader
from pypdf.generic import RectangleObject

from _version import __version__
__author__ = "Grant Hernandez"

log = logging.getLogger()

def serializer(o):
    if hasattr(o, '__json__'):
        return getattr(o, '__json__')()

    return json.JSONEncoder.default(o, o)

class PDFObject:
    def __init__(self, ctm):
        # CompressedTransformationMatrix
        self.ctm = ctm

    @property
    def xy(self):
        # NOTE: this does not take into account rotations or scaling!
        return (self.ctm[4], self.ctm[5])

    def __repr__(self):
        return "<PdfObject x=%.2f, y=%.2f>" % (self.xy[0], self.xy[1])

    def dist(self, other):
        return math.dist(self.xy, other.xy)

class TextObject(PDFObject):
    def __init__(self, ctm, text):
        super().__init__(ctm)
        self.text = text

    def __repr__(self):
        return "<PdfTextObject text='%s', x=%.2f, y=%.2f>" %(self.text, self.xy[0], self.xy[1])

    def __json__(self):
        return self.text

class ImageObject(PDFObject):
    def __init__(self, ctm, name, image):
        super().__init__(ctm)
        self.name = name
        self.image = image
        self._obj = image.indirect_reference.get_object()

        self._bbox = self._calc_bbox(image.image.width, image.image.height)

        # Display width / height

    @property
    def xy(self):
        return self.center

    @property
    def bbox(self):
        return self._bbox

    @property
    def height(self):
        return self._bbox.height

    @property
    def width(self):
        return self._bbox.width

    @property
    def center(self):
        return (self._bbox.left + self.width/2, self._bbox.bottom + self.height/2)

    # Check out to learn more about PDF commands https://stackoverflow.com/a/68032712
    def _calc_bbox(self, width, height):
        width_user = self.ctm[0] + self.ctm[2]
        height_user = self.ctm[1] + self.ctm[3]

        (x0, y0) = (self.ctm[4], self.ctm[5])
        (x1, y1) = (x0 + width_user, y0 + height_user)

        bl = [0, 0]
        tr = [0, 0]

        if x0 < x1:
            bl[0] = x0
            tr[0] = x1
        else:
            bl[0] = x1
            tr[0] = x0

        if y0 < y1:
            bl[1] = y0
            tr[1] = y1
        else:
            bl[1] = y1
            tr[1] = y0

        bbox = RectangleObject(bl + tr)

        #log.info("Image %s: w=%d, h=%d, bl=(%.2f, %.2f), tr=(%.2f, %.2f), bbox=%s, ctm=%s",
                #self.name, width, height,
                #bbox,
                #self.ctm)

        return bbox

    def __repr__(self):
        return "<PdfImageObject name='%s', bbox=%s>" % (self.name, self.bbox)

# (min, max)
def get_bounded_text_objects(objects, x=None, y=None):
    text_objects = []

    for obj in objects:
        if not isinstance(obj, TextObject):
            continue

        xy = obj.xy

        if x is not None and not (xy[0] >= x[0] and xy[0] < x[1]):
            continue

        if y is not None and not (xy[1] >= y[0] and xy[1] < y[1]):
            continue

        text_objects.append(obj)

    return text_objects

def get_matching_text_objects(objects, pattern=r'.*', invert=False):
    text_objects = []

    pat = re.compile(pattern)

    for obj in objects:
        if not isinstance(obj, TextObject):
            continue

        t = obj.text

        m = pat.match(t)

        if invert and m:
            continue

        if not invert and not m:
            continue

        text_objects.append(obj)

    return text_objects

def get_nonempty_text_objects(objects):
    return get_matching_text_objects(objects, pattern=r'^\s*$', invert=True)

class WMATAPageObject:
    def __init__(self, page_no, page):
        self.page_no = page_no
        self.page = page

    @cached_property
    def page_title(self):
        artbox = self.page.artbox
        left, bottom, right, top = artbox

        objects = self.get_page_text()

        # Find highest text object that is near the header which is the title
        page_title = get_matching_text_objects(
            get_bounded_text_objects(objects, y=(top-40, top)),
            pattern=r'^\s*$',
            invert=True
            )

        if len(page_title) > 1:
            log.error("Multiple page titles extracted: %s", page_title)
            return
        elif len(page_title) == 0:
            log.error("No page title found")
            return
        else:
            page_title = page_title[0]
            return page_title

    @lru_cache
    def get_page_text(self, cluster=False):
        objects = []

        # PDF transformation matrix origin is bottom-left
        def visit_text(text, ctm, tm, font_dict, font_size):
            #print(text, ctm, tm)
            objects.append(TextObject(ctm, text))

        self.page.extract_text(visitor_text=visit_text)

        if cluster:
            new_objects = list(objects)

            while True:
                changed = False

                for obj in new_objects:
                    similar = []

                    for obj2 in new_objects:
                        if obj == obj2:
                            continue

                        if obj.dist(obj2) < 0.1:
                            similar.append(obj2)

                    if not similar:
                        continue

                    # Coalese text objects, keep parent xy
                    new_text = obj.text

                    for txt in similar:
                        new_text += txt.text
                        new_objects.remove(txt)

                    new_objects.remove(obj)

                    new_objects = [TextObject(obj.ctm, new_text)] + new_objects
                    changed = True
                    break

                if not changed:
                    break

            objects = new_objects

        return objects

    @lru_cache
    def get_drawn_images(self):
        images_referenced = {}

        for name, img in self.page.images.items():
            #obj = img.indirect_reference.get_object()
            images_referenced[name] = img

        objects = []

        references = []

        # PDF transformation matrix origin is bottom-left
        def visit_oper(operator, operands, cm, tm):
            if operator == b"Do":
                assert len(operands) == 1
                image_name = operands[0]
                assert image_name in images_referenced, "Drawn but not referenced image %s" % (image_name)
                references.append(ImageObject(cm, image_name, images_referenced[image_name]))

        self.page.extract_text(visitor_operand_before=visit_oper)

        return references

    def get_ranked_text_captions(self):
        text_objects = get_nonempty_text_objects(self.get_page_text(cluster=True))
        image_objects = self.get_drawn_images()

        # For each image object, get the distances to every other text object
        ranked_text = {}

        # O(n^2) but object count *should* be small
        for img in image_objects:
            ranked_text[img] = sorted(map(lambda x: (x.dist(img), x), text_objects), key=lambda x: x[0])

        return ranked_text

    @cached_property
    def get_metro_lines(self):
        artbox = self.page.artbox
        left, bottom, right, top = artbox
        height = top - bottom

        inner_platform = False
        objects = self.get_page_text()

        # Some metro lines are surrounded by the station platform, others surround the platform
        title_obj = self.page_title

        title_top = title_obj.xy[1]

        left_line = get_nonempty_text_objects(
                get_matching_text_objects(
            get_bounded_text_objects(objects, y=(title_top-50, title_top-1)),
            pattern=r'^\s*[0-9,\s]+\s*$',
            invert=True
            ))

        # Inner platform
        if len(left_line) == 1:
            left_line = left_line[0]
            inner_platform = True
        elif len(left_line) > 1:
            log.error("Multiple text objects when searching for left line: %s", left_line)
            return
        else:
            # Outer platform
            inner_platform = False

            left_line = get_nonempty_text_objects(
                    get_matching_text_objects(
                get_bounded_text_objects(objects, y=(height/2-40, height/2 + 40)),
                pattern=r'^\s*[0-9,\s]+\s*$',
                invert=True
                ))

            # get highest
            if len(left_line) == 2:
                if left_line[0].xy[1] > left_line[1].xy[1]:
                    left_line = left_line[0]
                else:
                    left_line = left_line[1]
            elif len(left_line) == 1:
                left_line = left_line[0]
            else:
                log.error("Could not also find inner lines")
                return

        if inner_platform:
            right_line = get_nonempty_text_objects(
                    get_matching_text_objects(
                        get_bounded_text_objects(objects, y=(0, 40)),
                        pattern=r'^\s*[0-9,\s]+\s*$',
                        invert=True
                    )
                )
        else:
            right_line = get_nonempty_text_objects(
                    get_matching_text_objects(
                        get_bounded_text_objects(objects, y=(left_line.xy[1]-40, left_line.xy[1]-1)),
                        pattern=r'^\s*[0-9,\s]+\s*$',
                        invert=True
                    )
                )

        if len(right_line) == 1:
            right_line = right_line[0]
        else:
            log.error("Failed to get right line: %s", right_line)
            return

        return {"inner": inner_platform, "left": left_line, "right": right_line}

class IndicatorType(Enum):
    ELEVATOR = auto()
    STAIR = auto()
    ESCALATOR = auto()
    EXIT = auto()

class Indicator:
    def __init__(self, ty, pos, img_obj, caption=""):
        self.ty = ty
        self.pos = pos
        self.img_obj = img_obj
        self.caption = ""

    def __repr__(self):
        return "Indicator<%s @ (%.2f, %.2f), caption=%r>" % (self.ty.name, self.pos[0], self.pos[1], self.caption)

    def __json__(self):
        return {"type": str(self.ty.name), "pos": self.pos, "caption": self.caption}

class WMATAStation:
    def __init__(self, ctx, wpage):
        self._ctx = ctx
        self._wpage = wpage

    def get_indicators(self):
        indicator_obj = []
        indicators = self._ctx._indicators

        for img in self._wpage.get_drawn_images():
            if img.name not in indicators:
                continue

            type_name = indicators[img.name]
            indicator_obj.append(Indicator(indicators[img.name], img.xy, img, ""))

        ranked_text = self._wpage.get_ranked_text_captions()

        ignore_text = [self.name, self.get_metro_lines()["left"], self.get_metro_lines()["right"]]

        # Determine captions, if any
        for ind in indicator_obj:
            #print("SEARCHING", ind)
            assert ind.img_obj in ranked_text

            candidates = ranked_text[ind.img_obj]
            filtered_candidates = []

            for dist, text_obj in candidates:
                text = text_obj.text

                should_ignore = False
                # We coalesed text objects. need to have a disregard radius instead of exact caption
                for ignore in ignore_text:
                    if ignore.dist(text_obj) < 4.0:
                        should_ignore = True
                        break

                if should_ignore:
                    continue

                # ignore light pair numbers
                if re.match(r'^[0-9],?\s*$', text.strip()):
                    continue

                filtered_candidates.append((dist, text_obj))
                #print("- CANDIDATE", dist, text_obj)

            if len(filtered_candidates) == 0:
                continue

            # if the nearest candidate is too far, assume no caption
            dist, text_obj = filtered_candidates[0]
            if dist > 150:
                continue

            ind.caption = filtered_candidates[0][1].text

            #for dist, text_obj in filtered_candidates:



        return indicator_obj

    @property
    def name(self):
        return self._wpage.page_title

    def get_metro_lines(self):
        return self._wpage.get_metro_lines

class WMATAPDFContext:
    def __init__(self):
        self._legend_page = None
        self._stations = []
        self._indicators = {}
        self._wpages = []
        self._reader = None

    def build(self, reader):
        self._reader = reader

        log.info("Processing PDF with %d pages", len(reader.pages))

        self._wpages = self._build_page_list()

        # Get legend page
        legend_page = list(filter(lambda x: "station diagram" in x.page_title.text.lower(), self._wpages))
        assert len(legend_page) == 1
        legend_page = legend_page[0]

        # Railcar page (last page before station pages)
        last_legend_page = list(filter(lambda x: "railcar door" in x.page_title.text.lower(), self._wpages))
        assert len(last_legend_page) == 1
        last_legend_page = last_legend_page[0]

        self._stations = list(map(lambda x: WMATAStation(self, x), self._wpages[last_legend_page.page_no+1:]))
        self._legend_page = legend_page

        self._indicators = self._build_image_map()

    def _build_page_list(self):
        wpages = []

        # First find all of the pages titles to split which pages are stations and guide pages
        for page_no, page in enumerate(self._reader.pages):
            wpage = WMATAPageObject(page_no, page)

            title = wpage.page_title

            if not title:
                return

            log.info("Page %d - title: %s", page_no+1, title)
            wpages.append(wpage)

        return wpages

    def _build_image_map(self):
        # For each image object, get the distances to every other text object
        ranked_text = self._legend_page.get_ranked_text_captions()

        image_map = {}

        for img, text_objects in ranked_text.items():
            # get top choice
            dist, obj = text_objects[0]
            caption = obj.text.strip()

            log.info("Identified %s with caption %s", img.name, caption)

            if img.name not in image_map:
                if caption.lower() == "elevator":
                    ty = IndicatorType.ELEVATOR
                elif caption.lower() == "exit":
                    ty = IndicatorType.EXIT
                elif caption.lower() == "escalator":
                    ty = IndicatorType.ESCALATOR
                elif caption.lower() == "stair":
                    ty = IndicatorType.STAIR
                else:
                    assert 0, "Unhandled indicator caption %s" % (caption)

                image_map[img.name] = ty

        return image_map

    def iter_stations(self):
        for station in self._stations:
            yield station

    def get_station_by_name(self, name):
        for s in self.iter_stations():
            if s.name.text == name:
                return s

def main():
    logging.basicConfig(stream=sys.stdout, format="[%(levelname)s] %(message)s", level=logging.INFO)

    print("WMATA Exit Guide PDF Parser v%s\n  Author: %s\n" % (__version__, __author__))

    parser = argparse.ArgumentParser()

    parser.add_argument("wmata_pdf", type=Path, help="The path to the rendered WMATA Metro Station Platform Exit Guide PDF file")
    parser.add_argument("--json-output",
            type=Path,
            default=Path("./wmata.json"),
            help="The output JSON file")
    args = parser.parse_args()

    reader = PdfReader(args.wmata_pdf)

    wctx = WMATAPDFContext()
    wctx.build(reader)

    output = {
        "version": __version__,
        "src_hash": hashlib.sha256(open(args.wmata_pdf, 'rb').read()).hexdigest(),
        "stations": []
    }

    for station in wctx.iter_stations():
        lines = station.get_metro_lines()

        log.info("Station: %s", station.name)
        log.info(" - Platform inner: %s", lines["inner"])
        log.info(" - Left line: %s", lines["left"])
        log.info(" - Right line: %s", lines["right"])

        indicators = station.get_indicators()
        for ind in indicators:
            log.info("   * %s", ind)

        station_info = {
            "name": station.name,
            "lines": lines,
            "indicators": indicators
        }

        output["stations"].append(station_info)

    with open(args.json_output, 'w') as fp:
        json.dump(output, fp, indent=2, default=serializer)

    log.info("Wrote to %s", str(args.json_output))


if __name__ == "__main__":
    main()
