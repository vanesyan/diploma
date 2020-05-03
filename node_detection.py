import cv2
import numpy as np
from equation import ast
from collections import namedtuple
import ogr
import thinning
import pytesseract
from PIL import Image

Point = namedtuple('Point', 'x y')
EPS = 20
SHAPE_APPROX_EPSILON = 0.03


class Node:
    class Type:
        UNDEFINED = 0
        OPERATOR = 1
        VARIABLE = 2
        OUTPUT = 3

    def __init__(self, shape_index, symbol_index, box):
        self.shape_index = shape_index
        self.symbol_index = symbol_index
        self.type = Node.Type.UNDEFINED
        self.box = box

    def __repr__(self):
        return '<Node shape={} symbol={} box={}>'.format(self.shape_index, self.symbol_index, box)


def is_intersect(r1, r2):
    """
    Checks whether r1 intersects with r2.
    """

    eps = 0 # int(EPS / 2)
    p11, p12 = r1
    p21, p22 = r2

    return (p11.x - eps <= p21.x <= p12.x + eps or
            p11.x - eps <= p22.x <= p12.x + eps) and \
        (p11.y - eps <= p21.y <= p12.y + eps or
            p11.y - eps <= p22.y <= p12.y + eps)


def is_contains(r1, r2):
    """
    Checks whether r1 contains r2
    """

    p11, p12 = r1
    p21, p22 = r2

    return p11.x <= p21.x <= p22.x <= p12.x and p11.y <= p21.y <= p22.y <= p12.y


def detect_shape(contour) -> int:
    p = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(
        contour, SHAPE_APPROX_EPSILON * p, True)
    point_nums = len(approx)

    # Triangle
    if point_nums == 3:
        return Node.Type.OPERATOR
    # Rectangle
    elif point_nums == 4:
        return Node.Type.OUTPUT

    # TODO: find circles.
    # Circle
    else:
        # https://www.sciencedirect.com/science/article/abs/pii/S0031320314001976
        return Node.Type.VARIABLE

    return Node.Type.UNDEFINED


def filter_contours(image, contours):
    rectangles = []

    preprocessed_image = image.copy()

    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = (
            Point(x, y),
            Point(x + w, y + h)
        )
        rectangles.append(rectangle)
        cv2.rectangle(preprocessed_image,
                      rectangle[0], rectangle[1], (0, 255, 0), 1)
        cv2.putText(preprocessed_image, str(index), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Preprocessed image", preprocessed_image)

    i = 0
    l = len(rectangles)
    klist = np.zeros(l, dtype=np.int16)
    potential_nodes = []

    while i < l - 1:
        r1 = rectangles[i]

        j = i + 1
        while j < l:
            r2 = rectangles[j]

            if is_contains(r1, r2):
                # print("contains", (i, j))

                klist[i] -= 1
                potential_nodes.append(Node(i, j, r1))
            elif is_contains(r2, r1):
                # print("contains", (j, i))

                klist[j] -= 1
                potential_nodes.append(Node(j, i, r1))
            if is_intersect(r1, r2) or is_intersect(r2, r1):
                p11, p12 = r1
                p21, p22 = r2

                s1 = abs(p11.x - p12.x) * abs(p11.y - p12.y)
                s2 = abs(p21.x - p22.x) * abs(p21.y - p22.y)

                if s1 > s2:
                    # print("intersects", (i, j))
                    klist[i] += 1
                else:
                    # print("intersects", (j, i))
                    klist[j] += 1
            j += 1
        i += 1

    filtered_contours = []

    # Remove intersecting regions with big areas.
    for index, count in enumerate(klist):
        if count > 2:
            potential_nodes = [
                node for node in potential_nodes if node.shape_index != index]
        else:
            filtered_contours.append(index)

    # Debug
    prefinal_contours = []
    seen = set()
    for node in potential_nodes:
        if not node.shape_index in seen:
            prefinal_contours.append(contours[node.shape_index])
            seen.add(node.shape_index)

    postprocessed_image = image.copy()

    # Debug
    for index, contour in enumerate(prefinal_contours):
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(postprocessed_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(postprocessed_image, str(index), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Postprocessed image", postprocessed_image)

    return prefinal_contours


def preprocessing(image):
    pixels = image.copy()

    processed_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    processed_pixels = cv2.GaussianBlur(processed_pixels, (3, 3), 0)
    _, processed_pixels = cv2.threshold(
        processed_pixels, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # edged = cv2.Canny(processed_pixels, 50, 100)

    # threshold = cv2.adaptiveThreshold(processed_pixels, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)

    # Detect contours
    nodes, _ = cv2.findContours(
        processed_pixels, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    nodes = sorted(nodes, key=lambda c: cv2.contourArea(c))

    nodes.pop()

    prefinal_contours = filter_contours(
        pixels.copy(), nodes)

    vertices_pixels = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    vertices_pixels = cv2.cvtColor(cv2.drawContours(vertices_pixels, prefinal_contours, -1,
                                             color=(255, 255, 255), thickness=cv2.FILLED), cv2.COLOR_BGR2GRAY)

    # Binary openning (first pass)
    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(9, 9))
    # vertices_pixels = cv2.morphologyEx(vertices_pixels, cv2.MORPH_CLOSE, kernel)

    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(7, 7))
    # vertices_pixels = cv2.dilate(vertices_pixels, kernel)

    # Fill nodes.
    processed_pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    processed_pixels = cv2.blur(processed_pixels, (3, 3))
    _, processed_pixels = cv2.threshold(
        processed_pixels, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    vertices_pixels_floodfill = vertices_pixels.copy()
    h, w = vertices_pixels_floodfill.shape
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Floodfill from point (0, 0).
    cv2.floodFill(vertices_pixels_floodfill, mask, (0, 0), 255)
    # cv2.imshow("Vertices pixels filled", vertices_pixels_floodfill)

    # Invert floodfilled image
    vertices_pixels_floodfill_inv = np.bitwise_not(vertices_pixels_floodfill)
    # cv2.imshow("Vertices pixels filled inverted", vertices_pixels_floodfill_inv)

    vertices_pixels = vertices_pixels | vertices_pixels_floodfill_inv
    processed_pixels = vertices_pixels | processed_pixels

    cv2.imshow("Vertices pixels", vertices_pixels)
    cv2.imshow("Processed image", processed_pixels)

    return processed_pixels, vertices_pixels


def recongnize_char():
    images = np.load('data/chars_glyphs.npy')
    labels = np.load('data/chars_labels.npy')

    from sys import maxsize
    np.set_printoptions(threshold=maxsize)

    # print(labels)


def detect_labels(image, vertices_pixels):
    nodes, _ = cv2.findContours(
        vertices_pixels, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    symbols = {} # symbols
    w, h, _ = image.shape
    test_polygon = np.zeros((w, h, 3), dtype=np.uint8)
    connected_components = np.zeros((w, h), dtype=np.uint32)

    debug_image = image.copy()
    processing_image = image.copy()

    processing_image = cv2.cvtColor(processing_image, cv2.COLOR_BGR2GRAY)
    # processing_image = cv2.blur(processing_image, (3, 3))
    _, processing_image = cv2.threshold(
        processing_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    result = cv2.copyTo(processing_image, vertices_pixels)
    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5,5))
    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    chars, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(chars)

    for node in chars:
        x, y, w, h = cv2.boundingRect(node)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow("Masked vertices", result)

    for index, node in enumerate(nodes):
        name = index+1
        x, y, w, h = cv2.boundingRect(node)

        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(debug_image, str(name), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        processing_image = cv2.drawContours(processing_image, [node], 0,
                         color=(255, 255, 255), thickness=10)
        # rect = processing_image[(y+10):(y+h-10), (x+10):(x+w-10)]


        # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))
        # rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)
        # rect = ~cv2.dilate(~rect, kernel)
        # kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5,5))
        # rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)

        # rect = Image.fromarray(rect)

        # rect.save(str(y)+str(x)+'.png')

        test_polygon = cv2.drawContours(test_polygon, [node], 0,
                         color=(1, 0, 0), thickness=cv2.FILLED)
        x, y = np.where(test_polygon[:, :, 0] == 1)
        # points = np.stack((x, y), axis=-1)

        connected_components[x, y] = name
        node_type = detect_shape(node)
        symbols[name] = (node_type, '1')

        # Clean Up
        test_polygon = test_polygon * 0

    # np.savetxt('test.txt', connected_components, fmt='%d')
    cv2.imshow("Detect labels image", debug_image)

    return connected_components, symbols


def classify_edges(pixels, vertices_pixels):
    skel = pixels.copy()

    thinning.zhang_and_suen_binary_thinning(skel)
    cv2.imshow("Skel", np.array(skel * 255, dtype=np.uint8))

    classified_pixels, port_pixels = ogr.edge_classification(skel, vertices_pixels)

    # Clean Up single port pixels
    # rows, cols = classified_pixels.shape
    # port_pixels_ = set(port_pixels)
    # for x in range(0, rows):
    #     for y in range(0, cols):
    #         # Port pixels
    #         if classified_pixels[x, y] == 4:
    #             should_remove = True
    #             for pixel in ogr.get_eight_neighborhood(classified_pixels, x, y):
    #                 if pixel > 1:
    #                     should_remove = False
    #                     break
    #             if should_remove:
    #                 port_pixels_.remove((x, y))
    #                 classified_pixels[x, y] = 1
    # port_pixels = list(port_pixels_)

    R = np.zeros((pixels.shape), dtype=np.uint8)
    G = np.zeros((pixels.shape), dtype=np.uint8)
    B = np.zeros((pixels.shape), dtype=np.uint8)
    for x in range(classified_pixels.shape[0]):
        for y in range(classified_pixels.shape[1]):
            if classified_pixels[x, y] == 2:
                B[x, y] = 1  # edge pixels
            elif classified_pixels[x, y] == 3:
                G[x, y] = 1  # crossing pixels
            elif classified_pixels[x, y] == 4:
                R[x, y] = 1  # port pixels

    R = (255 * R)
    G = (255 * G)
    B = (255 * B)
    bgr = np.dstack((B, G, R)).astype(np.uint8)

    cv2.imshow("Classified Pixels", bgr)

    return classified_pixels, port_pixels


i1 = 0

class Graph:
    nodes = []

    class Node:
        def __init__(self, index, type):
            self.index = index
            self.edges = []
            self.type = type

        def add_edge(self, node):
            self.edges.append(node)

        def eq(self):
            if self.type == Node.Type.OUTPUT:
                # assuming only one node
                child = self.edges[0]
                return ast.Equation(self.edges[0].eq())
            elif self.type == Node.Type.VARIABLE:
                return ast.Term(str(self.index))
            elif self.type == Node.Type.OPERATOR:
                if len(self.edges) == 2:
                    return ast.BinaryOperationExpression(ast.OperationExpression.Op.AND, self.edges[0].eq(), self.edges[1].eq())
                else:
                    return ast.UnaryOperationExpression(ast.OperationExpression.Op.NOT, self.edges[0].eq())


def get_eq_tree(connected_components, symbols, dict_edge_sections):
    indexes = symbols.keys()
    nodes = {}

    for (pos_u, pos_v) in dict_edge_sections:
        u = connected_components[pos_u[0], pos_u[1]]
        v = connected_components[pos_v[0], pos_v[1]]

        if u not in nodes:
            nodes[u] = Graph.Node(u, symbols[u][0])
        if v not in nodes:
            nodes[v] = Graph.Node(v, symbols[v][0])

        node1 = nodes[u]
        node2 = nodes[v]

        # print(u, v)
        # print(pos_u, pos_v)

        if pos_u[0] < pos_v[0]:
            node2.add_edge(node1)
        else:
            node1.add_edge(node2)

    root = None
    for i in nodes.keys():
        if nodes[i].type == Node.Type.OUTPUT:
            root = nodes[i]
            break

    return root.eq()


def main():
    from PIL import Image

    image = cv2.imread("fuzzy.png", 1)
    # image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)

    cv2.imshow("Original", image)

    pixels, vertices_pixels = preprocessing(image)
    connected_components, symbols = detect_labels(image, vertices_pixels)

    # binarizy
    pixels = np.array(pixels > 1, dtype=np.uint32)
    vertices_pixels = np.array(vertices_pixels > 1, dtype=np.uint8)

    classified_pixels, port_pixels = classify_edges(pixels, vertices_pixels)

    trivial_sections,\
        port_sections,\
        crossing_pixels_in_port_sections,\
        last_gradients = ogr.edge_sections_identify(
            classified_pixels, port_pixels)
    merged_sections = ogr.traversal_subphase(classified_pixels,
                                             crossing_pixels_in_port_sections,
                                             last_gradients)

    edge_sections = trivial_sections + merged_sections
    dict_edge_sections = ogr.get_dict_edge_sections(edge_sections, vertices_pixels)

    # print(edge_sections)
    # print(dict_edge_sections)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # return

    equal = get_eq_tree(connected_components, symbols, dict_edge_sections)

    visitor = ast.GraphvizVisitor()
    g = visitor.visit(equal)

    # from graphviz import Graph
    # G = Graph('eq')

    # for u in symbols.keys():
    #     G.node(str(u))


    # print("Getting edges extremes and adding edges in the graph.")
    # for (pos_u, pos_v) in dict_edge_sections:
    #     u = connected_components[pos_u[0], pos_u[1]]
    #     v = connected_components[pos_v[0], pos_v[1]]
    #     G.edge(str(u), str(v))

    g.format = 'PNG'
    g.render()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
