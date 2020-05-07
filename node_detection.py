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
        UNDEFINED = 'undefined'
        OPERATOR = 'operator'
        VARIABLE = 'variable'
        OUTPUT = 'output'

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

    eps = 0  # int(EPS / 2)
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


def detect_shape(polygon) -> int:
    h, w = polygon.shape
    tbox = np.zeros((h, w), dtype=np.uint8)
    center = (int(w / 2), int(h / 2))
    r = int(min(w, h) / 2)

    # Circle shape
    tcbox = tbox.copy()
    tcbox = cv2.circle(tcbox, center, r, color=255, thickness=cv2.FILLED)

    # Rectangle shape
    ttbox = tbox.copy()
    ttbox[:, :] = 255

    # Triangle shape
    ttrbox = tbox.copy()
    ttrbox = cv2.drawContours(ttrbox, [np.array(
        [[0, 0], [int(w/2), h], [w, 0]], dtype=np.int32)], -1, color=255, thickness=cv2.FILLED)

    # Ellipse shape
    tebox = tbox.copy()
    tebox = cv2.ellipse(tebox, center, (int(w/2), int(h/2)),
                        0, 0, 360, color=255, thickness=cv2.FILLED)

    # cv2.imwrite("cbox.png", polygon)
    # cv2.imwrite("ttbox.png", ttbox)
    # cv2.imwrite("tcbox.png", tcbox)
    # cv2.imwrite("trtbox.png", ttrbox)
    # cv2.imwrite("tebox.png", tebox)

    tcdiff = np.bitwise_xor(tcbox, polygon)
    ttdiff = np.bitwise_xor(ttbox, polygon)
    ttrdiff = np.bitwise_xor(ttrbox, polygon)
    tediff = np.bitwise_xor(tebox, polygon)

    # cv2.imwrite("tcdiff.png", tcdiff)
    # cv2.imwrite("ttdiff.png", ttdiff)
    # cv2.imwrite("ttrdiff.png", ttrdiff)
    # cv2.imwrite("tediff.png", tediff)

    shapes = [Node.Type.OPERATOR, Node.Type.VARIABLE,
              Node.Type.VARIABLE, Node.Type.OUTPUT]

    index = np.argmin([
        np.sum(np.bitwise_xor(ttrbox, polygon) > 1, dtype=np.int32),
        np.sum(np.bitwise_xor(tcbox, polygon)
               > 1, dtype=np.int32),
        np.sum(np.bitwise_xor(tebox, polygon)
               > 1, dtype=np.int32),
        np.sum(np.bitwise_xor(ttbox, polygon) > 1, dtype=np.int32)
    ])

    print(shapes[index])

    return shapes[index]


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
        processed_pixels, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    cv2.imshow("Binarized image", processed_pixels)

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
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2, 2))
    processed_pixels = cv2.dilate(processed_pixels, kernel)

    cv2.imshow("Vertices pixels", vertices_pixels)
    cv2.imshow("Processed image", processed_pixels)

    return processed_pixels, vertices_pixels


def detect_labels(image, vertices_pixels):
    nodes, _ = cv2.findContours(
        vertices_pixels, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    symbols = {}  # symbols
    w, h, _ = image.shape
    test_polygon = np.zeros((w, h, 3), dtype=np.uint8)
    connected_components = np.zeros((w, h), dtype=np.uint32)

    debug_image = image.copy()
    processing_image = image.copy()

    processing_image = cv2.cvtColor(processing_image, cv2.COLOR_BGR2GRAY)
    _, processing_image = cv2.threshold(
        processing_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    result = vertices_pixels.copy()
    result = cv2.copyTo(processing_image, result)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(1, 1))
    result = cv2.erode(result, kernel)

    cv2.imshow("Masked", result)

    for index, node in enumerate(nodes):
        name = index+1
        x, y, w, h = cv2.boundingRect(node)

        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(debug_image, str(name), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        part = vertices_pixels[y:(y+h), x:(x+w)]
        node_type = detect_shape(part)

        processing_image = cv2.drawContours(processing_image, [node], 0,
                                            color=(255, 255, 255), thickness=10)
        test_polygon = cv2.drawContours(test_polygon, [node], 0,
                                        color=(1, 0, 0), thickness=cv2.FILLED)
        xa, ya = np.where(test_polygon[:, :, 0] == 1)

        # Create polygon of marked nodes
        connected_components[xa, ya] = name

        # Detect text
        rect = result[y:(y+h), x:(x+w)]
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
        mask = cv2.dilate(np.bitwise_not(
            vertices_pixels[y:(y+h), x:(x+w)]), kernel)
        rect = np.bitwise_or(rect, mask)
        edge_mask = np.full_like(rect, 255)
        edge_mask[1:-1, 1:-1] = 0
        rect = np.bitwise_or(rect, edge_mask)
        text = pytesseract.run_and_get_output(Image.fromarray(
            rect), extension='txt', lang='eng', config='--oem 1 --psm 10 tesseract_config.ini')

        print(text)

        if text == 'I':
            text = '|'

        symbols[name] = (node_type, text, x+w/2)

        # Clean Up
        test_polygon = test_polygon * 0

    # t = ''

    # for node in nodes:
    #     x, y, w, h = cv2.boundingRect(node)

    #     rect = result[y:(y+h), x:(x+w)]
    #     kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
    #     mask = cv2.dilate(np.bitwise_not(
    #         vertices_pixels[y:(y+h), x:(x+w)]), kernel)
    #     rect = np.bitwise_or(rect, mask)
    #     edge_mask = np.full_like(rect, 255)
    #     edge_mask[1:-1, 1:-1] = 0
    #     rect = np.bitwise_or(rect, edge_mask)
    #     text = pytesseract.run_and_get_output(Image.fromarray(rect), extension='txt', lang='eng', config='--oem 1 --psm 10 tesseract_config.ini')
    #     print(len(text))
    #     t += text

    # np.savetxt('test.txt', connected_components, fmt='%d')
    cv2.imshow("Detect labels image", debug_image)

    return connected_components, symbols


def classify_edges(pixels, vertices_pixels):
    skel = pixels.copy()

    thinning.zhang_and_suen_binary_thinning(skel)
    cv2.imshow("Skel", np.array(skel * 255, dtype=np.uint8))

    classified_pixels, port_pixels = ogr.edge_classification(
        skel, vertices_pixels)

    # Clean Up single port pixels
    rows, cols = classified_pixels.shape
    port_pixels_ = set(port_pixels)
    for x in range(0, rows):
        for y in range(0, cols):
            # Port pixels
            if classified_pixels[x, y] == 4:
                should_remove = True
                for pixel in ogr.get_eight_neighborhood(classified_pixels, x, y):
                    if pixel > 1:
                        should_remove = False
                        break
                if should_remove:
                    port_pixels_.remove((x, y))
                    classified_pixels[x, y] = 1
    port_pixels = list(port_pixels_)

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


class Graph:
    nodes = []

    class Node:
        def __init__(self, index, type, name, x):
            self.index = index
            self.edges = []
            self.type = type
            self.name = name
            self.x = x

        def add_edge(self, node):
            self.edges.append(node)

        def eq(self):
            if self.type == Node.Type.OUTPUT:
                # assuming only one node
                child = self.edges[0]
                return ast.Equation(self.edges[0].eq())
            elif self.type == Node.Type.VARIABLE:
                return ast.Term(str(self.name))
            elif self.type == Node.Type.OPERATOR:
                if len(self.edges) == 2:
                    if self.name == '&':
                        op = ast.OperationExpression.Op.AND
                    elif self.name == '|':
                        op = ast.OperationExpression.Op.OR

                    n1 = self.edges[0]
                    n2 = self.edges[1]
                    
                    if n1.x > n2.x:
                        n1, n2 = n2, n1

                    return ast.BinaryOperationExpression(op, n1.eq(), n2.eq())
                else:
                    if self.name == '!':
                        return ast.UnaryOperationExpression(ast.OperationExpression.Op.NOT, self.edges[0].eq())


def get_eq_tree(connected_components, symbols, dict_edge_sections):
    indexes = symbols.keys()
    nodes = {}

    for (pos_u, pos_v) in dict_edge_sections:
        u = connected_components[pos_u[0], pos_u[1]]
        v = connected_components[pos_v[0], pos_v[1]]

        if u not in nodes:
            nodes[u] = Graph.Node(u, symbols[u][0], symbols[u][1], symbols[u][2])
        if v not in nodes:
            nodes[v] = Graph.Node(v, symbols[v][0], symbols[v][1], symbols[v][2])

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

    image = cv2.imread("fuzzy.png")
    # image = cv2.resize(image, None, fx=0.5, fy=0.5,
    #                    interpolation=cv2.INTER_LANCZOS4)

    cv2.imshow("Original", image)

    pixels, vertices_pixels = preprocessing(image)
    connected_components, symbols = detect_labels(image, vertices_pixels)

    # binarizy
    pixels = np.array(pixels > 1, dtype=np.uint32)
    vertices_pixels = np.array(vertices_pixels > 1, dtype=np.uint8)

    classified_pixels, port_pixels = classify_edges(pixels, vertices_pixels)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # return

    print("Traversing graph")

    trivial_sections,\
        port_sections,\
        crossing_pixels_in_port_sections,\
        last_gradients = ogr.edge_sections_identify(
            classified_pixels, port_pixels)
    merged_sections = ogr.traversal_subphase(classified_pixels,
                                             crossing_pixels_in_port_sections,
                                             last_gradients)

    edge_sections = trivial_sections + merged_sections
    dict_edge_sections = ogr.get_dict_edge_sections(
        edge_sections, vertices_pixels)

    # # print(edge_sections)
    # # print(dict_edge_sections)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # return

    print("Getting eq")

    equal = get_eq_tree(connected_components, symbols, dict_edge_sections)

    image_visitor = ast.GraphvizVisitor()
    string_visitor = ast.StringVisitor()

    g = image_visitor.visit(equal)
    f = string_visitor.visit(equal)

    print(f)

    g.format = 'PNG'
    g.render()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
