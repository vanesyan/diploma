import numpy as np

import morphology
from circular_list import CircularList
from collections import Counter


def edge_classification(skel, vertices_pixel):
    height, width = skel.shape

    classified_pixels = np.zeros((skel.shape))
    port_pixels = []

    for x in range(1, height - 1):
        for y in range(1, width - 1):

            if skel[x, y] == 1 and vertices_pixel[x, y] == 0:

                # eight neighborhood of pixel (x, y)
                skel_neighborhood = get_eight_neighborhood(skel, x, y)

                vertex_neighborhood = get_eight_neighborhood(vertices_pixel, x,
                                                             y)
                vertex_neighborhood_in_skel = np.logical_and(skel_neighborhood,
                                                             vertex_neighborhood)

                # n0 is the number of object pixels in 8-neighborhood of (x,y)
                n0 = np.sum(skel_neighborhood)

                if n0 < 2:
                    classified_pixels[x, y] = 1  # miscellaneous pixels
                elif n0 == 2 and np.any(vertex_neighborhood_in_skel):
                    classified_pixels[x, y] = 4  # port pixels
                    port_pixels.append((x, y))
                elif n0 == 2:
                    classified_pixels[x, y] = 2  # edge pixels
                elif n0 > 2:
                    classified_pixels[x, y] = 3  # crossing pixels

    return classified_pixels, port_pixels  # , crossing_pixels


def get_eight_neighborhood(pixels, x, y):
    neighborhood = np.array(
        [pixels[x - 1, y], pixels[x + 1, y], pixels[x, y - 1], pixels[x, y + 1],
         pixels[x + 1, y + 1], pixels[x + 1, y - 1], pixels[x - 1, y + 1],
         pixels[x - 1, y - 1]])
    return neighborhood


def edge_sections_identify(classified_pixels, port_pixels):
    trivial_sections = []
    port_sections = []
    crossing_sections = []

    start_pixels = {}
    start_pixels = dict.fromkeys(port_pixels, 0)

    last_gradients = {}
    crossing_pixels_in_port_sections = {}

    # dictionary of predecessors de crossing pixels
    back_port_sections = {}

    for start in start_pixels:

        # if port pixel is already visited, then continue
        if start_pixels[start] == 1:
            continue
        else:
            # marks start pixel as already visited
            start_pixels[start] = 1
            section, last_gradients[start], next = get_basic_section(start,
                                                                     classified_pixels)

        next_value = classified_pixels[next[0], next[1]]

        if next_value == 4:  # port pixel

            # marks the next pixel as already visited
            start_pixels[(next[0], next[1])] = 1

            trivial_sections.append(section)

        elif next_value == 3:  # crossing pixel

            port_sections.append(section)

            pos = (next[0], next[1])

            if not pos in crossing_pixels_in_port_sections:
                crossing_pixels_in_port_sections[pos] = []

            crossing_pixels_in_port_sections[pos].append([section, 0])

    # clear dictionary of start pixels
    start_pixels.clear()

    return trivial_sections, port_sections, crossing_pixels_in_port_sections, last_gradients


def traversal_subphase(classified_pixels, crossing_pixels_in_port_sections,
                       last_gradients):
    merged_sections = []

    for crossing_pixel in crossing_pixels_in_port_sections:
        for info_section in crossing_pixels_in_port_sections[crossing_pixel]:

            # if crossing pixel is already visited, then continue
            if info_section[1] == 1:
                continue

            section = info_section[0]
            start_pixel = crossing_pixel

            flag_found_section = False
            iteration = 0

            while not flag_found_section:
                crossing_section_direction = get_crossing_section_direction(
                    classified_pixels, start_pixel, last_gradients[section[0]],
                    section)

                flag_found_section = merge_sections(
                    crossing_pixels_in_port_sections, section,
                    crossing_section_direction, merged_sections)

                if not flag_found_section:
                    if len(crossing_section_direction) > 1:
                        # start_back is a crossing pixel
                        start_back = crossing_section_direction[-2]
                    else:
                        start_back = start_pixel

                    # next is an edge pixel
                    next = crossing_section_direction[-1]

                    _section, _last_gradients, next = get_basic_section(next,
                                                                        classified_pixels,
                                                                        start_back)
                    crossing_section_direction.extend(_section[1:])

                    _crossing_section_direction = get_crossing_section_direction(
                        classified_pixels, _section[-1],
                        last_gradients[section[0]], _section)
                    crossing_section = crossing_section_direction + _crossing_section_direction

                    flag_found_section = merge_sections(
                        crossing_pixels_in_port_sections, section,
                        crossing_section, merged_sections)

                    if not flag_found_section:
                        # start pixel is a crossing pixel
                        start_pixel = crossing_section[-2]
                        section = section + crossing_section

                    # if iteration == 1:
                    #   _last_gradients.clear()
                    #   del _last_gradients
                    # else:
                    last_gradients[section[0]].extend(_last_gradients)

                iteration += 1

    return merged_sections


def merge_sections(crossing_pixels_in_port_sections, section, crossing_section,
                   merged_sections):
    if len(crossing_section) > 1:
        # back is a crossing pixel
        back = crossing_section[-2]
    else:
        back = section[-1]

    key_back = (back[0], back[1])

    # next is an edge pixel
    next = crossing_section[-1]

    if key_back not in crossing_pixels_in_port_sections:
        return False

    for info_section in crossing_pixels_in_port_sections[key_back]:
        _section = info_section[0]

        if next[0] == _section[-2][0] and next[1] == _section[-2][1]:
            merged_sections.append(
                section + crossing_section + _section[::-1][1:])

            # mark back (crossing pixel) as already visited
            info_section[1] = 1

            return True

    return False


def get_crossing_section_direction(classified_pixels, crossing_pixel,
                                   last_gradients, section):
    # counter gradients frequency
    cnt_gradient = Counter(last_gradients.get_list())
    # count in list
    grads = cnt_gradient.most_common()

    crossing_section_direction = []

    next = crossing_pixel
    next_value = classified_pixels[next[0], next[1]]

    # back is a edge pixel
    back = section[-2][0], section[-2][1]

    # avoid local minima
    iterations = 0
    loop_grads = CircularList(3)
    excluded_grad = None

    while next_value != 2:  # edge pixel
        aux_value = 0
        i = 0

        if iterations == 3:
            list_loop_grads = loop_grads.get_list()
            excluded_grad = list_loop_grads[1]
            crossing_section_direction[:] = []
            iterations = 0

        # blank pixel or miscellaneous and i < len
        while aux_value < 2 and i < len(grads):
            if grads[i][0] == excluded_grad:
                continue

            delta = grads[i][0]
            aux = np.add(next, delta)

            if aux[0] == back[0] and aux[1] == back[
                    1]:  # back[0] >= 0 and back[1] >= 0 and
                aux_value = 0
            else:
                aux_value = classified_pixels[aux[0], aux[1]]

            i += 1

        if aux_value < 2 and i == len(grads):
            delta = get_direction(classified_pixels, back, next, grads,
                                 excluded_grad)

            loop_grads.insert(delta)
            back = next[0], next[1]
            next = np.add(next, delta)
            next_value = classified_pixels[next[0], next[1]]
        else:
            loop_grads.insert(delta)
            back = next[0], next[1]
            next = aux
            next_value = aux_value

        crossing_section_direction.append(next)
        iterations += 1

    return crossing_section_direction


def get_basic_section(start, classified_pixels, start_back=None):
    # 'gradient' vector
    delta = np.array([0, 0])

    last_gradients = CircularList()

    section = []
    section.append(start)

    x, y = start
    next, next_value = get_max_neighbor(classified_pixels, x, y, start_back)
    delta = np.subtract(next, start)

    while next_value == 2:  # edge pixel
        last_gradients.insert((delta[0], delta[1]))
        section.append(next)

        next = np.add(next, delta)
        next_value = classified_pixels[next[0], next[1]]

        if next_value < 2:  # blank pixel or miscellaneous pixel
            last = section[-1]  # get last element added in section
            x, y = last
            back = np.subtract(last, delta)

            # get max value in the neighborhood, unless the 'back'
            next, next_value = get_max_neighbor(classified_pixels, x, y, back)

            delta = np.subtract(next, last)

    last_gradients.insert((delta[0], delta[1]))
    section.append(next)
    last_element = next

    return section, last_gradients, last_element


# get max value in the neighborhood, unless the 'back'
def get_max_neighbor(classified_pixels, x, y, back=None):
    neighbor = None
    neighbor_value = -float('inf')

    for i in range(0, 3):
        for j in range(0, 3):
            if (back is None or (
                    x + i - 1 != back[0] or y + j - 1 != back[1])) and (
                    i != 1 or j != 1) and (
                    classified_pixels[x + i - 1, y + j - 1] > neighbor_value):
                neighbor = np.array([x + i - 1, y + j - 1])
                neighbor_value = classified_pixels[x + i - 1, y + j - 1]

    return neighbor, neighbor_value


def get_direction(classified_pixels, back, current, common_grads,
                 excluded_grad=None):
    possible_grads = {(0, 1), (1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1),
                      (0, -1), (-1, 0)}
    s_grads = [x[0] for x in common_grads]
    possible_grads = possible_grads - set(s_grads)

    if not excluded_grad is None:
        possible_grads = possible_grads - {excluded_grad}

    min_d = float('inf')
    min_grad = None

    for grad in possible_grads:
        aux = np.add(current, grad)

        sat_condition = (aux[0] != back[0] or aux[1] != back[1]) and (
            classified_pixels[aux[0], aux[1]] > 1)
        d = distance_heuristic_grads(common_grads, possible_grads, grad)

        if sat_condition and (d < min_d):
            min_d = d
            min_grad = grad

    return min_grad


def distance_heuristic_grads(common_grads, possible_grads, grad):
    n = 0.0
    average_common_grad = [0.0, 0.0]

    for _grad in common_grads:
        aux = [_grad[1] * z for z in _grad[0]]
        average_common_grad = map(sum, zip(average_common_grad, aux))
        n += _grad[1]

    average_common_grad = [z / n for z in average_common_grad]

    # determine weights to calculate distances
    amount_non_zero_x = 0
    amount_non_zero_y = 0

    for _grad in common_grads:
        if _grad[0][0] != 0:
            amount_non_zero_x += _grad[1]
        if _grad[0][1] != 0:
            amount_non_zero_y += _grad[1]

    total_non_zeros = amount_non_zero_x + amount_non_zero_y

    alpha = (total_non_zeros - amount_non_zero_y)
    betha = (total_non_zeros - amount_non_zero_x)

    d = weighted_euclidean_distance(average_common_grad, grad, alpha, betha)

    return d


def weighted_euclidean_distance(grad1, grad2, alpha=1, betha=1):
    [x, y] = [grad1[0] - grad2[0], grad1[1] - grad2[1]]
    d = np.sqrt((alpha * (x ** 2)) + (betha * (y ** 2)))
    return d


def get_dict_edge_sections(edge_sections, vertices_pixel):
    dict_edge_sections = {}

    for section in edge_sections:

        u_x, u_y = section[0]
        v_x, v_y = section[-1]

        u_vertex_neighbor_positions = [[u_x - 1, u_y - 1], [u_x - 1, u_y],
                                       [u_x - 1, u_y + 1], [u_x, u_y - 1],
                                       [u_x, u_y + 1], [u_x + 1, u_y - 1],
                                       [u_x + 1, u_y], [u_x + 1, u_y + 1]]
        v_vertex_neighbor_positions = [[v_x - 1, v_y - 1], [v_x - 1, v_y],
                                       [v_x - 1, v_y + 1], [v_x, v_y - 1],
                                       [v_x, v_y + 1], [v_x + 1, v_y - 1],
                                       [v_x + 1, v_y], [v_x + 1, v_y + 1]]

        pixel_u = (0, 0)
        pixel_v = (0, 0)

        for (i, j) in u_vertex_neighbor_positions:
            if vertices_pixel[i, j] > 0:
                pixel_u = (i, j)
                break

        for (i, j) in v_vertex_neighbor_positions:
            if vertices_pixel[i, j] > 0:
                pixel_v = (i, j)
                break

        dict_edge_sections[pixel_u, pixel_v] = section

    return dict_edge_sections
