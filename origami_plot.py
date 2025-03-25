from shapely.geometry import Polygon
from shapely import intersection_all
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

DEFAULT_ALPHA = 0.25

def calculate_polygon_area(x, y):
    """
    Calculate the area of a polygon given its vertices.

    Parameters:
        x (list or array-like): x-coordinates of the polygon vertices.
        y (list or array-like): y-coordinates of the polygon vertices.

    Returns:
        float: Area of the polygon.
    """
    x = np.array(x)
    y = np.array(y)
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def get_xy_from_polygon(region):
    """
    Extracts the x and y coordinates of the exterior boundaries of a polygon or multi-polygon.

    Parameters:
        region (Polygon or MultiPolygon): A polygon or multi-polygon object from the `shapely` library.

    Returns:
        list of tuples: A list where each tuple contains two elements:
                        - The first element is an array-like object of x-coordinates.
                        - The second element is an array-like object of y-coordinates.
                        Each tuple corresponds to the exterior boundary of a polygon in the input region.
    """
    xy = []
    if isinstance(region, Polygon):
        xy.append(tuple(region.exterior.xy))
    elif hasattr(region, 'geoms'):
        for geom in region.geoms:
            xy.append(tuple(geom.exterior.xy))
    return xy

def origami_plot(
    outer_radii_list,
    inner_radius: float = 0.1,
    edge_color_list: list[str] | None = None,
    edge_linestyle_list: list[str] | None = None,
    edge_linewidth: float = 0.5,
    fill: bool = True,
    fill_color_list: list[str] | None = None,
    alpha_list: list[float] | None = None,
    spider_net_radii: list[float] | None = None,
    spider_net_radii_names: list[str] | None = None,
    spider_net_params: dict = {"color":"gray", "alpha":0.3, "linewidth": 1},
    spider_net_text_params: dict = {"fontsize":8, "color":"gray", "ha":"center", "va":"center"},
    vertex_names: list[str] | None = None,
    vertex_names_params: dict = {"fontsize":10, "color":"black",
                    "ha":"center", "va":"center"},
    vertex_params: dict = {"marker":"o", "alpha":0.99, "linewidth":1.0, "zorder":2.5},
    vertex_names_position: tuple[float] = (1.05, 0.05),
    star_labels: list[str] | None = None,
    star_params: dict = {},
    add_legend: bool = True,
    legend_params: dict = {"loc":"best", "fontsize":10, "frameon":True},
    figsize: tuple = (8,8),
    title: str | None = None,
    title_params: dict = {"fontsize":15},
    rescaling_radius_func: Callable = lambda x: x,
    intersection_params = {"color": "cornsilk", "edgecolor": "none", "alpha": 0.4}
):
    """
    Plots multiple stars with custom outer radii for each star.
    Adds a spider net background with concentric polygons at specified radii.
    Optionally adds vertex names, a legend with star areas, and spider net radius labels to the plot.
    Handles intersections and colors them conditionally based on fill_color_list.

    Parameters:
        outer_radii_list (list of lists): List of lists, where each sublist specifies the outer radii
                                          for the vertices of a star-shaped polygon.
        inner_radius (float, optional): Radius of the inner vertices of the star. Default is 0.1.
        edge_color_list (list of str, optional): Colors for the edges of each star. Default uses `tab10` colormap.
        edge_linestyle_list (list of str, optional): Linestyles for the edges of each star. Default is solid lines.
        edge_linewidth (float, optional): Width of the edges. Default is 0.5.
        fill (bool, optional): Whether to fill the stars with color. Default is True.
        fill_color_list (list of str, optional): Fill colors for the stars. Default uses `tab10` colormap.
        alpha_list (list of float, optional): Transparency levels for the stars. Default is 0.5 for all stars.
        spider_net_radii (list of float, optional): Radii for the concentric circles in the spider net background.
        spider_net_radii_names (list of str, optional): Names of spider net radii.
        spider_net_params (dict, optional): Styling parameters for the spider net lines.
        spider_net_text_params (dict, optional): Styling parameters for the spider net radius labels.
        vertex_names (list of str, optional): Names for the vertices of the polygon.
        vertex_names_params (dict, optional): Styling parameters for the vertex names.
        vertex_params (dict, optional): Styling parameters for the vertex markers.
        vertex_names_position (tuple of float, optional): Position offsets for the vertex names.
        star_labels (list of str, optional): Labels for the stars in the legend.
        star_params (dict, optional): Additional styling parameters for the stars.
        add_legend (bool, optional): Whether to plot the legend. Default is True.
        legend_params (dict, optional): Styling parameters for the legend.
        figsize (tuple of float, optional): Size of the figure. Default is (8, 8).
        title (str, optional): Title of the plot.
        title_params (dict, optional): Styling parameters for the title.
        rescaling_radius_func (Callable, optional): Function to rescale radii. Default is identity function.
        intersection_params (dict, optional): Styling parameters for the global intersection region.

    Raises:
        ValueError: If the number of vertices is less than 3.
        ValueError: If the lengths of input lists do not match the number of stars.

    Returns:
        None: Displays the plot.
    """
    # Infer N (number of vertices) from the first star's outer radii
    N = len(outer_radii_list[0])
    if N < 3:
        raise ValueError("The number of vertices (N) must be at least 3.")
    for i, outer_radii in enumerate(outer_radii_list):
        if len(outer_radii) != N:
            raise ValueError(f"Length of outer_radii_list[{i}] ({len(outer_radii)}) does not match the number of vertices (N={N}).")

    num_stars = len(outer_radii_list)

    # Default values for lists
    if edge_color_list is None:
        edge_color_list = plt.cm.tab10.colors[:num_stars]
    if fill_color_list is None:
        fill_color_list = plt.cm.tab10.colors[:num_stars]
    if alpha_list is None:
        alpha_list = [DEFAULT_ALPHA] * num_stars
    if edge_linestyle_list is None:
        edge_linestyle_list = ['-' for _ in range(num_stars)]
    if not fill:
        fill_color_list = ["none" for i in range(num_stars)]

    # Validate lengths of lists
    if len(edge_color_list) != num_stars:
        raise ValueError(f"Length of edge_color_list ({len(edge_color_list)}) must match the number of stars ({num_stars}).")
    if len(fill_color_list) != num_stars:
        raise ValueError(f"Length of fill_color_list ({len(fill_color_list)}) must match the number of stars ({num_stars}).")
    if len(alpha_list) != num_stars:
        raise ValueError(f"Length of alpha_list ({len(alpha_list)}) must match the number of stars ({num_stars}).")
    if len(edge_linestyle_list) != num_stars:
        raise ValueError(f"Length of edge_linestyle_list ({len(edge_linestyle_list)}) must match the number of stars ({num_stars}).")

    # Generate angles for the star points
    angles = np.linspace(0, 2 * np.pi, 2 * N, endpoint=False)

    # Create polygons for all stars
    star_polygons = []
    areas = []
    for i, outer_radii in enumerate(outer_radii_list):
        radii = []
        for j in range(2 * N):
            if j % 2 == 0:  # Outer point
                radii.append(outer_radii[j // 2])
            else:  # Inner point
                radii.append(inner_radius)
        areas.append(calculate_polygon_area(np.array(radii) * np.cos(angles), np.array(radii) * np.sin(angles)))
        x_star = np.array(rescaling_radius_func(radii)) * np.cos(angles)
        y_star = np.array(rescaling_radius_func(radii)) * np.sin(angles)
        star_polygons.append(Polygon(zip(x_star, y_star)))

    # Create labels for legend
    if star_labels:
        labels = [f"{star_labels[i]} (Area: {areas[i]:.2f})" for i in range(num_stars)]
    elif add_legend:
        labels = [f"Star {i+1} (Area: {areas[i]:.2f})" for i in range(num_stars)]

    # Plot the spider net background
    fig, ax = plt.subplots(figsize=figsize)
    if spider_net_radii is not None:
        spn_radii = spider_net_radii.copy()
        spider_net_radii = list(rescaling_radius_func(spider_net_radii))
        for radius in spider_net_radii:
            x_polygon = radius * np.cos(angles)
            y_polygon = radius * np.sin(angles)
            ax.plot(np.append(x_polygon, x_polygon[0]), np.append(y_polygon, y_polygon[0]), **spider_net_params)

        # Add radial lines from the center to the outer vertices
        for angle in angles[::2]:
            ax.plot([0, max(spider_net_radii) * np.cos(angle)],
                    [0, max(spider_net_radii) * np.sin(angle)],
                    **spider_net_params)

        # Add dashed radial lines from the center to the concave vertices
        middle_lines_params = spider_net_params.copy()
        middle_lines_params["linestyle"] = "-." if middle_lines_params.get("linestyle") != "-." else "--"
        for angle in angles[1::2]:
            ax.plot([0, max(spider_net_radii) * np.cos(angle)],
                    [0, max(spider_net_radii) * np.sin(angle)],
                    **middle_lines_params)

        # Draw dots for vertex names
        dots_params = spider_net_params.copy()
        marker = middle_lines_params.get("marker")
        dots_params["marker"] = "s" if marker is None else marker
        ax.scatter([max(spider_net_radii) * np.cos(angle) for angle in angles[::2]],
                    [max(spider_net_radii) * np.sin(angle) for angle in angles[::2]],
                    **dots_params)

        # Add spider net radius labels between 1st and 2nd radial lines
        label_angle = angles[1]  # Angle for the radial line where labels will be placed
        for i, radius in enumerate(spider_net_radii):
            label_x = radius * np.cos(label_angle)
            label_y = radius * np.sin(label_angle)
            if spider_net_radii_names is not None:
                assert len(spider_net_radii_names) == len(spider_net_radii)
                text = spider_net_radii_names[i]
            else:
                text = f"{spn_radii[i]:.2f}"
            ax.text(label_x, label_y, text, **spider_net_text_params)

    # Plot global intersection without edges
    legend_handles = [] # list with pathces for legend
    intersection_global = intersection_all(star_polygons)
    xy = get_xy_from_polygon(intersection_global)
    for x,y in xy:
        intrsct = ax.fill(x, y, **intersection_params)[0]
        if "label" in intersection_params:
            legend_handles.append(intrsct)

    # Plot local intersections with filling, but without edges
    for i in range(N):
        radii_i = [orl[i] for orl in outer_radii_list]
        indexes = [j for j in range(num_stars)]
        sorted_indexes = sorted(indexes, key=lambda x: radii_i[x]) #sorting indexes by radius
        if num_stars > 1:
            for num, idx in enumerate(sorted_indexes[1:]):
                cur_star = star_polygons[idx]

                prev_x = star_polygons[sorted_indexes[num]].exterior.xy[0][i*2]
                prev_y = star_polygons[sorted_indexes[num]].exterior.xy[1][i*2]

                cur_x = cur_star.exterior.xy[0][i*2]
                cur_y = cur_star.exterior.xy[1][i*2]

                v1_x = cur_star.exterior.xy[0][i*2 - 1]
                v1_y = cur_star.exterior.xy[1][i*2 - 1]
                v2_x = cur_star.exterior.xy[0][i*2 + 1]
                v2_y = cur_star.exterior.xy[1][i*2 + 1]

                local_x = [prev_x, v1_x, cur_x, v2_x]
                local_y = [prev_y, v1_y, cur_y, v2_y]

                ax.fill(local_x, local_y, color=fill_color_list[idx], edgecolor="none",
                        alpha=alpha_list[idx], **star_params)

    # Draw polygons without filling
    for i, star_polygon in enumerate(star_polygons):
        xy = get_xy_from_polygon(star_polygon)
        for x,y in xy:
            star_patch = ax.fill(x, y, color="none", edgecolor=edge_color_list[i], linewidth=edge_linewidth,
                                 linestyle=edge_linestyle_list[i], alpha=alpha_list[i],
                                 label=labels[i], **star_params)[0]

    # Create dots for outer vertices
    for i in range(num_stars):
        xy = get_xy_from_polygon(star_polygons[i])
        for x,y in xy:
            ax.scatter(x[::2], y[::2], color=edge_color_list[i], **vertex_params)

    # Add vertex names
    if vertex_names is not None:
        for i, name in enumerate(vertex_names):
            x_label = max(spider_net_radii) * np.cos(angles[::2][i]) * vertex_names_position[0] # Offset slightly outward
            y_label = max(spider_net_radii) * np.sin(angles[::2][i]) * vertex_names_position[0] # Offset slightly outward
            y_label += vertex_names_position[1] * np.sign(y_label - 1e-8) # shift vertex names a bit to avoid 0 sign
            ax.text(x_label, y_label, name, **vertex_names_params)

    # Add legend
    if add_legend:
        for i in range(num_stars):
            proxy_artist = Patch(facecolor=fill_color_list[i], edgecolor=edge_color_list[i],
                                linewidth=edge_linewidth, linestyle=edge_linestyle_list[i],
                                alpha=alpha_list[i], label=labels[i], **star_params)
            legend_handles.append(proxy_artist)

        ax.legend(handles=legend_handles, **legend_params)

    # Set aspect ratio and limits
    max_radius = max([max(outer_radii + [inner_radius]) for outer_radii in outer_radii_list] +
                     (spider_net_radii if spider_net_radii else []))
    ax.set_aspect('equal')
    ax.set_xlim(-max_radius * 2, max_radius * 2)
    ax.set_ylim(-max_radius * 1.1, max_radius * 1.1)
    ax.axis('off')  # Hide axes

    if title is not None:
        ax.set_title(title, **title_params)

    plt.show()
