from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np

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
    edge_color_list: list[str] = None,
    edge_linestyle_list: list[str] = None,
    edge_linewidth: float = 0.5,
    fill: bool = True,
    fill_color_list: list[str] = None,
    alpha_list: list[float] = None,
    spider_net_radii: list[float] = None,
    spider_net_params: dict = {"color":"gray", "alpha":0.3, "linewidth": 1},
    spider_net_text_params: dict = {"fontsize":8, "color":"gray", "ha":"center", "va":"center"},
    vertex_names: list[str] = None,
    vertex_names_params: dict = {"fontsize":10, "color":"black",
                    "ha":"center", "va":"center"},
    vertex_params: dict = {"marker":"o", "alpha":0.99, "linewidth":1.0, "zorder":2.5},
    vertex_names_position: tuple[float] = (1.05, 0.05),
    star_labels: list[str] = None,
    star_params: dict = {},
    legend_params: dict = {"loc":"best", "fontsize":10, "frameon":True},
    figsize: tuple = (8,8),
    title: str = None,
    title_params: dict = {"fontsize":15},
):
    """
    Plots multiple stars with custom outer radii for each star.
    Adds a spider net background with concentric polygons at specified radii.
    Optionally adds vertex names, a legend with star areas, and spider net radius labels to the plot.
    Handles intersections and colors them conditionally based on fill_color_list.
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
        alpha_list = [0.5] * num_stars
    if edge_linestyle_list is None:
        edge_linestyle_list = ['-' for _ in range(num_stars)]
    
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
        x_star = np.array(radii) * np.cos(angles)
        y_star = np.array(radii) * np.sin(angles)
        areas.append(calculate_polygon_area(x_star, y_star))
        star_polygons.append(Polygon(zip(x_star, y_star)))
    
    non_overlapping_regions = []
    intersections = []
    # Compute non-overlapping regions with fill
    if not fill:
        non_overlapping_regions = [(i, star_polygons[i]) for i in range(num_stars)]
    else:
        none_colored_polygons_idx = [i for i in range(num_stars) if fill_color_list[i].lower() == 'none']
        colored_polygons_idx = [i for i in range(num_stars) if fill_color_list[i].lower() != 'none']

        if len(none_colored_polygons_idx) != 0 and len(colored_polygons_idx) != 0:
            none_colored_union = unary_union([star_polygons[i] for i in none_colored_polygons_idx])

            for idx, star_polygon in [(i, star_polygons[i]) for i in colored_polygons_idx]:
                non_overlapping_region = star_polygon.difference(none_colored_union)
                non_overlapping_regions.append((idx, non_overlapping_region))
        else:
            non_overlapping_regions = [(i, star_polygons[i]) for i in range(num_stars)]
    
    # Plot the spider net background
    fig, ax = plt.subplots(figsize=figsize)
    if spider_net_radii is not None:
        for radius in spider_net_radii:
            x_polygon = radius * np.cos(angles)
            y_polygon = radius * np.sin(angles)
            ax.plot(np.append(x_polygon, x_polygon[0]), np.append(y_polygon, y_polygon[0]), **spider_net_params)
        
        # Add radial lines from the center to the outer vertices
        for angle in angles[::2]:
            ax.plot([0, max(spider_net_radii) * np.cos(angle)],
                    [0, max(spider_net_radii) * np.sin(angle)],
                    **spider_net_params)
        
        # Add radial lines from the center to the concave vertices
        middle_lines_params = spider_net_params.copy()
        middle_lines_params["linestyle"] = "-." if middle_lines_params.get("linestyle") != "-." else "--"
        for angle in angles[1::2]:
            ax.plot([0, max(spider_net_radii) * np.cos(angle)],
                    [0, max(spider_net_radii) * np.sin(angle)],
                    **middle_lines_params)
            
        # draw dots for vertex names
        dots_params = spider_net_params.copy()
        marker = middle_lines_params.get("marker")
        dots_params["marker"] = "s" if marker is None else marker
        ax.scatter([max(spider_net_radii) * np.cos(angle) for angle in angles[::2]],
                    [max(spider_net_radii) * np.sin(angle) for angle in angles[::2]],
                    **dots_params)
        
        # Add spider net radius labels between 1st and 2nd radial lines
        label_angle = angles[1]  # Angle for the radial line where labels will be placed
        for radius in spider_net_radii[:-1]:
            label_x = radius * np.cos(label_angle)
            label_y = radius * np.sin(label_angle)
            ax.text(label_x, label_y, f"{radius:.1f}", **spider_net_text_params)

    legend_handles = []
    legend_idxs = []
    if star_labels:
        labels = [f"{star_labels[i]} (Area: {areas[i]:.2f})" for i in range(num_stars)]
    else:
        labels = [f"Star {i+1} (Area: {areas[i]:.2f})" for i in range(num_stars)]

    # Plot non-overlapping regions without edges
    for i, region in non_overlapping_regions:
    
        xy = get_xy_from_polygon(region)
        for x,y in xy:
            if fill:
                star_patch = ax.fill(x, y, color=fill_color_list[i], edgecolor="none", alpha=alpha_list[i],
                                    label=labels[i], **star_params)[0]

        if i not in legend_idxs:
            star_patch.set(edgecolor=edge_color_list[i], linewidth=edge_linewidth, linestyle=edge_linestyle_list[i])
            legend_handles.append(star_patch)
            legend_idxs.append(i)


    # Draw polygons without filling
    for i, star_polygon in enumerate(star_polygons):
        
        xy = get_xy_from_polygon(star_polygon)
        for x,y in xy:
            star_patch = ax.fill(x, y, color="none", edgecolor=edge_color_list[i], linewidth=edge_linewidth,
                                 linestyle=edge_linestyle_list[i], alpha=alpha_list[i],
                                 label=labels[i], **star_params)[0]

        if fill_color_list[i] == "none" and i not in legend_idxs:
            legend_handles.append(star_patch)
            legend_idxs.append(i)


    # Create vertex dots
    for i in range(num_stars):
        xy = get_xy_from_polygon(star_polygons[i])
        for x,y in xy:
            ax.scatter(x[::2], y[::2], color=edge_color_list[i], **vertex_params)

    # Add vertex names
    if vertex_names is not None:
        for i, name in enumerate(vertex_names):
            x_label = max(spider_net_radii) * np.cos(angles[::2][i]) * vertex_names_position[0] # Offset slightly outward
            y_label = max(spider_net_radii) * np.sin(angles[::2][i]) * vertex_names_position[0] # Offset slightly outward
            y_label += vertex_names_position[1] * np.sign(y_label - 1e-6) # shift vertex names a bit
            ax.text(x_label, y_label, name, **vertex_names_params)
    
    # Add legend
    handles=[legend_handles[legend_idxs.index(i)] for i in range(num_stars)]
    ax.legend(handles=handles, **legend_params)
    
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
