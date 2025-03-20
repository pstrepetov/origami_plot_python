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
    vertex_params: dict = {"marker":"o", "alpha":0.9, "linewidth":1.0},
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

    Parameters:
        outer_radii_list (list of lists or array-like): 
            List of outer radii for each star. Each element must have the same length, representing the number of vertices.
        inner_radius (float, optional): 
            Radius of the inner points of the stars. Default is 0.1.
        edge_color_list (list[str], optional): 
            Colors of the star outlines. Must match the number of stars. If not provided, defaults to distinct colors from a colormap.
        edge_linestyle_list (list[str], optional): 
            Line styles for the star outlines. Must match the number of stars. Defaults to solid lines if not provided.
        edge_linewidth (float, optional): 
            Width of the star outlines. Default is 0.5.
        fill (bool, optional): 
            Whether to fill the stars with color. Default is True.
        fill_color_list (list[str], optional): 
            Fill colors for the stars. Must match the number of stars. If not provided, defaults to distinct colors from a colormap.
        alpha_list (list[float], optional): 
            Transparency levels for the stars. Must match the number of stars. Defaults to a predefined value if not provided.
        spider_net_radii (list or array-like, optional): 
            Radii for the spider net background polygons. If not provided, no spider net is drawn.
        spider_net_params (dict, optional): 
            Additional styling parameters for the spider net lines. Default is {"color": "gray", "alpha": 0.3, "linewidth": 1}.
        spider_net_text_params (dict, optional): 
            Styling parameters for the spider net radius labels. Default is {"fontsize": 8, "color": "gray", "ha": "center", "va": "center"}.
        vertex_names (list[str], optional): 
            Labels for each vertex. Must have the same length as the number of vertices. If not provided, no vertex names are displayed.
        vertex_names_params (dict, optional): 
            Styling parameters for the vertex name labels. Default is {"fontsize": 10, "color": "black", "ha": "center", "va": "center"}.
        vertex_params (dict, optional): 
            Styling parameters for the vertex markers. Default is {"marker": "o", "alpha": 0.9, "linewidth": 1.0}.
        vertex_names_position (tuple[float], optional):
            A tuple containing two parameters responsible for vertex name positioning. 
            - The first parameter (multiplicative factor) scales the radial distance of the vertex names from the center.
            - The second parameter (additive offset) adjusts the vertical position of the vertex names.
            Defaults to (1.05, 0.05).
        star_labels (list[str], optional): 
            Labels for each star in the legend. Must match the number of stars. If not provided, default labels are generated.
        star_params (dict, optional): 
            Additional styling parameters for the stars. Default is an empty dictionary.
        legend_params (dict, optional): 
            Styling parameters for the legend. Default is {"loc": "best", "fontsize": 10, "frameon": True}.
        figsize (tuple, optional): 
            The width and height of the plot in inches. Default is (8, 8).
        title (str, optional): 
            Title of the plot. If not provided, no title is displayed.
        title_params (dict, optional): 
            Styling parameters for the title. Default is an empty dictionary.

    Raises:
        ValueError: 
            - If the number of vertices (N) is less than 3.
            - If the lengths of `outer_radii_list` elements do not match.
            - If the lengths of `edge_color_list`, `fill_color_list`, `alpha_list`, `edge_linestyle_list`, or `star_labels` do not match the number of stars.

    Notes:
        - The function calculates and displays the area of each star in the legend.
        - The spider net background consists of concentric polygons and radial lines.
        - Vertex names are placed slightly outside the outermost polygon for better visibility.
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
        edge_color_list = plt.cm.tab10.colors[:num_stars]  # Use distinct colors from colormap
    if fill_color_list is None:
        fill_color_list = plt.cm.tab10.colors[:num_stars]
    if alpha_list is None:
        alpha_list = [DEFAULT_ALPHA] * num_stars
    if edge_linestyle_list is None:
        edge_linestyle_list = ['-' for _ in range(num_stars)]  # Default solid line
    
    # Validate lengths of lists
    if len(edge_color_list) != num_stars:
        raise ValueError(f"Length of color_list ({len(edge_color_list)}) must match the number of stars ({num_stars}).")
    if len(fill_color_list) != num_stars:
        raise ValueError(f"Length of fill_color_list ({len(fill_color_list)}) must match the number of stars ({num_stars}).")
    if len(alpha_list) != num_stars:
        raise ValueError(f"Length of alpha_list ({len(alpha_list)}) must match the number of stars ({num_stars}).")
    if len(edge_linestyle_list) != num_stars:
        raise ValueError(f"Length of linestyle_list ({len(edge_linestyle_list)}) must match the number of stars ({num_stars}).")
    if star_labels is not None and len(star_labels) != num_stars:
        raise ValueError(f"Length of star_labels ({len(star_labels)}) must match the number of stars ({num_stars}).")
    
    # Generate angles for the star points
    angles = np.linspace(0, 2 * np.pi, 2 * N, endpoint=False)
    
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
    
    # Plot each star and calculate its area
    legend_handles = []
    for i, outer_radii in enumerate(outer_radii_list):
        # Alternate between outer and inner points for the star
        radii = []
        for j in range(2 * N):
            if j % 2 == 0:  # Outer point
                radii.append(outer_radii[j // 2])
            else:  # Inner point
                radii.append(inner_radius)
        
        # Compute the x and y coordinates of the star points
        x_star = np.array(radii) * np.cos(angles)
        y_star = np.array(radii) * np.sin(angles)
        
        # Close the star by repeating the first point
        x_star = np.append(x_star, x_star[0])
        y_star = np.append(y_star, y_star[0])
        
        # Calculate the area of the star
        area = calculate_polygon_area(x_star, y_star)
        
        # Create label for the legend
        if star_labels:
            label = f"{star_labels[i]} (Area: {area:.2f})"
        else:
            label = f"Star {i+1} (Area: {area:.2f})"
        
        # Plot the star
        if fill:
            star_patch = ax.fill(x_star, y_star, color=fill_color_list[i], edgecolor=edge_color_list[i], 
                                 linewidth=edge_linewidth, linestyle=edge_linestyle_list[i], alpha=alpha_list[i],
                                 label=label, **star_params)[0]
        else:
            star_patch = ax.plot(x_star, y_star, color=edge_color_list[i], linewidth=edge_linewidth, 
                                 linestyle=edge_linestyle_list[i], alpha=alpha_list[i],
                                 label=label, **star_params)[0]
        # Create vertex dots
        ax.scatter(x_star[::2], y_star[::2], color=edge_color_list[i], **vertex_params)

        
        # Add to legend handles
        legend_handles.append(star_patch)
    
    # Add vertex names
    if vertex_names is not None:
        for i, name in enumerate(vertex_names):
            x_label = max(spider_net_radii) * np.cos(angles[::2][i]) * vertex_names_position[0] # Offset slightly outward
            y_label = max(spider_net_radii) * np.sin(angles[::2][i]) * vertex_names_position[0] # Offset slightly outward
            y_label += vertex_names_position[1] * np.sign(y_label - 1e-6) # shift vertex names a bit
            ax.text(x_label, y_label, name, **vertex_names_params)
    
    # Add legend
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
